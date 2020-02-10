#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author: Wei Wu
@license: Apache Licence
@file: corefqa_model.py
@time: 2019/12/23
@contact: wu.wei@pku.edu.cn
"""
import tensorflow as tf
from bert.modeling import BertModel, BertConfig
import utils


class CorefQAModel(object):

    def __init__(self, config):
        self.config = config
        self.bert_config = BertConfig.from_json_file(self.config.bert_config_file)

    def forward(self, features, is_training):
        doc_idx = features['doc_idx']  # (1,)
        sentence_map = features['sentence_map']  # (num_tokens, ) tokens to sentence id
        subtoken_map = features['subtoken_map']  # (num_tokens, ) tokens to word id
        flattened_input_ids = features['flattened_input_ids']  # (num_windows * window_size)
        flattened_input_mask = features['flattened_input_mask']  # (num_windows * window_size)
        span_starts = features['span_starts']  # (num_spans, ) span start indices
        span_ends = features['span_ends']  # (num_spans, ) span end indices
        cluster_ids = features['cluster_ids']  # (num_spans, ) span to cluster indices

        dropout = 1 - (tf.cast(is_training, tf.float32) * self.config.dropout_rate)
        bert_embeddings = self.get_bert_embeddings(flattened_input_ids, flattened_input_mask, is_training)  # (num_tokens, embed_size)
        candidate_starts, candidate_ends = self.get_candidate_spans(sentence_map)  # (num_candidates), (num_candidates)
        num_candidate_mentions = tf.cast(tf.shape(bert_embeddings)[0] * self.config.span_ratio, tf.int32)
        k = tf.minimum(self.config.max_candidate_num, num_candidate_mentions)
        c = tf.minimum(self.config.max_antecedent_num, k)
        top_span_scores, top_span_indices, top_span_starts, top_span_ends, top_span_emb = self.filter_by_mention_scores(
            bert_embeddings, candidate_starts, candidate_ends, dropout, k)
        top_span_cluster_ids = self.get_top_span_cluster_ids(candidate_starts, candidate_ends, span_starts, span_ends,
                                                             cluster_ids, top_span_indices)

        i0 = tf.constant(0)
        top_antecedent_starts = tf.zeros((0, c), dtype=tf.int32)
        top_antecedent_ends = tf.zeros((0, c), dtype=tf.int32)
        top_antecedent_labels = tf.zeros((0, c), dtype=tf.int32)
        top_antecedent_scores = tf.zeros((0, c), dtype=tf.float32)

        def qa_loop_body(i, starts, ends, labels, scores):
            input_ids = tf.reshape(flattened_input_ids,
                                   [-1, self.config.sliding_window_size])  # (num_windows, window_size)
            input_mask = tf.reshape(flattened_input_mask, [-1, self.config.sliding_window_size])
            actual_mask = tf.cast(tf.not_equal(input_mask, self.config.pad_idx), tf.int32)  # (num_windows, window_size)

            num_windows = tf.shape(actual_mask)[0]
            question_tokens = self.get_question_token_ids(sentence_map, flattened_input_ids, flattened_input_mask,
                                                          top_span_starts[i], top_span_ends[i])  # (num_question_tokens)
            tiled_question = tf.tile(tf.expand_dims(question_tokens, 0),
                                     [num_windows, 1])  # (num_windows, num_ques_tokens)
            question_ones = tf.ones_like(tiled_question, dtype=tf.int32)
            question_zeros = tf.zeros_like(tiled_question, dtype=tf.int32)
            qa_input_ids = tf.concat([tiled_question, input_ids], 1)  # (num_windows, num_ques_tokens + window_size)
            qa_input_mask = tf.concat([question_ones, actual_mask], 1)  # (num_windows, num_ques_tokens + window_size)
            token_type_ids = tf.concat([question_zeros, actual_mask], 1)

            bert_model = BertModel(self.bert_config, is_training, qa_input_ids, qa_input_mask, token_type_ids,
                                   scope='qa')
            bert_embeddings = bert_model.get_sequence_output()  # num_windows, num_ques_tokens + window_size, embed_size
            flattened_embeddings = tf.reshape(bert_embeddings, [-1, self.bert_config.hidden_size])
            output_mask = tf.concat([-1 * question_ones, input_mask], 1)  # (num_windows, num_ques_tokens + window_size)
            flattened_mask = tf.reshape(tf.greater_equal(output_mask, 0), [-1])
            qa_embeddings = tf.boolean_mask(flattened_embeddings, flattened_mask)  # (num_tokens, embed_size)
            qa_scores, qa_indices, qa_starts, qa_ends, qa_embs = self.filter_by_mention_scores(qa_embeddings,
                                                                                               candidate_starts,
                                                                                               candidate_ends, dropout,
                                                                                               c)
            qa_cluster_ids = self.get_top_span_cluster_ids(candidate_starts, candidate_ends, span_starts, span_ends,
                                                           cluster_ids, qa_indices)
            return (i + 1,
                    tf.concat([starts, tf.expand_dims(qa_starts, axis=0)], axis=0),
                    tf.concat([ends, tf.expand_dims(qa_ends, axis=0)], axis=0),
                    tf.concat([labels, tf.expand_dims(qa_cluster_ids, axis=0)], axis=0),
                    tf.concat([scores, tf.expand_dims(qa_scores, axis=0)], axis=0))

        _, antecedent_starts, antecedent_ends, antecedent_labels, antecedent_scores = tf.while_loop(
            lambda i, o1, o2, o3, o4: i < k,
            qa_loop_body,
            [i0, top_antecedent_starts, top_antecedent_ends, top_antecedent_labels, top_antecedent_scores],
            [[], [None, c], [None, c], [None, c], [None, c]])
        predictions = [doc_idx, subtoken_map, top_span_starts, top_span_ends,
                       antecedent_starts, antecedent_ends, antecedent_scores]

        same_cluster_indicator = tf.equal(antecedent_labels, tf.expand_dims(top_span_cluster_ids, 1))  # (k, c)
        pairwise_labels = tf.logical_and(same_cluster_indicator, tf.expand_dims(top_span_cluster_ids > 0, 1))  # (k, c)
        dummy_labels = tf.logical_not(tf.reduce_any(pairwise_labels, 1, keepdims=True))  # [k, 1]
        loss_antecedent_labels = tf.concat([dummy_labels, pairwise_labels], 1)  # [k, c + 1]
        dummy_scores = tf.zeros([k, 1])
        loss_antecedent_scores = tf.concat([dummy_scores, antecedent_scores], 1)  # [k, c + 1]
        losses = self.softmax_loss(loss_antecedent_scores, loss_antecedent_labels)
        return predictions, losses

    def get_bert_embeddings(self, flattened_input_ids, flattened_input_mask, is_training: bool):
        """
        applying BERT to each sliding window, and get token embeddings corresponding to the right tokens
        :param flattened_input_ids: [-1]
        :param flattened_input_mask: [-1]
        :param is_training:
        :return: (num_tokens, embed_size)
        """
        input_ids = tf.reshape(flattened_input_ids, [-1, self.config.sliding_window_size])
        input_mask = tf.reshape(flattened_input_mask, [-1, self.config.sliding_window_size])
        actual_mask = tf.cast(tf.not_equal(input_mask, self.config.pad_idx), tf.int32)
        bert_model = BertModel(self.bert_config, is_training, input_ids, actual_mask, scope='bert')
        bert_embeddings = bert_model.get_sequence_output()  # (num_windows, window_size, embed_size)
        flattened_embeddings = tf.reshape(bert_embeddings, [-1, self.bert_config.hidden_size])
        flattened_mask = tf.greater_equal(flattened_input_mask, 0)
        output_embeddings = tf.boolean_mask(flattened_embeddings, flattened_mask)
        return output_embeddings

    def get_candidate_spans(self, sentence_map):
        """
        get candidate spans based on:
        the length of candidate spans <= max_span_width
        each span is located in a single sentence
        :param sentence_map:
        :return: start and end indices w.r.t num_tokens (num_candidates, ), (num_candidates, )
        """
        num_tokens = tf.shape(sentence_map)[0]
        # candidate_span: every position can be span start, there are max_span_width kinds of end for each start
        candidate_starts = tf.tile(tf.expand_dims(tf.range(num_tokens), 1), [1, self.config.max_span_width])
        candidate_ends = candidate_starts + tf.expand_dims(tf.range(self.config.max_span_width), 0)

        # [num_tokens, max_span_width]，get sentence_id for each token indices
        candidate_start_sentence_indices = tf.gather(sentence_map, candidate_starts)
        candidate_end_sentence_indices = tf.gather(sentence_map, tf.minimum(candidate_ends, num_tokens - 1))
        # [num_tokens, max_span_width]，legitimate spans should reside in a single sentence.
        candidate_mask = tf.logical_and(candidate_ends < num_tokens,
                                        tf.equal(candidate_start_sentence_indices, candidate_end_sentence_indices))
        flattened_candidate_mask = tf.reshape(candidate_mask, [-1])  # [num_tokens * max_span_width] -> [num_candidates]
        # get start indices and end indices for each candidate span
        candidate_starts = tf.boolean_mask(tf.reshape(candidate_starts, [-1]), flattened_candidate_mask)
        candidate_ends = tf.boolean_mask(tf.reshape(candidate_ends, [-1]), flattened_candidate_mask)  # [num_candidates]
        return candidate_starts, candidate_ends

    def get_span_embeddings(self, token_embeddings, span_starts, span_ends):
        """
        get span embeddings from span start embedding, span end embedding and optionally span width embedding
        :param token_embeddings: (num_tokens, embed_size)
        :param span_starts: (num_candidates, )
        :param span_ends: (num_candidates, )
        :return: (num_candidates, embed_size)
        """
        span_emb_list = []

        span_start_emb = tf.gather(token_embeddings, span_starts)  # [num_candidates, embed_size]
        span_emb_list.append(span_start_emb)

        span_end_emb = tf.gather(token_embeddings, span_ends)  # [num_candidates, embed_size]
        span_emb_list.append(span_end_emb)

        span_width = span_ends - span_starts  # [num_candidates]

        if self.config.use_span_width_embeddings:
            span_width_embeddings = tf.get_variable("span_width_embeddings", [self.config.max_span_width,
                                                                              self.config.span_width_embed_size],
                                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
            span_width_emb = tf.gather(span_width_embeddings, span_width)  # [num_candidates, embed_size]
            span_emb_list.append(span_width_emb)

        span_emb = tf.concat(span_emb_list, 1)  # [num_candidates, embed_size]
        return span_emb  # [num_candidates, embed_size]

    def filter_by_mention_scores(self, bert_embeddings, candidate_starts, candidate_ends, dropout, k):
        """
        filter candidate mentions based on mention scores
        :param bert_embeddings: (num_tokens, embed_size)
        :param candidate_starts: (num_candidates, )
        :param candidate_ends: (num_candidates, )
        :param dropout: scalar, dropout keep probability
        :param k: scalar, number of selected mentions
        :return: top_span_indices: (k, ) selected span indices w.r.t num_candidates
        """
        candidate_embeddings = self.get_span_embeddings(bert_embeddings, candidate_starts, candidate_ends)  # [num_candidates, embed_size]
        candidate_mention_scores = utils.ffnn(candidate_embeddings, self.config.ffnn_depth, self.config.ffnn_size,
                                              1, dropout)  # [num_candidates]

        top_span_scores, top_span_indices = tf.nn.top_k(candidate_mention_scores, k)
        top_span_starts = tf.gather(candidate_starts, top_span_indices)  # [k]
        top_span_ends = tf.gather(candidate_ends, top_span_indices)  # [k]
        top_span_emb = tf.gather(candidate_embeddings, top_span_indices)  # [k, emb]
        return top_span_scores, top_span_indices, top_span_starts, top_span_ends, top_span_emb

    def get_top_span_cluster_ids(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels, top_span_indices):
        """
        method to get top_span_cluster_ids
        :param candidate_starts: [num_candidates, ]
        :param candidate_ends: [num_candidates, ]
        :param labeled_starts: [num_mentions, ]
        :param labeled_ends: [num_mentions, ]
        :param labels: [num_mentions, ] gold truth cluster ids
        :param top_span_indices: [k, ]
        :return: [k, ] ground truth cluster ids for each proposed candidate span
        """
        same_start = tf.equal(tf.expand_dims(labeled_starts, 1), tf.expand_dims(candidate_starts, 0))
        same_end = tf.equal(tf.expand_dims(labeled_ends, 1), tf.expand_dims(candidate_ends, 0))
        same_span = tf.logical_and(same_start, same_end)  # [num_labeled, num_candidates] predict_i == label_j

        candidate_labels = tf.matmul(tf.expand_dims(labels, 0), tf.to_int32(same_span))  # [1, num_candidates]
        candidate_labels = tf.squeeze(candidate_labels, 0)  # [num_candidates]
        top_span_cluster_ids = tf.gather(candidate_labels, top_span_indices)
        return top_span_cluster_ids

    def get_question_token_ids(self, sentence_map, flattened_input_ids, flattened_input_mask, top_start, top_end):
        """
        construct question based on the selected mention
        :param sentence_map: (num_tokens, ) tokens to sentence id
        :param flattened_input_ids: (num_windows * window_size)
        :param flattened_input_mask: (num_windows * window_size)
        :param top_start: integer, mention start position w.r.t num_tokens
        :param top_end: integer, mention end position w.r.t num_tokens
        :return: vector of integer, question tokens
        """
        sentence_idx = sentence_map[top_start]
        sentence_tokens = tf.where(tf.equal(sentence_map, sentence_idx))
        sentence_start = tf.where(tf.equal(flattened_input_mask, sentence_tokens[0][0]))
        sentence_end = tf.where(tf.equal(flattened_input_mask, sentence_tokens[-1][0]))
        original_tokens = flattened_input_ids[sentence_start[0][0]: sentence_end[0][0] + 1]

        mention_start = tf.where(tf.equal(flattened_input_mask, top_start))
        mention_end = tf.where(tf.equal(flattened_input_mask, top_end))
        mention_start_in_sentence = mention_start[0][0] - sentence_start[0][0]
        mention_end_in_sentence = mention_end[0][0] - sentence_start[0][0]

        question_token_ids = tf.concat([original_tokens[: mention_start_in_sentence],
                                        [self.config.mention_start_idx],
                                        original_tokens[mention_start_in_sentence: mention_end_in_sentence + 1],
                                        [self.config.mention_end_idx],
                                        original_tokens[mention_end_in_sentence + 1:],
                                        ], 0)
        tf.debugging.assert_less_equal(tf.shape(question_token_ids)[0], self.config.max_question_len)
        return question_token_ids

    def softmax_loss(self, antecedent_scores, antecedent_labels):
        """

        :param antecedent_scores: [k, c+1] the predicted scores by the model
        :param antecedent_labels: [k, c+1] the gold-truth cluster labels
        :return:
        """
        gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels))
        marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1])  # [k]
        log_norm = tf.reduce_logsumexp(antecedent_scores, [1])  # [k]
        loss = log_norm - marginalized_gold_scores  # [k]
        return tf.reduce_sum(loss)
