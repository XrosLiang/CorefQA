# CorefQA: Coreference Resolution as Query-based Span Prediction
This repo contains the code and data for the paper [CorefQA: Coreference Resolution as Query-based Span Prediction](https://arxiv.org/abs/1911.01746).

## Preparation
* Install python requirements: `pip install -r requirements.txt`
* Prepare training data: `python prepare_training_data.py`
* Fine-tuning the hyper-parameters in `experiments.conf`

## Training
1. Download the [Ontonotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19) dataset.
2. Download the [SpanBERT](https://github.com/facebookresearch/SpanBERT) pre-trained model.
3. Run `./setup_training.sh <ontonotes/path/ontonotes-release-5.0> $data_dir` for data preparation.
4. Training the model. `GPU=0 python train.py <experiment>`，Results are saved in the `log_root` directory. You can see the training details with `TensorBoard`.

## Using Pre-trained Model
Download the pre-trained `CorefQA` model using the following command. If you want to train the CorefQA model, you can skip this step.
`./download_pretrained.sh <model_name>` (e.g,: spanbert_base, spanbert_large)  Will download the fine-tuned version of `CorefQA`. You can use it with `evaluate.py` and `predict.py`。

## Evaluation
Run `GPU=0 python evaluate.py <experiment>` to evaluate the model. You can set `eval_path` and `conll_eval_path` in `experiments.conf` to choose the evaluation files：

| Model          | F1 (%) |
| -------------- |:------:|
| CorefQA + SpanBERT-base  | 79.9  |
| CorefQA + SpanBERT-large | 83.1   |

## Prediction

* Save the text for prediction in a txt file. If the text contains speaker name information, wrap the speaker with `<speaker></speaker>` and put it in front of its utterence. For example:
```text
<speaker> Host </speaker> A traveling reporter now on leave and joins us to tell her story. Thank you for coming in to share this with us.
```
* run `GPU=0 python predict.py <experiment> <input_file> <output_file>` will save the prediction results in `<output_file>`, The prediction for each instance is a list of clusters，each cluster is a list of mentions. Each mention is (text, (span_start, span_end)). For example:
```python
[[('A traveling reporter', (26, 46)), ('her', (81, 84)), ('you', (98, 101))]]
```

## Citing
If you think our paper is interesting, please cite [Coreference Resolution as Query-based Span Prediction](https://arxiv.org/abs/1911.01746).
```
@article{wu2019coreference,
  title={Coreference Resolution as Query-based Span Prediction},
  author={Wu, Wei and Wang, Fei and Yuan, Arianna and Wu, Fei and Li, Jiwei},
  journal={arXiv preprint arXiv:1911.01746},
  year={2019}
}
```

## Acknowledgement
We borrow some code from `https://github.com/mandarjoshi90/coref`，Thanks to them!
