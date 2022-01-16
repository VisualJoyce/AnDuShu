![AuDuShu](logo.png)

AnDuShu
==============

An inclusive natural language to code library built over ``allennlp 2``.

---------------------------------

This repository is using code from the following resources:
* [allennlp-models](https://github.com/allenai/allennlp-models)
* [allennlp_sempar](https://github.com/jbkjr/allennlp_sempar)
* [allennlp-semparse](https://github.com/allenai/allennlp-semparse)

In this repo, we adapt the code to the latest allennlp version.

----------------------------------

ATIS
----

```shell
PYTHONPATH=src ANNOTATION_DIR=$PWD/data/annotations/atis/ \
  allennlp train configs/atis/seq2seq/defaults.jsonnet \
  -s data/output/atis/seq2seq --include-package andushu
```

```shell
PYTHONPATH=src ANNOTATION_DIR=$PWD/data/annotations/atis/ \
  allennlp predict \
  --output-file=data/output/atis/seq2seq/seq2seq.jsonl \
  --predictor seq2seq \
  --include-package andushu \
  data/output/atis/seq2seq/model.tar.gz \
  data/annotations/atis/atis_test.jsonl 
```

GEOQUERY
--------

```shell
PYTHONPATH=src ANNOTATION_DIR=$PWD/data/annotations/geoquery/ \
  allennlp train configs/geoquery/seq2seq/defaults.jsonnet \
  -s data/output/geoquery/seq2seq --include-package andushu
```

```shell
PYTHONPATH=src ANNOTATION_DIR=$PWD/data/annotations/geoquery/ \
  allennlp predict \
  --output-file=data/output/geoquery/seq2seq/seq2seq.jsonl \
  --predictor seq2seq \
  --include-package andushu data/output/geoquery/seq2seq/model.tar.gz \
  data/annotations/geoquery/geo_test.jsonl 
```

Math Word Problems
------------------

This repo includes the code for 
```bibtex
@article{Tan2021InvestigatingMW,
  title={Investigating Math Word Problems using Pretrained Multilingual Language Models},
  author={Minghuan Tan and Lei Wang and Lingxiao Jiang and Jing Jiang},
  journal={ArXiv},
  year={2021},
  volume={abs/2105.08928}
}
```

### Annotations

Annotations used for this paper can be found at 
* [百度网盘](https://pan.baidu.com/s/1bjqJqxsAPM1cmBwrq9gePg?pwd=o734)

![9B9A5987B7C1CF1482CAAF28B32AC0DE](https://user-images.githubusercontent.com/2136700/149643314-89aba90a-186b-4e1a-9882-e2923cadd00a.png)
* [GoogleDrive](https://drive.google.com/drive/folders/1l6o1nE4qNS8gfjKK6Q8edQq4w4I53uIR?usp=sharing).

### Training

Training Math23K using `bert-base-multilingual-cased`.
```shell
CUDA_VISIBLE_DEVICES=2 PROJECT=math2tree SUB_PROJECT=math23k CONFIG=transformer_vocab SPACY_LANGUAGE=zh MODEL_NAME=bert-base-multilingual-cased OP_TYPE=disallow_pow bash docker_train.sh
```

Training over MathQA-Adapted without `Pow`.
```shell
CUDA_VISIBLE_DEVICES=2 PROJECT=math2tree SUB_PROJECT=mathqa CONFIG=transformer_vocab SPACY_LANGUAGE=zh MODEL_NAME=bert-base-multilingual-cased OP_TYPE=disallow_pow bash docker_train.sh
```

Training over MathQA-Adapted with `Pow`.
```shell
CUDA_VISIBLE_DEVICES=2 PROJECT=math2tree SUB_PROJECT=mathqa CONFIG=transformer_vocab SPACY_LANGUAGE=zh MODEL_NAME=bert-base-multilingual-cased OP_TYPE=allow_pow bash docker_train.sh
```

Training over MathXLing without `Pow`.
```shell
CUDA_VISIBLE_DEVICES=2 PROJECT=math2tree SUB_PROJECT=mathxling CONFIG=transformer_vocab SPACY_LANGUAGE=zh MODEL_NAME=bert-base-multilingual-cased OP_TYPE=disallow_pow bash docker_train.sh
```

Training over MathXLing with `Pow`.
```shell
CUDA_VISIBLE_DEVICES=2 PROJECT=math2tree SUB_PROJECT=mathxling CONFIG=transformer_vocab SPACY_LANGUAGE=zh MODEL_NAME=bert-base-multilingual-cased OP_TYPE=allow_pow bash docker_train.sh
```

### Evaluation

```
CUDA_VISIBLE_DEVICES=7 bash docker_eval.sh math2tree data/output/mathqa/transformer_vocab_en_disallow_pow_bert-base-multilingual-cased evaluate
CUDA_VISIBLE_DEVICES=7 bash docker_eval.sh math2tree data/output/mathqa/transformer_vocab_en_allow_pow_bert-base-multilingual-cased evaluate

CUDA_VISIBLE_DEVICES=7 bash docker_eval.sh math2tree data/output/mathqa/transformer_vocab_en_disallow_pow_bert-base-multilingual-cased evaluate 
CUDA_VISIBLE_DEVICES=7 bash docker_eval.sh math2tree data/output/math23k/transformer_vocab_zh_disallow_pow_bert-base-multilingual-cased evaluate
 
CUDA_VISIBLE_DEVICES=7 bash docker_eval.sh math2tree data/output/math23k/transformer_vocab_zh_allow_pow_bert-base-multilingual-cased evaluate
CUDA_VISIBLE_DEVICES=7 bash docker_eval.sh math2tree data/output/math23k/transformer_vocab_en_allow_pow_bert-base-multilingual-cased evaluate

CUDA_VISIBLE_DEVICES=7 bash docker_eval.sh math2tree data/output/math23k/transformer_vocab_en_disallow_pow_bert-base-multilingual-cased evaluate

CUDA_VISIBLE_DEVICES=7 bash docker_eval.sh math2tree data/output/mathxling/transformer_vocab_zh_sallow_pow_bert-base-multilingual-cased evaluate
CUDA_VISIBLE_DEVICES=7 bash docker_eval.sh math2tree data/output/mathxling/transformer_vocab_zh_allow_pow_bert-base-multilingual-cased evaluate

CUDA_VISIBLE_DEVICES=7 bash docker_eval.sh math2tree data/output/mathxling/transformer_vocab_zh_allow_pow_xlm-roberta-base evaluate
```
