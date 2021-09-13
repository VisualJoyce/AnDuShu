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

Training Math23K using multiple GPUs.
```shell
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src ANNOTATION_DIR=$PWD/data/annotations/ \
  MODEL_NAME=bert-base-multilingual-cased TOKENIZERS_PARALLELISM=false \
  allennlp train configs/mathxling/seq2seq/copynet_mbert_distributed_no_finetune_vocab.jsonnet \
  -s data/output/mathxling/seq2seq/copynet_mbert_distributed_no_finetune_vocab --include-package andushu
```

```shell
PYTHONPATH=src ANNOTATION_DIR=$PWD/data/annotations/Math23k/ \
  MODEL_NAME=bert-base-multilingual-cased \
  allennlp evaluate data/output/math23k/seq2seq/copynet_mbert/model.tar.gz \
  data/annotations/Math23k/aggregate_test.json \
  --output-file data/annotations/Math23k/aggregate_test_results.json \
  --include-package andushu
```

```shell
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src ANNOTATION_DIR=$PWD/data/annotations/   \
  MODEL_NAME=xlm-roberta-base TOKENIZERS_PARALLELISM=false POS_TAGS=false \
  allennlp train configs/mathxling/seq2seq/copynet_xlm_distributed_vocab.jsonnet \
  -s data/output/mathxling/seq2seq/copynet_xlm_distributed_vocab --include-package andushu 
```

```shell
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src ANNOTATION_DIR=$PWD/data/annotations/Math23k/ MODEL_NAME=bert-base-multilingual-cased  allennlp train  configs/math23k/seq2seq/copynet_mbert_distributed.jsonnet -s data/output/math23k/seq2seq/copynet_mbert_distributed --include-package andushu
```

```shell
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src ANNOTATION_DIR=$PWD/data/annotations/MathXLing/ MODEL_NAME=bert-base-multilingual-cased  allennlp train  configs/math23k/seq2seq/copynet_mbert_distributed.jsonnet -s data/output/math23k/seq2seq/copynet_mbert_distributed --include-package andushu
```

```shell
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src ANNOTATION_DIR=$PWD/data/annotations/MathXLing/ MODEL_NAME=bert-base-multilingual-cased  allennlp train  configs/mathxling/seq2seq/copynet_mbert_distributed.jsonnet -s data/output/mathxling/seq2seq/copynet_mbert_distributed --include-package andushu
```
```shell
PYTHONPATH=src ANNOTATION_DIR=$PWD/data/annotations/Math23K/ MODEL_NAME=bert-base-multilingual-cased  allennlp evaluate data/output/math23k/seq2seq/copynet_mbert_distributed/model.tar.gz data/annotations/Math23K/dev.json --output-file data/output/math23k/seq2seq/copynet_mbert_distributed/dev_results.json --include-package andushu

PYTHONPATH=src ANNOTATION_DIR=$PWD/data/annotations/Math23K/ MODEL_NAME=bert-base-multilingual-cased  allennlp evaluate data/output/math23k/seq2seq/copynet_mbert_distributed/model.tar.gz data/annotations/Math23K/math23k_test.json --output-file data/output/math23k/seq2seq/copynet_mbert_distributed/test_results.json --include-package andushu

PYTHONPATH=src ANNOTATION_DIR=$PWD/data/annotations/Math23K/ MODEL_NAME=bert-base-multilingual-cased  allennlp evaluate data/output/math23k/seq2seq/copynet_mbert_distributed/model.tar.gz data/annotations/MathQA/dev.json --output-file data/output/math23k/seq2seq/copynet_mbert_distributed/mathqa_dev_results.json --include-package andushu

PYTHONPATH=src ANNOTATION_DIR=$PWD/data/annotations/Math23K/ MODEL_NAME=bert-base-multilingual-cased  allennlp evaluate data/output/math23k/seq2seq/copynet_mbert_distributed/model.tar.gz data/annotations/MathQA/test.json --output-file data/output/math23k/seq2seq/copynet_mbert_distributed/mathqa_test_results.json --include-package andushu

```


```shell
PYTHONPATH=src ANNOTATION_DIR=$PWD/data/annotations/Math23K/ MODEL_NAME=bert-base-multilingual-cased  allennlp evaluate data/output/math23k/seq2seq/copynet_mbert_distributed/model.tar.gz data/annotations/MathXLing/multiarith_en.json --output-file data/output/math23k/seq2seq/copynet_mbert_distributed/multiarith_en_results.json --include-package andushu

PYTHONPATH=src ANNOTATION_DIR=$PWD/data/annotations/Math23K/ MODEL_NAME=bert-base-multilingual-cased  allennlp evaluate data/output/math23k/seq2seq/copynet_mbert_distributed/model.tar.gz data/annotations/MathXLing/singleop_zh.json --output-file data/output/math23k/seq2seq/copynet_mbert_distributed/singleop_zh_results.json --include-package andushu

PYTHONPATH=src ANNOTATION_DIR=$PWD/data/annotations/Math23K/ MODEL_NAME=bert-base-multilingual-cased  allennlp evaluate data/output/math23k/seq2seq/copynet_mbert_distributed/model.tar.gz data/annotations/MathXLing/addsub_en.json --output-file data/output/math23k/seq2seq/copynet_mbert_distributed/addsub_en_results.json --include-package andushu
```


```shell
PYTHONPATH=src ANNOTATION_DIR=$PWD/data/annotations/MathXLing/ MODEL_NAME=bert-base-multilingual-cased  allennlp evaluate data/output/mathxling/seq2seq/copynet_mbert_distributed/model.tar.gz data/annotations/Math23K/math23k_test.json --output-file data/output/mathxling/seq2seq/copynet_mbert_distributed/test_results.json --include-package andushu
```
