![AuDuShu](logo.png)

AnDuShu (案牍术)
==============

An inclusive Text2SQL library built over ``allennlp 2``.

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


```shell
root@ce3daedafb62:/src# PYTHONPATH=src ANNOTATION_DIR=$PWD/data/annotations/Math23k/ \
  MODEL_NAME=bert-base-multilingual-cased \
  allennlp evaluate data/output/math23k/seq2seq/copynet_mbert/model.tar.gz \
  data/annotations/Math23k/aggregate_test.json \
  --output-file data/annotations/Math23k/aggregate_test_results.json \
  --include-package andushu
```