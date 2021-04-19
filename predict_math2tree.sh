#!/bin/bash
#ANNOTATION_DIR=$PWD/data/annotations/Math23K/
#MODEL_NAME=bert-base-multilingual-cased
MODEL_DIR=$1

declare -a datasets=(Math23K MathQA)
declare -a splits=(dev test)
declare -a evals=(addsub singleop multiarith)
declare -a langs=(zh en)

# Evaluation over Math23K and MathQA
for d in "${datasets[@]}"
do
  for s in "${splits[@]}"
  do
    if [ -f "${MODEL_DIR}"/"$d"_"$s".jsonl ];
    then
      continue
    fi

    PYTHONPATH=src \
    allennlp predict \
    --output-file="${MODEL_DIR}"/"$d"_"$s".jsonl \
    --predictor math2tree --use-dataset-reader \
    --cuda-device 0 \
    --include-package andushu \
    "${MODEL_DIR}"/model.tar.gz \
    data/annotations/"$d"/"$s".json
  done
done

# Evaluation over operation-based splits
for e in "${evals[@]}"
do
  for l in "${langs[@]}"
  do
    if [ -f "${MODEL_DIR}"/"$e"_"$l".jsonl ];
    then
      continue
    fi

    PYTHONPATH=src \
    allennlp predict \
    --output-file="${MODEL_DIR}"/"$e"_"$l".jsonl \
    --predictor seq2seq --use-dataset-reader \
    --cuda-device 0 \
    --include-package andushu \
    "${MODEL_DIR}"/model.tar.gz \
    data/annotations/MathXLing/"$e"_"$l".json
  done
done
