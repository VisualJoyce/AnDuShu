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
    echo "*************************************************************************************************************"
    echo "                                               Evaluating $d $s                                              "
    echo "*************************************************************************************************************"
    PYTHONPATH=src \
    allennlp evaluate "${MODEL_DIR}"/model.tar.gz \
    data/annotations/"$d"/"$s".json \
    --output-file "${MODEL_DIR}"/"$d"_"$s"_results.json \
    --cuda-device 0 \
    --include-package andushu
  done
done

# Evaluation over operation-based splits
for e in "${evals[@]}"
do
  for l in "${langs[@]}"
  do
    echo "*************************************************************************************************************"
    echo "                                               Evaluating $e $l                                              "
    echo "*************************************************************************************************************"
    PYTHONPATH=src \
    allennlp evaluate "${MODEL_DIR}"/model.tar.gz \
    data/annotations/MathXLing/"$e"_"$l".json \
    --output-file "${MODEL_DIR}"/"$e"_"$l"_results.json \
    --cuda-device 0 \
    --include-package andushu
  done
done

grep -Hn "answer_acc" "${MODEL_DIR}"/*_results.json
