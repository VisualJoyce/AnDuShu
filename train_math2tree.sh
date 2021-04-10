# Copyright (c) VisualJoyce.
# Licensed under the MIT license.
WORK_DIR=$(readlink -f .)
DATA_DIR=$(readlink -f "${WORK_DIR}"/data)
CONFIG_DIR=$(readlink -f "${WORK_DIR}"/configs)

PROJECT=$1
CONFIG=$2
SPACY_LANGUAGE=$3
MODEL_NAME=$4

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src ANNOTATION_DIR=${DATA_DIR}/annotations/ \
  MODEL_NAME=${MODEL_NAME} TOKENIZERS_PARALLELISM=false POS_TAGS=false LANGUAGE="${SPACY_LANGUAGE}" \
  allennlp train "${CONFIG_DIR}"/"${PROJECT}"/seq2seq/"${CONFIG}".jsonnet \
  -s data/output/"${PROJECT}"/seq2seq/"${CONFIG}_${SPACY_LANGUAGE}" \
  --include-package andushu
