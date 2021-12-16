# Copyright (c) VisualJoyce.
# Licensed under the MIT license.
WORK_DIR=$(readlink -f .)
DATA_DIR=$(readlink -f "${WORK_DIR}"/data)
CONFIG_DIR=$(readlink -f "${WORK_DIR}"/configs)
OUTPUT_DIR=data/output/"${PROJECT}"/"${CONFIG}"_"${SPACY_LANGUAGE}"_"${OP_TYPE}"_${MODEL_NAME}
#PROJECT=$1
#CONFIG=$2
#SPACY_LANGUAGE=$3

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src ANNOTATION_DIR=${DATA_DIR}/annotations/ \
  MODEL_NAME=${MODEL_NAME} TOKENIZERS_PARALLELISM=false POS_TAGS=false LANGUAGE="${SPACY_LANGUAGE}" OP_TYPE=${OP_TYPE} \
  allennlp train "${CONFIG_DIR}"/"${PROJECT}"/seq2seq_with_copy/"${CONFIG}".jsonnet \
  -s "${OUTPUT_DIR}" \
  --include-package andushu
