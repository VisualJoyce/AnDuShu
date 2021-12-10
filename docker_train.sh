#!/bin/bash
# Copyright (c) VisualJoyce.
# Licensed under the MIT license.
WORK_DIR=$(readlink -f .)
DATA_DIR=$(readlink -f "${WORK_DIR}"/data)
CACHE_DIR=${DATA_DIR}/.cache
ALLENNLP_DIR=${DATA_DIR}/.allennlp
NLTK_DATA=${DATA_DIR}/nltk_data

#PROJECT=$1
#SUB_PROJECT=$2
#CONFIG=$3
#SPACY_LANGUAGE=$4
#MODEL_NAME=$5

docker run --gpus '"'device="$CUDA_VISIBLE_DEVICES"'"' --ipc=host --rm -it \
  --mount src="${WORK_DIR}",dst=/src,type=bind \
  --mount src="${DATA_DIR}",dst=/src/data,type=bind \
  --mount src="$CACHE_DIR",dst=/root/.cache,type=bind \
  --mount src="$ALLENNLP_DIR",dst=/root/.allennlp,type=bind \
  --mount src="$NLTK_DATA",dst=/root/nltk_data,type=bind \
  -e NVIDIA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  -w /src visualjoyce/andushu:latest \
  bash -c "PROJECT=${SUB_PROJECT} CONFIG=${CONFIG} SPACY_LANGUAGE=${SPACY_LANGUAGE} MODEL_NAME=${MODEL_NAME} bash train_${PROJECT}.sh "
