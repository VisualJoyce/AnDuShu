#!/bin/bash
# Copyright (c) VisualJoyce.
# Licensed under the MIT license.
WORK_DIR=$(readlink -f .)
DATA_DIR=$(readlink -f "${WORK_DIR}"/data)
CACHE_DIR=${DATA_DIR}/.cache
ALLENNLP_DIR=${DATA_DIR}/.allennlp
NLTK_DATA=${DATA_DIR}/nltk_data

PROJECT=$1
MODEL_DIR=$2
RUN_TYPE=$3

docker run --gpus '"'device="$CUDA_VISIBLE_DEVICES"'"' --ipc=host --rm -it \
  --mount src="${WORK_DIR}",dst=/src,type=bind \
  --mount src="${DATA_DIR}",dst=/src/data,type=bind \
  --mount src="$CACHE_DIR",dst=/root/.cache,type=bind \
  --mount src="$ALLENNLP_DIR",dst=/root/.allennlp,type=bind \
  --mount src="$NLTK_DATA",dst=/root/nltk_data,type=bind \
  -e NVIDIA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  -w /src visualjoyce/andushu:latest \
  bash -c "bash ${PROJECT}_${RUN_TYPE}.sh ${MODEL_DIR}"
