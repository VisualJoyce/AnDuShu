# Copyright (c) VisualJoyce.
# Licensed under the MIT license.
WORK_DIR=$(readlink -f .)
DATA_DIR=${WORK_DIR}/data
CACHE_DIR=${DATA_DIR}/.cache
ALLENNLP_DIR=${DATA_DIR}/.allennlp
NLTK_DATA=${DATA_DIR}/nltk_data

#  --mount src="${DATA_DIR}",dst=/mnt/data,type=bind \

docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm -m4g -it \
  --mount src="${WORK_DIR}",dst=/src,type=bind \
  --mount src="$CACHE_DIR",dst=/root/.cache,type=bind \
  --mount src="$ALLENNLP_DIR",dst=/root/.allennlp,type=bind \
  --mount src="$NLTK_DATA",dst=/root/nltk_data,type=bind \
  -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
  -w /src visualjoyce/andushu:latest

#PYTHONPATH=src ANNOTATION_DIR=$PWD/data/annotations/spider/ allennlp train configs/spider/gnn/defaults.jsonnet -s data/output/spider/global_gcn --include-package andushu
#PYTHONPATH=src ANNOTATION_DIR=$PWD/data/annotations/geoquery/ allennlp train configs/geoquery/seq2tree/defaults.jsonnet -s data/output/geoquery/seq2tree/defaults --include-package andushu
