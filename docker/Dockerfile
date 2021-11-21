FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
ENV OPENMPI_VERSION=4.0.5
ENV TORCH_VERSION=1.8.0
ENV CUDA_VERSION=11.1
ENV CU_VERSION=111
ENV CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5;8.0"

RUN apt-get update
RUN apt-get install -y wget libibverbs-dev libsysfs-dev cmake libboost-all-dev libeigen3-dev libbz2-dev liblzma-dev ssh git

# basic python packages
RUN conda install cython
RUN pip install msgpack msgpack_numpy cytoolz more_itertools matplotlib scipy nltk transformers tensorboard \
    ipdb lz4 lmdb jsonlines gputil opencv-python pandas \
    xlrd scikit-learn git+https://github.com/PetrochukM/PyTorch-NLP.git \
    pkuseg ftfy asdl astor attrs babel bpemb jsonnet networkx pyrsistent \
    pytest records tabulate tqdm entmax sortedcollections

RUN pip install allennlp unidecode parsimonious editdistance sqlparse
RUN pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+cu${CU_VERSION}.html
RUN pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+cu${CU_VERSION}.html
RUN pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+cu${CU_VERSION}.html
RUN pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+cu${CU_VERSION}.html
RUN pip install torch-geometric

RUN pip install spacy
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download zh_core_web_sm
RUN pip install jieba word2number

WORKDIR /src