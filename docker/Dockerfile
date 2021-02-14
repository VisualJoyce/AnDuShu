FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

ENV OPENMPI_VERSION=4.0.5
ENV CUDA_HOME=/usr/local/cuda-11.0
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
RUN pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
RUN pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
RUN pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
RUN pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
RUN pip install torch-geometric

RUN pip install spacy
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download zh_core_web_sm

WORKDIR /src