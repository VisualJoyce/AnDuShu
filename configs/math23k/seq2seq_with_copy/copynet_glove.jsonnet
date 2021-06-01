local dataset_path = std.extVar("ANNOTATION_DIR");
local MODEL_NAME = std.extVar("MODEL_NAME");

{
  "dataset_reader": {
    "type": "copynet_math23k",
    'source_tokenizer': {
      "type": "spacy",
      "language": "zh_core_web_sm"
    },
    'target_tokenizer': {
      "type": "spacy"
    },
    "source_token_indexers": {
      "tokens": {
        "type": "single_id",
      }
    },
    "target_token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": "target_tokens",
      }
    },
  },
  "train_data_path": dataset_path + "math23k_train.json",
  "validation_data_path": dataset_path + "math23k_test.json",
  "model": {
    "type": "copynet_seq2seq",
    "source_text_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 100,
          "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
          "trainable": true
        }
      }
    },
    "encoder": {
      "type": "lstm",
      "input_size": 100,
      "hidden_size": 100,
      "num_layers": 1,
      "bidirectional": true
    },
    "attention": {
      "type": "bilinear",
      "vector_dim": 200,
      "matrix_dim": 200
    },
    "beam_size": 5,
    "max_decoding_steps": 100,
    "target_embedding_dim": 100,
    "target_namespace": "target_tokens",
    "token_based_metric": "equation_answer_accuracy"
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "padding_noise": 0.0,
      "batch_size": 200
    }
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": 0.01
    },
    "learning_rate_scheduler": {
      "type": "noam",
      "warmup_steps": 1000,
      "model_size": 200
    },
    "num_epochs": 150,
    "patience" : 30,
    "cuda_device": 0,
    "validation_metric": "+answer_acc"
  }
}
