local dataset_path = std.extVar("ANNOTATION_DIR");
local MODEL_NAME = std.extVar("MODEL_NAME");

{
  "dataset_reader": {
    "type": "seq2seq",
    "source_token_indexers": {
      "tokens": {
        "type": "single_id",
      },
      "bert": {
        "type": "pretrained_transformer_mismatched",
        "max_length": 512,
        "model_name": MODEL_NAME
      }
    },
    "target_token_indexers": {
      "tokens": {
        "type": "single_id",
      }
    }
  },
  "train_data_path": dataset_path + "geo.train",
  "validation_data_path": dataset_path + "geo.val",
  "model": {
    "type": "composed_seq2seq",
    "source_text_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 100,
          "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
          "trainable": true
        },
        "bert": {
          "type": "pretrained_transformer_mismatched",
          "model_name": MODEL_NAME,
          "max_length": 512,
          "last_layer_only": true,
          "train_parameters": true
        }
      }
    },
    "encoder": {
      "type": "lstm",
      "input_size": 868,
      "hidden_size": 100,
      "num_layers": 1,
      "bidirectional": true
    },
    "decoder": {
      "type": "auto_regressive_seq_decoder",
      "target_embedder": {
        "embedding_dim": 100,
        "trainable": true
      },
      "decoder_net": {
        "type": "lstm_cell",
        "decoding_dim": 200,
        "target_embedding_dim": 100,
        "attention": {
          "type": "bilinear",
          "vector_dim": 200,
          "matrix_dim": 200
        },
        "bidirectional_input": true
      },
      "beam_size": 5,
      "max_decoding_steps": 100,
      "token_based_metric": "token_sequence_accuracy"
    }
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "padding_noise": 0.0,
      "batch_size": 15
    }
  },
  "trainer": {
    "num_epochs": 50,
    "patience" : 10,
    "optimizer": {
        "type": "huggingface_adamw",
        "lr": 3e-5,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "correct_bias": true
    },
    "learning_rate_scheduler": {
        "type": "polynomial_decay",
    },
    "grad_norm": 1.0,
    "cuda_device": 0,
    "validation_metric": "+seq_acc"
  }
}
