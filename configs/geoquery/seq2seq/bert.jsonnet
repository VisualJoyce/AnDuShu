local dataset_path = std.extVar("ANNOTATION_DIR");
local MODEL_NAME = std.extVar("MODEL_NAME");

{
  "dataset_reader": {
    "type": "seq2seq",
    "source_token_indexers": {
      "bert": {
        "type": "pretrained_transformer_mismatched",
        "max_length": 512,
        "model_name": MODEL_NAME
      }
    },
    "target_token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": "target_tokens",
      }
    }
  },
  "train_data_path": dataset_path + "geo.train",
  "validation_data_path": dataset_path + "geo.val",
  "model": {
    "type": "seq2seq",
    "source_text_embedder": {
      "token_embedders": {
        "bert": {
          "type": "pretrained_transformer_mismatched",
          "model_name": MODEL_NAME,
          "max_length": 512,
          "last_layer_only": true,
          "train_parameters": false
        }
      }
    },
    "encoder": {
      "type": "lstm",
      "input_size": 768,
      "hidden_size": 100,
      "num_layers": 1,
      "bidirectional": true
    },
    "decoder": {
      "type": "auto_regressive_seq_decoder",
      "target_namespace": "target_tokens",
      "target_embedder": {
        "embedding_dim": 100,
        "vocab_namespace": "target_tokens",
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
    "validation_metric": "+seq_acc"
  }
}
