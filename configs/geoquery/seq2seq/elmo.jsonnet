local dataset_path = std.extVar("ANNOTATION_DIR");

{
  "dataset_reader": {
    "type": "seq2seq",
    "source_token_indexers": {
      "elmo": {
        "type": "elmo_characters"
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
        "elmo": {
          "type": "elmo_token_embedder",
          "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
          "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
          "do_layer_norm": false,
          "dropout": 0.5,
          "requires_grad": false,
          "scalar_mix_parameters": [
            0,
            0,
            1
          ]
        }
      }
    },
    "encoder": {
      "type": "lstm",
      "input_size": 1024,
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
