local stringToBool(s) =
  if s == "true" then true
  else if s == "false" then false
  else error "invalid boolean: " + std.manifestJson(s);

local dataset_path = std.extVar("ANNOTATION_DIR");
local MODEL_NAME = std.extVar("MODEL_NAME");
local CUDA_DEVICES = std.map(std.parseInt, std.split(std.extVar("CUDA_VISIBLE_DEVICES"), ","));
local POS_TAGS = stringToBool(std.extVar("POS_TAGS"));
local LANGUAGE = std.extVar("LANGUAGE");
local TRIPLES_PATH = dataset_path + 'CCKS2021/triples.txt';

{
  "dataset_reader": {
    "type": "copynet_ccks2021",
    "triples_data_path": TRIPLES_PATH,
    'source_tokenizer': {
      "type": "spacy",
      "pos_tags": POS_TAGS,
      "language": LANGUAGE + "_core_web_sm"
    },
//    'target_tokenizer': {
//      "pos_tags": POS_TAGS,
//      "type": "spacy"
//    },
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
    },
  },
  "train_data_path": dataset_path + "CCKS2021/train.xlsx",
  "validation_data_path": dataset_path + "CCKS2021/train.xlsx",
  "test_data_path": dataset_path + "CCKS2021/test1.xlsx",
  "model": {
    "type": "seq2seq",
    "source_text_embedder": {
      "token_embedders": {
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
      "input_size": 768,
      "hidden_size": 256,
      "num_layers": 2,
      "bidirectional": true
    },
    "decoder": {
      "type": "auto_regressive_seq_decoder_with_copy",
      "target_namespace": "target_tokens",
      "target_embedder": {
        "embedding_dim": 100,
        "vocab_namespace": "target_tokens",
        "trainable": true
      },
      "decoder_net": {
        "type": "lstm_cell",
        "decoding_dim": 512,
        "target_embedding_dim": 100,
        "attention": {
          "type": "bilinear",
          "vector_dim": 512,
          "matrix_dim": 512
        },
        "bidirectional_input": true
      },
      "beam_size": 5,
      "max_decoding_steps": 100,
      "token_based_metric": {
        "type": "sparql_answer_accuracy",
        "triples_data_path": TRIPLES_PATH,
      }
    }
  },
  "data_loader": {
//    "num_workers": 4,
    "batch_sampler": {
      "type": "bucket",
      "padding_noise": 0.0,
      "batch_size": 30
    }
  },
  "validation_data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "padding_noise": 0.0,
      "batch_size": 100
    }
  },
  "trainer": {
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
    "num_epochs": 150,
    "patience" : 150,
    "num_gradient_accumulation_steps": std.ceil(5 / std.length(CUDA_DEVICES)),
    "cuda_device": 0,
    "validation_metric": "+answer_acc"
  }
}
