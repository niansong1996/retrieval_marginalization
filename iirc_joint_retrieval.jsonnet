local bert_model = "bert-base-uncased";
local preprocessed_wiki_file_path = "data/iirc/preprocessed_context_articles.json";
local top_k_link_per_question = 3;

{
    "dataset_reader" : {
        "type": "iirc-joint-retrieval-reader",
        "wiki_file_path": preprocessed_wiki_file_path,
#        "cache_directory": "data/iirc/__cache__",
        "transformer_model_name": bert_model,
        "q_max_tokens": 64,
        "c_max_tokens": 384,
        "skip_invalid_examples": false,
        "sent_n": 1,
        "padding_sent_n": 1,
        "stride": 1,
        "neg_n": 7,
        "include_main": false,
        "add_ctx_sep": false,
        "add_init_context": false,
        "link_per_question": top_k_link_per_question,
    },
    "validation_dataset_reader" : {
        "type": "iirc-joint-retrieval-reader",
        "wiki_file_path": preprocessed_wiki_file_path,
#        "cache_directory": "data/iirc/__cache__",
        "transformer_model_name": bert_model,
        "q_max_tokens": 64,
        "c_max_tokens": 384,
        "skip_invalid_examples": false,
        "sent_n": 1,
        "padding_sent_n": 1,
        "stride": 1,
        "neg_n": 7,
        "max_neg_n": 500,
        "include_main": false,
        "add_ctx_sep": false,
        "add_init_context": false,
        "link_per_question": top_k_link_per_question,
    },
    "train_data_path": "data/iirc/preprocessed_iirc_tiny.json",
    "validation_data_path": "data/iirc/preprocessed_iirc_tiny.json",
    "vocabulary": {
        "type": "empty",
    },
    "model": {
        "type": "joint-retriever",
        "transformer_model_name": bert_model,
        "beam_size_link": top_k_link_per_question,
        "beam_size_context": 5,
        "print_trajectory": false,
        "load_model_weights": false,
        "link_predictor_weights_file": "",
        "context_retriever_weights_file": "",
        "use_joint_prob": false,
    },
    "data_loader": {
        "batch_size": 2,
        "shuffle": true
    },
    "validation_data_loader": {
        "batch_size": 1,
        "shuffle": false
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1.0e-5
        },
        "num_epochs": 10,
        "cuda_device": 0,
        "validation_metric": "+jr_recall",
    }
}
