local bert_model = "roberta-base";
local bert_hidden_size = 768;
local preprocessed_wiki_file_path = "data/iirc/preprocessed_context_articles.json";
local top_k_link_per_question = 3;

{
    "dataset_reader" : {
        "type": "iirc-joint-qa-reader",
        "wiki_file_path": preprocessed_wiki_file_path,
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
        "type": "iirc-joint-qa-reader",
        "wiki_file_path": preprocessed_wiki_file_path,
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
        "type": "joint-qa",
 #        below are for retrieval
        "transformer_model_name": bert_model,
        "beam_size_link": top_k_link_per_question,
        "print_trajectory": false,
        "use_joint_prob": false,
 #        below are for qa
        "skip_when_all_empty": ["passage_span", "question_span", "addition_subtraction", "counting", "none", "binary"],
        "relaxed_span_match_for_finding_labels": true,
        "q_max_tokens": 64,
        "c_max_tokens": 463,
        "hidden_size": bert_hidden_size,
        "answering_abilities": ["passage_span_extraction", "question_span_extraction",
                                "addition_subtraction", "counting", "none", "binary"],
        "use_gcn": true,
        "gcn_steps": 3,
        "dropout_prob": 0.1,
        "top_m_context": 0,
        "gold_link_for_retrieval_training": true,
        "marginalization_loss_weight": 0.0,
        "gold_context_loss_weight": 0.0,
        "invalid_context_loss_weight": 0.0,
        "use_link_prediction_model": false,
        "use_context_retrieval_model": true,
        "use_qa_model": false,
    },
    "data_loader": {
        "batch_size": 2,
        "shuffle": true,
    },
    "validation_data_loader": {
        "batch_size": 2,
        "shuffle": false
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1.0e-5,
            "eps": 1e-6,
        },
        "num_epochs": 30,
        "cuda_device": 0,
        "validation_metric": "+_cf_f",
        "grad_clipping": 1.0,
    },
}
