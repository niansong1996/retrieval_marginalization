# Mitigating False-Negative Contexts in Multi-Document Question Answering with Retrieval Marginalization
Authors: [Ansong Ni](https://niansong1996.github.io/), [Matt Gardner](https://matt-gardner.github.io/) and [Pradeep Dasigi](https://pdasigi.github.io/)

## Full Model Outputs for IIRC
We include the full model outputs on the validation and test set for IIRC in `joint_retrieval_results`. If you wish to use another QA model to improve over our performance, feel free to use the predicted links or retrieved contexts by our model. The `jsonl` file is organized as follows:
```
{
question: # orginal question in IIRC, 
original_paragraph: # introductory paragraph in IIRC, 
link_prediction: {
        predicted_links: # a list of the predicted links to other articles,
        gold_question_links: # a list of the gold links to other articles,
},
context_retrieval: {
        gold_link_name_sent_list: # gold text snippets from different documents, 
        predicted_link_name_set_list: # predicted text snippets from different documents,
},
qa: {
        predicted_answer_ability: # we use NumNet+, which predicts the question type first,
        predicted_answer: # the final predicted answer,
        gold_answer_type: # the ground truth answer type
        gold_answer: # a dictionary of the gold answer that NumNet+ uses
        em: # the exact match score, either 0 or 1
        f1: # the f1 score, between 0 and 1
}
}
```

## Citation

If you use our code or model outputs, please cite:
```
@inproceedings{ni-etal-2021-mitigating,
    title = "Mitigating False-Negative Contexts in Multi-document Question Answering with Retrieval Marginalization",
    author = "Ni, Ansong  and
      Gardner, Matt  and
      Dasigi, Pradeep",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.497",
    doi = "10.18653/v1/2021.emnlp-main.497",
    pages = "6149--6161",
}
```
