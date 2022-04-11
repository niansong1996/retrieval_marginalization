import json
import spacy
import re
from tqdm import tqdm
from pathlib import Path

from typing import List, Dict


def preprocess_wiki(data_prefix: Path, wiki_file_name: str, redirect_file_name: str):
    # load spacy to sentencize the documents
    spacy_pipeline = spacy.load("en", disable=["tagger", "parser", "ner"])
    spacy_pipeline.add_pipe(spacy_pipeline.create_pipe("sentencizer"))

    # load the redirect dictionary
    with open(str(data_prefix.joinpath(redirect_file_name)), 'r') as f:
        redirect_dict = json.load(f)

    new_wiki_dict = dict()
    print(type(redirect_dict))
    with open(str(data_prefix.joinpath(wiki_file_name)), 'r') as f:
        wiki_dict: dict = json.load(f)

        for title, content in tqdm(wiki_dict.items()):
            content = re.sub(re.compile('<.*?>'), '', content)
            sent_dict_list = []

            spacy_doc = spacy_pipeline(content)
            for sent in spacy_doc.sents:
                s, e = sent.start_char, sent.end_char
                sent_dict = {'text': sent.string, 'start_idx': s, 'end_idx': e}
                sent_dict_list.append(sent_dict)

            new_wiki_dict[title] = sent_dict_list

        # process the redirected articles
        for k, v in redirect_dict.items():
            if k not in new_wiki_dict and v in new_wiki_dict:
                new_wiki_dict[k] = new_wiki_dict[v]
                print('{0} got redirected to {1}.'.format(k, v))

    preprocess_wiki_file_name = 'preprocessed_'+wiki_file_name
    with open(str(data_prefix.joinpath(preprocess_wiki_file_name)), 'w+') as f:
        json.dump(new_wiki_dict, f)

    return preprocess_wiki_file_name


def preprocess_iirc(data_prefix: Path, preprocessed_wiki_file_name: str, iirc_dataset_names: List[str]):

    # load the wiki file that preprocessed with spacy
    with open(str(data_prefix.joinpath(preprocessed_wiki_file_name)), 'r') as f:
        wiki_dict: Dict[List[Dict]] = json.load(f)

    for iirc_dataset_name in iirc_dataset_names:
        with open(str(data_prefix.joinpath(iirc_dataset_name))) as f:
            print('Start preprocessing {} ...'.format(iirc_dataset_name))

            dataset = json.load(f)
            for context_questions in tqdm(dataset):
                for question_json in context_questions['questions']:
                    for context_dict in question_json['context']:
                        doc_title = context_dict['passage'].lower()

                        if doc_title == 'main' or doc_title not in wiki_dict:
                            context_dict['sent_indices'] = [-1, -1]
                            if doc_title != 'main' and doc_title not in wiki_dict:
                                print('document titled {} is not found!'.format(doc_title))
                        else:
                            # try to match up the char indices and sent indices
                            doc_sents = wiki_dict[doc_title]
                            s, e = -1, -1
                            for i, sent_dict in enumerate(doc_sents):
                                cs, ce = context_dict['indices']
                                ss, se = sent_dict['start_idx'], sent_dict['end_idx']
                                if ss <= cs < se:
                                    s = i
                                if ss <= ce <= se:
                                    e = i
                                if s != -1 and e != -1:
                                    break
                            context_dict['sent_indices'] = [s, e]

            with open(str(data_prefix.joinpath('preprocessed_'+iirc_dataset_name)), 'w+') as f1:
                json.dump(dataset, f1)



if __name__ == '__main__':
    data_prefix = Path(__file__).parent.joinpath('data/iirc/')

    wiki_file_name = 'context_articles.json'
    redirect_file_name = 'redirect_index.json'
    iirc_file_list = ['iirc_train.json', 'iirc_test.json', 'iirc_dev.json', 'iirc_tiny.json']

    preprocessed_wiki_file_name = preprocess_wiki(data_prefix, wiki_file_name, redirect_file_name)

    preprocess_iirc(data_prefix, preprocessed_wiki_file_name, iirc_file_list)

