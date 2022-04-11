import json

if __name__ == '__main__':
    # reader = IIRCContextRetrievalReader('../../data/iirc/context_articles.json')
    # reader.read('../../data/iirc/iirc_tiny.json')


    with open('../data/iirc/preprocessed_iirc_train.json') as f:
        dataset = json.load(f)

        context_n = 0
        context_first_gold_n = 0
        context_top_10_n = 0
        context_top_100_n = 0

        init_title_set = set()
        context_title_set = set()
        for context_questions in dataset:
            init_context = context_questions['text']
            all_links = list(map(lambda x: (x['indices'], x['target']), context_questions['links']))
            title = context_questions['title']

            init_title_set.add(title.lower())
            context_title_set.add(title.lower())

            for question_json in context_questions['questions']:
                for context_dict in question_json['context']:
                    context_title_set.add(context_dict['passage'].lower())
                    if context_dict['passage'] == 'main':
                        continue
                    context_n += 1
                    if context_dict['sent_indices'][0] == 0:
                        context_first_gold_n += 1
                    if context_dict['sent_indices'][0] < 10:
                        context_top_10_n += 1
                    if context_dict['sent_indices'][0] < 100:
                        context_top_100_n += 1

        context_first_gold_n /= context_n
        context_top_10_n /= context_n
        context_top_100_n /= context_n

    with open('../data/iirc/preprocessed_iirc_test.json') as f:
        dataset = json.load(f)

        context_questions_n = 0
        cq_in_set_n = 0
        dev_init_context_title_set = set()
        dev_context_title_set = set()
        for context_questions in dataset:
            context_questions_n += 1
            init_context = context_questions['text']
            all_links = list(map(lambda x: (x['indices'], x['target']), context_questions['links']))
            title = context_questions['title']
            dev_init_context_title_set.add(title)

            if title.lower() in context_title_set:
                cq_in_set_n += 1

            for question_json in context_questions['questions']:
                for context_dict in question_json['context']:
                    dev_context_title_set.add(context_dict['passage'].lower())

    init_intersection_set = init_title_set.intersection(dev_init_context_title_set)
    intersection_set = context_title_set.intersection(dev_context_title_set)
    cq_in_set_n /= context_questions_n

    with open('../data/iirc/context_articles.json', 'r') as f:
        wiki_dict = json.load(f)

    print("")
