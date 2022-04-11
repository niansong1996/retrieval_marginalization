import unittest
import torch

from numpy.testing import assert_allclose
from rranm_modules.utils.iirc_metric import SetFMeasure
from rranm_modules.readers.drop_reader import DropReader
from rranm_modules.readers.iirc_qa_reader import IIRCQAReader

drop_reader = DropReader('../data/iirc/preprocessed_context_articles.json', 384, 64)
iirc_qa_reader = IIRCQAReader('../data/iirc/preprocessed_context_articles.json')

drop_reads = list(drop_reader._read('../data/iirc/drop_dataset_dev.json'))
iirc_qa_reads = list(iirc_qa_reader._read('../data/iirc/preprocessed_iirc_dev.json'))


class CrossReaderTest(unittest.TestCase):
    def test_length_consistency(self):
        assert len(drop_reads) == len(iirc_qa_reads)

    def test_question_consistency(self):
        pass

class SelfTest(unittest.TestCase):
    def test_number_indices(self):
        for instance in iirc_qa_reads:
            question_tokens = instance.fields['question'].tokens
            passage_tokens = instance.fields['context'].tokens

            all_numbers = instance.fields['all_numbers'].metadata

            for i in instance.fields['question_number_indices']:
                if i in [0, -1]:
                    continue
                assert question_tokens[i].text in all_numbers, \
                    'question {} select {} num length {} \n'.format(question_tokens, question_tokens[i].text)

            for i in instance.fields['passage_number_indices']:
                if i in [0, -1]:
                    continue
                assert passage_tokens[i-1].text in all_numbers, \
                    'question {} select {} \n'.format(passage_tokens, passage_tokens[i-1].text)


if __name__ == '__main__':
    unittest.main()