import unittest

SRC_FILE_PATH = './data/iirc/preprocessed_iirc_train.json'
SHARD_DIR = './data/iirc/preprocessed_iirc_train_shards/'
SHARD_FILE_PREFIX = 'preprocessed_iirc_train_shard'


class CrossReaderTest(unittest.TestCase):
    def test_file_integrity(self):
        pass
