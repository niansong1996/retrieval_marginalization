import json
import os
import random
import tarfile

SRC_FILE_PATH = './data/iirc/preprocessed_iirc_train.json'
SHARD_DIR = './data/iirc/preprocessed_iirc_train_shards/'
SHARD_FILE_PREFIX = 'preprocessed_iirc_train_shard'
SHARD_NUM = 4


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

if __name__ == '__main__':
    # create the shard folder
    if not os.path.exists(SHARD_DIR):
        os.makedirs(SHARD_DIR)

    with open(SRC_FILE_PATH, 'r') as f:
        dataset = json.load(f)

        # sort by number of questions for a more even split
        dataset = sorted(dataset, key=lambda json_obj: len(json_obj["questions"]), reverse=True)

        sharded_list = [[] for _ in range(SHARD_NUM)]
        for i, json_obj in enumerate(dataset):
            shard_idx = i % SHARD_NUM
            sharded_list[shard_idx].append(json_obj)

        # save it back to different sharded files
        for i in range(SHARD_NUM):
            save_file_name = f'{SHARD_FILE_PREFIX}_{i}.json'
            with open(os.path.join(SHARD_DIR, save_file_name), 'w+') as shard_f:
                questions_num = sum(map(lambda x: len(x["questions"]), sharded_list[i]))
                print(f'saving {len(sharded_list[i])} paragraphs and {questions_num} questions to file {save_file_name}')

                random.shuffle(sharded_list[i])
                json.dump(sharded_list[i], shard_f)

        # tar the file
        make_tarfile(os.path.join(SHARD_DIR, f'{SHARD_FILE_PREFIX}s.tar'))
