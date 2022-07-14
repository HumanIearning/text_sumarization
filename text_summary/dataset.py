import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import tqdm
from operator import itemgetter
import sentencepiece as spm

class CustomDataset(Dataset):
    def __init__(self, src_path, tgt_path, encoding='utf-8'):
        vocab_file = "./src.model"
        src_vocab = spm.SentencePieceProcessor()
        src_vocab.load(vocab_file)
        vocab_file = "./tgt.model"
        tgt_vocab = spm.SentencePieceProcessor()
        tgt_vocab.load(vocab_file)

        with open(src_path, "r", encoding=encoding) as src_file, open(tgt_path, 'r', encoding=encoding) as tgt_file:
            self.src_lines = [src_vocab.encode_as_ids(line) for line in tqdm.tqdm(src_file, desc="Loading Dataset")]
            tgt_vocab.SetEncodeExtraOptions('bos:eos') # only attach bos, eos on target data
            self.tgt_lines = [tgt_vocab.encode_as_ids(line) for line in tqdm.tqdm(tgt_file, desc="Loading Dataset")]

        if len(self.src_lines) != len(self.tgt_lines):
            raise Exception('the number of data of src and tgt is not same')
        self.n_of_lines = len(self.src_lines)




    def __len__(self):
        return self.n_of_lines

    def __getitem__(self, idx):
        return [self.src_lines[idx], self.tgt_lines[idx]]

def collate(batch):
    batch_len = [[],[]]

    for idx in range(len(batch)):
        batch_len[0] += [len(batch[idx][0])]
        batch_len[1] += [len(batch[idx][1])]

    src_max_len = max(batch_len[0])
    tgt_max_len = max(batch_len[1])


    src, tgt = [], []

    for idx, l in enumerate(zip(batch_len[0],batch_len[1])):
            src += [batch[idx][0] + [1] * (src_max_len - l[0])]
            tgt += [batch[idx][1] + [1] * (tgt_max_len - l[1])]


    return {'src' : torch.tensor(src),
            'tgt' : torch.tensor(tgt),
            'src_len' : batch_len[0],
            'tgt_len' : batch_len[1]}



if __name__ == '__main__':
    train_src = "/Users/humanlearning/text_summarization/summary_data/Training/train.shuf.src.tsv"
    train_tgt = "/Users/humanlearning/text_summarization/summary_data/Training/train.shuf.tgt.tsv"
    valid_src = "/Users/humanlearning/text_summarization/summary_data/Validation/valid.shuf.src.tsv"
    valid_tgt = "/Users/humanlearning/text_summarization/summary_data/Validation/valid.shuf.tgt.tsv"
    train_dataset = CustomDataset(valid_src, valid_tgt)

    # print(np.argmin(np.array(dataset.tgt_lines)[:,1]))
    # print(max(dataset.tgt_lines, key=itemgetter(1)))
    # print(dataset.tgt_lines[116850])


    loader = DataLoader(train_dataset, batch_size=3, num_workers=1, collate_fn=collate, shuffle=True)
    #
    # data_iter = tqdm.tqdm(enumerate(loader),
    #                       desc="test",
    #                       total=len(loader),
    #                       )
    #
    # for i, data in data_iter:
    #     print(data)
    #     break
    for data in iter(loader):
        print(data)



        break



