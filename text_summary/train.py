import pprint
import argparse

import sentencepiece as spm

import torch
import torch.nn as nn
from torch import optim

import dataset
from torch.utils.data import DataLoader

from models.transformer import Transformer

from trainer import Trainer
from trainer import MyEngine

def define_argparser():
    p = argparse.ArgumentParser()

    # p.add_argument(
    #     '--train',
    #     required=True
    # )
    # p.add_argument(
    #     '--valid',
    #     required=True
    # )
    p.add_argument(
        '--model_fn',
        help='model file name to save.'
    )
    p.add_argument(
        '--gpu_id',
        type=int,
        default=-1,
        help='GPU ID to train. Currently, GPU parallel is not supported. -1 for CPU. Default=%(default)s'
    )
    p.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Mini batch size for gradient descent. Default=%(default)s'
    )
    p.add_argument(
        '--n_epochs',
        type=int,
        default=20,
        help='Number of epochs to train. Default=%(default)s'
    )
    p.add_argument(
        '--max_length',
        type=int,
        default=5000,
        help='max_length of training sequence, default=%(default)s'
    )
    p.add_argument(
        '--dropout',
        type=float,
        default=.2,
        help='dropout rate, default=%(default)s'
    )
    p.add_argument(
        '--hidden_size',
        type=int,
        default=768,
        help='dense vector size of Transformer, default=%(default)s'
    )
    p.add_argument(
        '--n_heads',
        type=int,
        default=8,
        help='number of heads in transformer, default=%(default)s'
    )
    p.add_argument(
        '--n_enc_layer',
        type=int,
        default=6,
        help='number of layer in transformer encoder, default=%(default)s'
    )
    p.add_argument(
        '--n_dec_layer',
        type=int,
        default=8,
        help='number of layer in transformer decoder, default=%(default)s'
    )
    p.add_argument(
        '--max_grad_norm',
        type=float,
        default=5.,
        help='Threshold for gradient clipping. Default=%(default)s'
    )
    p.add_argument(
        '--lr',
        type=float,
        default=.01,
        help='Initial learning rate. Default=%(default)s'
    )

    config = p.parse_args()

    return config

def get_model(input_size, output_size, config):
    model = Transformer(
        input_size,
        output_size,
        config.hidden_size,
        config.n_heads,
        config.n_enc_layer,
        config.n_dec_layer,
        config.dropout
    ) # .to(torch.device("mps"))

    return model

def get_crit(output_size, pad_index):
    crit = nn.NLLLoss(
        weight = torch.ones(output_size),
        ignore_index = pad_index,
        reduction = 'sum'
    ) # .to(torch.device("mps"))

    return crit

def get_optimizer(model, config):
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(.9, .98))

    return optimizer

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

def main(config):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(config))

    torch.autograd.set_detect_anomaly(True)

    # train_src, train_tgt = '.'.join(config.train.split('.')[:-2] + ["src"] + ["tsv"]), '.'.join(config.train.split('.')[:-2] + ["tgt"] + ["tsv"])
    # valid_src, valid_tgt = '.'.join(config.valid.split('.')[:-2] + ["src"] + ["tsv"]), '.'.join(config.valid.split('.')[:-2] + ["tgt"] + ["tsv"])
    train_src = "../summary_data/Training/train.shuf.src.tsv"
    train_tgt = "../summary_data/Training/train.shuf.tgt.tsv"
    valid_src = "../summary_data/Validation/valid.shuf.src.tsv"
    valid_tgt = "../summary_data/Validation/valid.shuf.tgt.tsv"

    train_dataset = dataset.CustomDataset(train_src, train_tgt)
    valid_dataset = dataset.CustomDataset(valid_src, valid_tgt)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=8, collate_fn=collate, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, num_workers=8, collate_fn=collate, shuffle=True)

    vocab_file = "./src.model"
    src_vocab = spm.SentencePieceProcessor()
    src_vocab.load(vocab_file)
    vocab_file = "./tgt.model"
    tgt_vocab = spm.SentencePieceProcessor()
    tgt_vocab.load(vocab_file)

    input_size, output_size = len(src_vocab), len(tgt_vocab)
    pad_index = src_vocab.PieceToId('<PAD>')

    model = get_model(input_size, output_size, config)
    crit = get_crit(output_size, pad_index)
    optimizer = get_optimizer(model, config)

    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)
        crit.cuda(config.gpu_id)

    trainer = Trainer(MyEngine, config)
    trainer.train(model, crit, optimizer,
                  train_loader, valid_loader,
                  src_vocab, tgt_vocab,
                  config.n_epochs)

if __name__ == '__main__':
    config = define_argparser()
    main(config)