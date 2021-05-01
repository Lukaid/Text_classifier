import argparse
from argparse import Namespace

import torch
import torch.nn as nn
import torch.optim as optim

from simple_ntc.trainer import Trainer
from simple_ntc.data_loader import DataLoader

from simple_ntc.models.rnn import RNNClassifier
from simple_ntc.models.cnn import CNNClassifier

import os
import time


def main(config):
    loaders = DataLoader(
        train_fn=config.train_fn,
        batch_size=config.batch_size,
        min_freq=config.min_vocab_freq,
        max_vocab=config.max_vocab_size,
        device=config.gpu_id
    )

    print(
        '|train| =', len(loaders.train_loader.dataset),
        '|valid| =', len(loaders.valid_loader.dataset),
    )

    vocab_size = len(loaders.text.vocab)
    n_classes = len(loaders.label.vocab)
    print('|vocab| =', vocab_size, '|classes| =', n_classes)

    if config.rnn is False and config.cnn is False:
        raise Exception(
            'You need to specify an architecture to train. (--rnn or --cnn)')

    if config.rnn:
        # Declare model and loss.
        model = RNNClassifier(
            input_size=vocab_size,
            word_vec_size=config.word_vec_size,
            hidden_size=config.hidden_size,
            n_classes=n_classes,
            n_layers=config.n_layers,
            dropout_p=config.dropout,
        )
        optimizer = optim.Adam(model.parameters())
        crit = nn.NLLLoss()
        print(model)

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
            crit.cuda(config.gpu_id)

        rnn_trainer = Trainer(config)
        rnn_model = rnn_trainer.train(
            model,
            crit,
            optimizer,
            loaders.train_loader,
            loaders.valid_loader
        )
    if config.cnn:
        # Declare model and loss.
        model = CNNClassifier(
            input_size=vocab_size,
            word_vec_size=config.word_vec_size,
            n_classes=n_classes,
            use_batch_norm=config.use_batch_norm,
            dropout_p=config.dropout,
            window_sizes=config.window_sizes,
            n_filters=config.n_filters,
        )
        optimizer = optim.Adam(model.parameters())
        crit = nn.NLLLoss()
        print(model)

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
            crit.cuda(config.gpu_id)

        cnn_trainer = Trainer(config)
        cnn_model = cnn_trainer.train(
            model,
            crit,
            optimizer,
            loaders.train_loader,
            loaders.valid_loader
        )

    torch.save({
        'rnn': rnn_model.state_dict() if config.rnn else None,
        'cnn': cnn_model.state_dict() if config.cnn else None,
        'config': config,
        'vocab': loaders.text.vocab,
        'classes': loaders.label.vocab,
    }, config.model_fn)


# config = {
#     'model_fn': './models_/test.pth',
#     'train_fn': './data/test.refined.tok.tsv',
#     'gpu_id': 0,
#     'verbose': 2,
#     'min_vocab_freq': 5,
#     'max_vocab_size': 999999,
#     'batch_size': 256,
#     'n_epochs': 30,
#     'word_vec_size': 256,
#     'dropout': .3,
#     'max_length': 256,
#     'rnn': False,
#     'hidden_size': 512,
#     'n_layers': 4,
#     'cnn': True,
#     'use_batch_norm': True,
#     'window_sizes': [2, 3, 4],
#     'n_filters': [200, 200, 200]
# }
# config = Namespace(**config)
# print(config)
# main(config)


if __name__ == '__main__':
    for i in range(2):
        if i == 0:
            config = {
                'model_fn': './models_/review_rnn_cnn2_5_bs128_epochs_20.pth',
                'train_fn': './data/review.sorted.uniq.refined.tok.shuf.train.tsv',
                'gpu_id': 0,
                'verbose': 2,
                'min_vocab_freq': 5,
                'max_vocab_size': 999999,
                'batch_size': 128,
                'n_epochs': 20,
                'word_vec_size': 256,
                'dropout': .3,
                'max_length': 256,
                'rnn': True,
                'hidden_size': 512,
                'n_layers': 4,
                'cnn': True,
                'use_batch_norm': True,
                'window_sizes': [2, 3, 4, 5, 6],
                'n_filters': [100, 100, 100, 100, 100]
            }

            config = Namespace(**config)
            print(config)
            main(config)

        else:
            config = {
                'model_fn': './models_/review_cnn2_4_bs128_epochs_20.pth',
                'train_fn': './data/review.sorted.uniq.refined.tok.shuf.train.tsv',
                'gpu_id': 0,
                'verbose': 2,
                'min_vocab_freq': 5,
                'max_vocab_size': 999999,
                'batch_size': 128,
                'n_epochs': 20,
                'word_vec_size': 256,
                'dropout': .3,
                'max_length': 256,
                'rnn': True,
                'hidden_size': 512,
                'n_layers': 4,
                'cnn': True,
                'use_batch_norm': True,
                'window_sizes': [2, 3, 4],
                'n_filters': [100, 100, 100]
            }

            config = Namespace(**config)
            print(config)
            main(config)
