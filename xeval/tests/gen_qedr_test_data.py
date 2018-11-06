#!/usr/bin/env python3

import os
import json
from argparse import ArgumentParser

import numpy as np
import pandas as pd


FEATURES_FILE_NAME = 'X.csv'
FACTORS_FILE_NAME = 'target.csv'
SAMPLE_DIR_PREFIX = 'flight'
SPLITS_JSON_NAME = 'splits.json'
TIME_LEN = 1013


def gen_factors(num):
    return np.random.rand(num)


def gen_entanglement_matrix(factors_num, features_num, max_entanglement=1):
    '''Output shape: ([features_num, factors_num], [features_num])'''
    coefs = np.zeros([features_num, factors_num])
    for feature in range(features_num):
        coefs[feature, np.random.choice(factors_num, max_entanglement)] = 1
        coefs[feature] *= np.random.rand(factors_num)

    bias = np.random.rand(features_num)
    return coefs, bias


def gen_sample(coefs, bias, noise_size):
    factors = gen_factors(coefs.shape[1])
    features = coefs.dot(factors) + bias + np.random.randn(bias.shape[0]) * noise_size
    features = np.tile(features, (TIME_LEN, 1))
    return features, factors


def save_sample(dir_name, coefs, bias, noise_size):
    os.makedirs(dir_name, exist_ok=True)
    features, factors = map(pd.DataFrame, gen_sample(coefs, bias, noise_size))
    features.T.to_csv(os.path.join(dir_name, FEATURES_FILE_NAME), index=False)

    dir_name = os.path.join(dir_name, 'qedr')
    os.makedirs(dir_name, exist_ok=True)
    features, factors = map(pd.DataFrame, gen_sample(coefs, bias, noise_size))
    features.T.to_csv(os.path.join(dir_name, FEATURES_FILE_NAME), index=False)
    factors.T.to_csv(os.path.join(dir_name, FACTORS_FILE_NAME), index=False)


def gen_dataset(dir_name, coefs, bias, sample_num, noise_size):
    for i in range(sample_num):
        save_sample(os.path.join(dir_name, SAMPLE_DIR_PREFIX + str(i)), coefs, bias, noise_size)


if __name__ == '__main__':
    aparser = ArgumentParser('Generator of data for QEDR benchmark')
    aparser.add_argument('--features-num', type=int, default=20)
    aparser.add_argument('--factors-num', type=int, default=5)
    aparser.add_argument('--max-entanglement', type=int, default=1)
    aparser.add_argument('--sample-num', type=int, default=10)
    aparser.add_argument('--noise-size', type=float, default=0.1)
    aparser.add_argument('--root-dir-name', default='disentanglement_synthetic')
    args = aparser.parse_args()
    coefs, bias = gen_entanglement_matrix(args.factors_num, args.features_num, args.max_entanglement)

    splits = {'train': ['fe_train'], 'val': ['fe_val'],
              'test': {'train': ['bn_train'], 'val': ['bn_val'], 'test': ['bn_test']}}
    os.makedirs(args.root_dir_name, exist_ok=True)
    with open(os.path.join(args.root_dir_name, SPLITS_JSON_NAME), 'w') as splits_file:
        json.dump(splits, splits_file)

    for type_ in ('bn', 'fe'):
        for prefix in ('train', 'test', 'val'):
            if type_ == 'fe' and prefix == 'test':
                continue
            dir_name = os.path.join(args.root_dir_name, f'{type_}_{prefix}')
            gen_dataset(dir_name, coefs, bias, args.sample_num, args.noise_size)
