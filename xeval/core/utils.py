import collections
import os
import importlib
import glob

import torch
from tqdm import tqdm
import numpy as np

from . import constants as core_constants


def collate_fn_for_unsupervised_dataset(batch):
    max_length = max([len(input_modalities[0]) for input_modalities in batch])
    padded_inputs = []
    for input_modality in batch[0]:
        shape = [len(batch)] + list(input_modality.shape)
        shape[1] = max_length
        padded_inputs.append(torch.zeros(*shape))

    lengths = []

    for batch_index, input_modalities in enumerate(batch):
        lengths.append(len(input_modalities[0]))
        for index, input_modality in enumerate(input_modalities):
            padded_inputs[index][batch_index, :len(input_modality)] = input_modality

    lengths = torch.from_numpy(np.array(lengths))
    return padded_inputs, lengths


def collate_fn_for_unsupervised_dataset_with_straightener(batch):
    max_length = max([len(input_modalities[0]) for input_modalities in batch])
    result = [[] for _ in range(len(batch[0]))]
    for input_modalities in batch:
        for index, input_modality in enumerate(input_modalities):
            result[index].append(input_modality)
    
    for index in range(len(result)):
        result[index] = torch.cat(result[index], dim=0).float()
    
    lengths = torch.from_numpy(np.array([0] * len(result[0])))
    return result, lengths



def collate_fn_for_feature_extractor(batch):
    unsupervised_tuples_list, fnames_list = zip(*batch)
    unsupervised_batch_tuple = collate_fn_for_unsupervised_dataset(unsupervised_tuples_list)
    return unsupervised_batch_tuple, fnames_list


def load_class(full_name):
    module_name, class_name = full_name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def move_data_to_device(data, cuda):
    if isinstance(data, torch.Tensor):
        if cuda:
            data = data.cuda()
        else:
            data = data.cpu()
        return data
    elif isinstance(data, collections.Sequence):
        return [move_data_to_device(elem, cuda) for elem in data]
    return data


def get_val_input_folders(dataset_root, airports):
    dirs_list = []
    for airport in airports:
        for task in core_constants.SUPERVISED_TASKS:
            dirs_list += [x for x in glob.glob(os.path.join(dataset_root, airport, '*', task)) if os.path.isdir(x)]
    return sorted(dirs_list)
