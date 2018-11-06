import os
import os.path
import random
import numpy as np
import pandas as pd
import PIL.Image as Image
import torch
from torch.utils.data.dataset import Dataset
import glob

from . import constants as core_constants


class UnsupervisedDataset(Dataset):
    def __init__(self, data_path, airports=None, use_images=False, use_actions=None, use_X=None):
        self._airports = airports
        self._use_images = use_images
        self._use_actions = use_actions
        self._use_X = use_X
        self._examples_paths = self.get_folders(data_path=data_path)

        if self._use_actions is None:
            self._use_actions = False
        if self._use_X is None:
            self._use_X = True

    def get_folders(self, data_path):
        folders = []
        for airport in self._airports:
            folders += glob.glob(os.path.join(data_path, airport, "*"))
        folders = sorted([x for x in folders if os.path.isdir(x)])
        return folders

    @staticmethod
    def get_random_inds(duration):
        if duration > core_constants.MAX_SEQ_LEN:
            start = random.randint(0, duration - core_constants.MAX_SEQ_LEN - 1)
        else:
            start = 0
        end = start + core_constants.MAX_SEQ_LEN
        return start, end

    def __getitem__(self, item):
        res = []
        example_path = self._examples_paths[item]
        data = pd.read_csv(os.path.join(example_path, "X_modified.csv"))
        start_ind, end_ind = self.get_random_inds(len(data))
        data_np = data.values[start_ind:end_ind, :]

        # remove actions from data
        if self._use_actions:
            x_width = data_np.shape[1]
            without_actions_inds = list(set(range(x_width)) - set(core_constants.ACTION_INDS))
            data_np = data_np[:, tuple(without_actions_inds)]
        res.append(torch.from_numpy(data_np))

        if self._use_images:
            images = []
            images_path = glob.glob(os.path.join(example_path, "data_*.jpg"))
            # split number from /path/data_100.jpg
            range_ = sorted([int(os.path.splitext(os.path.basename(x))[0].split('_')[-1])
                             for x in images_path])[start_ind:end_ind]

            for time_index in range_:
                image_path = os.path.join(example_path, "data_{}.jpg".format(time_index))
                image = Image.open(image_path)
                images.append(np.asarray(image) / 255.0)

            res.append(torch.from_numpy(np.array(images)))

        if self._use_actions:
            actions_data = data.diff().shift(-1).iloc[:-1]
            actions_data_val = actions_data.values[:, tuple(core_constants.ACTION_INDS)][start_ind:end_ind, :]
            res.append(torch.from_numpy(actions_data_val))

        if not self._use_X:
            res = res[1:]
        return res

    def __len__(self):
        return len(self._examples_paths)


class UnsupervisedListDataset(UnsupervisedDataset):
    def __init__(self, folders_list, use_images=False, use_actions=None, use_X=None):
        self._examples_paths = folders_list
        self._use_images = use_images
        self._use_actions = use_actions
        self._use_X = use_X

        if self._use_actions is None:
            self._use_actions = False
        if self._use_X is None:
            self._use_X = True

class TimeSeriesDatasetSampler(Dataset):
    def __init__(self, dataset, min_length=10, max_length=None):
        self._dataset = dataset
        self._min_length = min_length
        self._max_length = max_length

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, item):
        modalities = self._dataset[item]
        start_pos = random.randint(0, len(modalities[0]) - self._min_length)
        max_length = self._max_length if self._max_length else len(modalities[0])
        max_length = np.clip(max_length,
                             min(core_constants.MIN_SEQ_LEN, self._min_length),
                             max(core_constants.MAX_SEQ_LEN, self._min_length))
        end_pos = start_pos + random.randint(self._min_length, max_length)
        end_pos = min(end_pos, len(modalities[0]))

        return tuple(modality[start_pos:end_pos] for modality in modalities)
