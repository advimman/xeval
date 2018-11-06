import abc

import pandas as pd
import torch
from torch.utils.data import Dataset
import pandas as pd

from ..core.utils import get_val_input_folders
from ..core.dataset import UnsupervisedListDataset
from ..core.constants import TIME_FIELD


class ExtractorDataset(Dataset):
    def __init__(self, data_root, airports, use_images=True, use_actions=False, use_X=True):
        self._example_dirs = get_val_input_folders(data_root, airports)
        self._unsupervised_dataset = UnsupervisedListDataset(self._example_dirs, use_images=use_images, use_actions=use_actions, use_X=use_X)

    def __getitem__(self, idx):
        return self._unsupervised_dataset[idx], self._example_dirs[idx]

    def __len__(self):
        return len(self._example_dirs)


class BenchmarkDatasetBase(Dataset, abc.ABC):
    def __init__(self, input_files_list, target_files_list):
        self._input_files_list = input_files_list
        self._target_files_list = target_files_list

    def __getitem__(self, idx):
        return self._load_input(self._input_files_list[idx]), self._load_target(self._target_files_list[idx])

    def __len__(self):
        return len(self._input_files_list)

    @abc.abstractmethod
    def _load_input(self, fname):
        pass

    @abc.abstractmethod
    def _load_target(self, fname):
        pass


class BenchmarkDataset(BenchmarkDatasetBase):
    def __init__(self, input_files_list, target_files_list):
        super().__init__(input_files_list, target_files_list)

    def _load_input(self, fname):
        return torch.load(fname)

    def _load_target(self, fname):
        csv = pd.read_csv(fname)
        csv = csv.drop([TIME_FIELD], axis=1)
        target = torch.Tensor(csv.values)
        return target.squeeze()
