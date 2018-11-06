import abc
import os
import glob
import logging
import json

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, mean_squared_error
import numpy as np

from . import benchmark_datasets as datasets
from . import benchmark_models as models
from ..core.utils import load_class

LOGGER = logging.getLogger()
EPS = 1e-10


class BenchmarkBase(abc.ABC):
    def __init__(self, benchmark_root, input_root, dataset_path, model_kwargs, loader_kwargs, train_epochs=10):
        self._input_root = input_root
        self._benchamrk_root = benchmark_root
        self._dataset_path = dataset_path
        self._train_epochs = train_epochs
        splits = json.load(open(os.path.join(dataset_path, "splits.json"), "r"))
        self.skip_counter = 0

        train_input_files = self._get_files(input_root, airports=splits['test']['train'], name="features")
        train_target_files = self._get_files(dataset_path, airports=splits['test']['train'], name='target_modified.csv')
        val_input_files = self._get_files(input_root, airports=splits['test']['val'], name="features")
        val_target_files = self._get_files(dataset_path, airports=splits['test']['val'], name='target_modified.csv')
        self._model_save_path = os.path.join(benchmark_root, 'model.pth')
        self._train_loader = DataLoader(self._get_dataset(train_input_files, train_target_files),
                                        **loader_kwargs)
        self._val_loader = DataLoader(self._get_dataset(val_input_files, val_target_files),
                                      **loader_kwargs)
        self._model = self._get_model(model_kwargs)
        self.skip_counter = self.skip_counter / 2
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._model.get_optimizer(), factor=0.5, patience=1, verbose=True)
        LOGGER.info(f"Skipped {self.skip_counter} flights for task {self._task}.")

    def run_benchmark(self):
        return self._train_model()

    def _train_model(self):
        best_val_error = np.inf
        for epoch_id in range(self._train_epochs):
            for step, (X, y) in enumerate(self._train_loader):
                loss_value = self._model.fit_batch(X, y)
                LOGGER.info(f"[Training] Epoch {epoch_id}, step {step}, loss {loss_value:5f}")

            val_error = 0
            val_examples_count = 0
            for step, (X, y) in enumerate(self._val_loader):
                pred = self._model.predict_batch(X)
                error = self._compute_error(pred, y)
                val_error += error * len(X)
                val_examples_count += len(X)
            val_error = val_error / val_examples_count
            self._scheduler.step(val_error, epoch_id)
            LOGGER.info(f"[Validation] Epoch {epoch_id}, mean error {val_error:5f}. "
                        f"Previous best error is {best_val_error:5f}")

            if val_error < best_val_error:
                best_val_error = val_error
                self._model.save(self._model_save_path)
        return best_val_error

    def _get_files(self, input_root, airports, name):
        flight_dirs = []
        for airport in airports:
            flight_dirs += glob.glob(os.path.join(input_root, airport, "*"))
        flight_files = [os.path.join(x, self._task, name) for x in flight_dirs
                        if os.path.isdir(os.path.join(x, self._task))]
        self.skip_counter += len(flight_dirs) - len(flight_files)
        return flight_files

    def _get_dataset(self, input_files, target_files):
        res = datasets.BenchmarkDataset(input_files_list=input_files,
                                         target_files_list=target_files)
        return res

    @abc.abstractmethod
    def _compute_error(self, y_pred, y_true):
        pass

    @abc.abstractmethod
    def _get_model(self, model_kwargs):
        return models.BenchmarkModelBase(**model_kwargs)


class BenchmarkLinearRegression(BenchmarkBase):
    def _get_model(self, model_kwargs):
        return models.LinearRegressionModel(**model_kwargs)

    def _compute_error(self, y_pred, y_true):
        return F.mse_loss(y_true, y_pred).data.cpu().numpy()


class BenchmarkLinearLogisticRegression(BenchmarkBase):
    def _get_model(self, model_kwargs):
        return models.BinaryLogisticModel(**model_kwargs)

    def _compute_error(self, y_pred, y_true):
        y_pred = y_pred.squeeze().cpu().detach().numpy()
        y_true = y_true.squeeze().cpu().numpy()
        return roc_auc_score(y_true, y_pred)


class BenchmarkWindRegression(BenchmarkLinearRegression):
    _task = "windflaw"


class BenchmarkAutoRegression(BenchmarkLinearRegression):
    _task = "autoregression"


class BenchmarkTurbulenceRegression(BenchmarkLinearRegression):
    _task = "turbulence"


class BenchmarkLandingRegression(BenchmarkLinearRegression):
    _task = "landing"


class BenchmarkFailuresRegression(BenchmarkLinearRegression):
    _task = "failures"


class BenchmarkQEDR(BenchmarkBase):
    '''Implementation of "A Framework for the Quantitative Evaluation of Disentangled Representations".
    https://openreview.net/pdf?id=By-7dz-AZ
    '''
    _task = "failures"

    def _train_model(self):
        best_val_error = np.inf
        for epoch_id in range(self._train_epochs):
            LOGGER.info(f"Epoch {epoch_id}, disentanglement {self.disentanglement:5f}, "
                        f"completness {self.completness:5f}")
            for step, (X, y) in enumerate(self._train_loader):
                loss_value = self._model.fit_batch(X, y)
                LOGGER.info(f"[Training] Epoch {epoch_id}, step {step}, loss {loss_value:5f}")

            val_error = 0
            val_examples_count = 0
            for step, (X, y) in enumerate(self._val_loader):
                with torch.no_grad():
                    pred = self._model.predict_batch(X)
                    error = self._compute_error(pred, y)
                val_error += error * len(X)
                val_examples_count += len(X)
            val_error = val_error / val_examples_count
            LOGGER.info(f"[Validation] Epoch {epoch_id}, mean informativeness {val_error:5f}. "
                        f"Previous best informativeness is {best_val_error:5f}")

            if val_error < best_val_error:
                best_val_error = val_error
                self._model.save(self._model_save_path)

        return {'informativeness': best_val_error.item(), 'disentanglement': self.disentanglement,
                'completness': self.completness}

    def _get_model(self, model_kwargs):
        model = load_class(model_kwargs['model'])
        del model_kwargs['model']
        return model(**model_kwargs)

    @staticmethod
    def norm_entropy(probabilities):
        n = probabilities.shape[0]
        return -probabilities.dot((probabilities + EPS).log()) / np.log(n + EPS)

    @staticmethod
    def entropic_scores(rel_importances):
        probabilities = rel_importances / (rel_importances.sum(dim=0) + EPS)
        scores = [1 - BenchmarkQEDR.norm_entropy(p) for p in probabilities.permute(1, 0)]
        return rel_importances.new_tensor(scores)

    def _compute_error(self, predicted, target):
        ''' root mean square error '''
        return (((target - predicted) ** 2).mean(dim=0).sqrt()).mean()

    @property
    def rel_importances(self):
        return self._model.rel_importances

    @property
    def disentanglement(self):
        disent_scores = BenchmarkQEDR.entropic_scores(self.rel_importances.permute(1, 0))
        code_rel_importance = self.rel_importances.sum(dim=1) / (self.rel_importances.sum() + EPS)
        return disent_scores.dot(code_rel_importance).item()

    @property
    def completness(self):
        return BenchmarkQEDR.entropic_scores(self.rel_importances).mean().item()
