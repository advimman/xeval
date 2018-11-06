import abc
import torch
from torch import nn
from torch.nn import functional as F

from ..core.utils import move_data_to_device


class BenchmarkModelBase(abc.ABC):
    def __init__(self, feature_size=128, output_size=2651, use_cuda=True, **model_kwargs):
        self._use_cuda = use_cuda
        self._feature_size = feature_size
        self._output_size = output_size
        self._module = self._get_module()
        if self._use_cuda:
            self._module.cuda()
        self._loss = self._get_loss()
        self._optimizer = self._get_optimizer()

    def fit_batch(self, X, y):
        X, y = move_data_to_device([X, y], self._use_cuda)
        pred = self._module(X)
        if pred.shape[1] == 1:
            pred = torch.squeeze(pred, dim=1)
        loss_value = self._loss(pred, y)
        self._optimizer.zero_grad()
        loss_value.backward()
        self._optimizer.step()
        return loss_value.cpu()

    def predict_batch(self, X):
        X = move_data_to_device(X, self._use_cuda)
        return self._module(X).cpu()

    def save(self, path):
        torch.save(self._module.state_dict(), path)

    @abc.abstractmethod
    def _get_loss(self):
        pass

    @abc.abstractmethod
    def _get_optimizer(self):
        pass

    @abc.abstractmethod
    def _get_module(self):
        pass

    def get_optimizer(self):
        return self._optimizer


class BinaryLogisticModel(BenchmarkModelBase):
    def __init__(self, feature_size, output_size, use_cuda=True, optim_lr=1e-3, weight_decay=0.1, **model_kwargs):
        self._optim_lr = optim_lr
        self._weight_decay = weight_decay
        super().__init__(feature_size=feature_size, output_size=output_size, use_cuda=use_cuda, **model_kwargs)

    def _get_loss(self):
        return F.binary_cross_entropy_with_logits

    def _get_optimizer(self):
        return torch.optim.Adam(self._module.parameters(), lr=self._optim_lr, weight_decay=self._weight_decay)

    def _get_module(self):
        return MLPModule(self._feature_size, self._output_size)

    def predict_batch(self, X):
        return F.sigmoid(super().predict_batch(X))


class MLPModule(nn.Module):
    def __init__(self, feature_size, output_size):
        super().__init__()
        self._impl = nn.Sequential(
            torch.nn.Linear(feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_size))

    def forward(self, X):
        x = self._impl(X)
        return x


class LinearModule(nn.Module):
    def __init__(self, feature_size, output_size):
        super().__init__()
        self._linear = nn.Sequential(
            torch.nn.Linear(feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_size))

    def forward(self, X):
        x = self._linear(X)
        return x


class LinearRegressionModel(BenchmarkModelBase):
    def __init__(self, feature_size=100, use_cuda=True, optim_lr=1e-3, weight_decay=0, output_size=4, **model_kwargs):
        self._optim_lr = optim_lr
        self._weight_decay = weight_decay
        super().__init__(feature_size=feature_size, output_size=output_size, use_cuda=use_cuda, **model_kwargs)

    def _get_loss(self):
        return F.mse_loss

    def _get_optimizer(self):
        return torch.optim.Adam(self._module.parameters(), lr=self._optim_lr, weight_decay=self._weight_decay)

    def _get_module(self):
        return MLPModule(self._feature_size, self._output_size)


class LassoLossModule(nn.Module):
    def __init__(self, alpha, linear):
        super().__init__()
        self.alpha = alpha
        self._linear = linear

    def forward(self, pred, y):
        reg = self._linear.weight.norm(1) + self._linear.bias.norm(1)
        return F.mse_loss(pred, y) + self.alpha * reg


class LassoRegressionModel(BenchmarkModelBase):
    def __init__(self, feature_size=10, output_size=5, alpha=1., use_cuda=True, optim_lr=1e-1):
        self._optim_lr = optim_lr
        self._linear = LinearModule(feature_size, output_size)
        self._loss = LassoLossModule(alpha, self._linear._linear)
        if use_cuda:
            self._loss = self._loss.cuda()
        super().__init__(feature_size=feature_size, use_cuda=use_cuda)

    def _get_loss(self):
        return self._loss

    def _get_optimizer(self):
        return torch.optim.Adam(self._module.parameters(), lr=self._optim_lr)

    def _get_module(self):
        return self._linear

    @property
    def rel_importances(self):
        with torch.no_grad():
            return self._get_module()._linear.weight.abs()
