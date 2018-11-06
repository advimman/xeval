import abc
import logging
import traceback

import torch
from torch.utils.data import DataLoader

from .losses import ReconstructionMSELoss, TimeSeriesRegressionMSELoss, TMLLoss, DecoupleDynamicsModelLoss, ContextReconstructionMSELoss, AELoss
from .models import TimeSeriesRegressor, TML, DecoupleDynamicsModel, ContextRegressor
from .utils import train_ae_feature_extractor
from ..core.dataset import TimeSeriesDatasetSampler
from ..core.constants import ACTION_INDS
from ..core.utils import collate_fn_for_unsupervised_dataset, collate_fn_for_unsupervised_dataset_with_straightener, load_class

LOGGER = logging.getLogger()


class FeatureExtractorBase(abc.ABC):
    def __init__(self, feature_size):
        self._feature_size = feature_size

    def get_feature_size(self):
        return self._feature_size

    def train(self, train_dataset, val_dataset):
        raise NotImplementedError

    @abc.abstractmethod
    def extract(self, inputs, lengths):
        pass

    def load_weights(self):
        """If the model is already trained -- try to load trained weights"""
        try:
            self._model.load_state_dict(torch.load(self._model_save_path))
            LOGGER.info("Not training. Loading trained weights.")
        except Exception as e:
            mesg = traceback.format_exc()
            LOGGER.info(f"Not training. Can't load weigths. Error: {mesg}")


class AEFeatureExtractor(FeatureExtractorBase):
    def __init__(self, feature_size, train_epoch_count, model_save_path,
                 optimizer_kwargs, model_class, use_cuda, data_loader_kwargs, model_kwargs,
                 loss_kwargs):
        super().__init__(feature_size)
        self._model = load_class(model_class)(**model_kwargs)

        if use_cuda:
            self._model.cuda()

        self._use_cuda = use_cuda
        self._train_epoch_count = train_epoch_count
        self._model_save_path = model_save_path
        self._optimizer_kwargs = optimizer_kwargs
        self._data_loader_kwargs = data_loader_kwargs
        self._loss_kwargs = loss_kwargs

    def train(self, train_dataset, val_dataset):
        train_dataset = TimeSeriesDatasetSampler(train_dataset)
        val_dataset = TimeSeriesDatasetSampler(val_dataset)

        train_loader = DataLoader(train_dataset, collate_fn=collate_fn_for_unsupervised_dataset,
                                  **self._data_loader_kwargs)
        val_loader = DataLoader(val_dataset, collate_fn=collate_fn_for_unsupervised_dataset,
                                **self._data_loader_kwargs)
        optimizer = torch.optim.Adam(self._model.parameters(), **self._optimizer_kwargs)
        loss = ReconstructionMSELoss(**self._loss_kwargs)

        train_ae_feature_extractor(
            self._model,
            train_loader,
            val_loader,
            optimizer,
            loss,
            self._train_epoch_count,
            self._model_save_path,
            self._use_cuda)

    def extract(self, inputs, lengths):
        return self._model.get_features(inputs, lengths)


class TimeSeriesRegressorFeatureExtractor(FeatureExtractorBase):
    def __init__(self, feature_size, time_distance, train_epoch_count, model_save_path,
                 optimizer_kwargs, encoder_class, use_cuda, data_loader_kwargs, model_fc_layers, encoder_kwargs,
                 loss_kwargs):
        super().__init__(feature_size)
        self._time_distance = time_distance
        self._model = TimeSeriesRegressor(
            encoder=load_class(encoder_class)(**encoder_kwargs),
            fc_layers=model_fc_layers,
            features_size=feature_size,
            time_distance=time_distance)

        if use_cuda:
            self._model.cuda()
            self._model = torch.nn.DataParallel(self._model)

        self._use_cuda = use_cuda
        self._train_epoch_count = train_epoch_count
        self._model_save_path = model_save_path
        self._optimizer_kwargs = optimizer_kwargs
        self._data_loader_kwargs = data_loader_kwargs
        self._loss_kwargs = loss_kwargs

    def train(self, train_dataset, val_dataset):
        train_dataset = TimeSeriesDatasetSampler(train_dataset, min_length=self._time_distance + 1)
        val_dataset = TimeSeriesDatasetSampler(val_dataset, min_length=self._time_distance + 1)

        train_loader = DataLoader(train_dataset, collate_fn=collate_fn_for_unsupervised_dataset,
                                  **self._data_loader_kwargs)
        val_loader = DataLoader(val_dataset, collate_fn=collate_fn_for_unsupervised_dataset,
                                **self._data_loader_kwargs)
        optimizer = torch.optim.Adam(self._model.parameters(), **self._optimizer_kwargs)
        loss = TimeSeriesRegressionMSELoss(**self._loss_kwargs)

        train_ae_feature_extractor(
            self._model,
            train_loader,
            val_loader,
            optimizer,
            loss,
            self._train_epoch_count,
            self._model_save_path,
            self._use_cuda)

    def extract(self, inputs, lengths):
        model = self._model.module
        model.eval()
        res = model.get_features(inputs, lengths)
        return res


class TMLFeatureExtractor(FeatureExtractorBase):
    def __init__(self, feature_size, time_distance, train_epoch_count, model_save_path,
                 optimizer_kwargs, encoder_class, use_cuda, data_loader_kwargs, encoder_kwargs,
                 loss_kwargs, multigpu=True):
        super().__init__(feature_size)
        self._time_distance = time_distance
        self.multigpu = multigpu
        self._model = TML(feature_size=feature_size,
                          lstm_input_size=feature_size,
                          lstm_hidden_size=feature_size,
                          lstm_n_modalities=2,
                          lstm_n_layers=1,
                          cnn_pretrained=encoder_kwargs['cnn_pretrained'])

        if use_cuda:
            self._model.cuda()
        if multigpu:
            self._model = torch.nn.DataParallel(self._model)

        self._use_cuda = use_cuda
        self._train_epoch_count = train_epoch_count
        self._model_save_path = model_save_path
        self._optimizer_kwargs = optimizer_kwargs
        self._data_loader_kwargs = data_loader_kwargs
        self._loss_kwargs = loss_kwargs

    def train(self, train_dataset, val_dataset):
        train_dataset = TimeSeriesDatasetSampler(train_dataset, min_length=self._time_distance + 1)
        val_dataset = TimeSeriesDatasetSampler(val_dataset, min_length=self._time_distance + 1)

        train_loader = DataLoader(train_dataset, collate_fn=collate_fn_for_unsupervised_dataset,
                                  **self._data_loader_kwargs)
        val_loader = DataLoader(val_dataset, collate_fn=collate_fn_for_unsupervised_dataset,
                                **self._data_loader_kwargs)
        optimizer = torch.optim.Adam(self._model.parameters(), **self._optimizer_kwargs)
        loss = TMLLoss(**self._loss_kwargs)
        train_ae_feature_extractor(
            self._model,
            train_loader,
            val_loader,
            optimizer,
            loss,
            self._train_epoch_count,
            self._model_save_path,
            self._use_cuda)

    def extract(self, inputs, lengths):
        if self.multigpu:
            model = self._model.module
        else:
            model = self._model
        model.eval()
        res = model.get_features(inputs, lengths)
        return res


class DecoupleDynamicsModelFeatureExtractor(FeatureExtractorBase):
    def __init__(self, feature_size, time_distance, train_epoch_count, model_save_path,
                 optimizer_kwargs, encoder_class, use_cuda, data_loader_kwargs, encoder_kwargs,
                 loss_kwargs, use_images, use_X, multigpu=True):
        super().__init__(feature_size)
        self._time_distance = time_distance
        self.multigpu = multigpu
        self._model = DecoupleDynamicsModel(use_images=use_images,
                                           use_X=use_X,
                                           action_size=len(ACTION_INDS),
                                           feature_size=feature_size,
                                           lstm_input_size=feature_size,
                                           lstm_hidden_size=feature_size,
                                           lstm_n_layers=1,
                                           cnn_pretrained=encoder_kwargs['cnn_pretrained'])

        if use_cuda:
            self._model.cuda()
        if multigpu:
            self._model = torch.nn.DataParallel(self._model)

        self._use_cuda = use_cuda
        self._train_epoch_count = train_epoch_count
        self._model_save_path = model_save_path
        self._optimizer_kwargs = optimizer_kwargs
        self._data_loader_kwargs = data_loader_kwargs
        self._loss_kwargs = loss_kwargs

    def train(self, train_dataset, val_dataset):
        train_dataset = TimeSeriesDatasetSampler(train_dataset, min_length=self._time_distance + 1)
        val_dataset = TimeSeriesDatasetSampler(val_dataset, min_length=self._time_distance + 1)

        train_loader = DataLoader(train_dataset, collate_fn=collate_fn_for_unsupervised_dataset,
                                  **self._data_loader_kwargs)
        val_loader = DataLoader(val_dataset, collate_fn=collate_fn_for_unsupervised_dataset,
                                **self._data_loader_kwargs)

        optimizer = torch.optim.Adam(self._model.parameters(), **self._optimizer_kwargs)
        loss = DecoupleDynamicsModelLoss(**self._loss_kwargs)
        train_ae_feature_extractor(
            self._model,
            train_loader,
            val_loader,
            optimizer,
            loss,
            self._train_epoch_count,
            self._model_save_path,
            self._use_cuda)

    def extract(self, inputs, lengths):
        if self.multigpu:
            res = self._model.module.get_features(inputs, lengths)
        else:
            res = self._model.get_features(inputs, lengths)
        return res
    
    
class ContextFeatureExtractor(FeatureExtractorBase):
    def __init__(self, feature_size, context_size, train_epoch_count, model_save_path,
                 optimizer_kwargs, encoder_class, use_cuda, data_loader_kwargs, model_fc_layers, encoder_kwargs):
        super().__init__(feature_size)
        self._context_size = context_size
        self._model = ContextRegressor(
            encoder=load_class(encoder_class)(**encoder_kwargs),
            fc_layers=model_fc_layers,
            features_size=feature_size,
            context_size=context_size)

        if use_cuda:
            self._model.cuda()

        self._use_cuda = use_cuda
        self._train_epoch_count = train_epoch_count
        self._model_save_path = model_save_path
        self._optimizer_kwargs = optimizer_kwargs
        self._data_loader_kwargs = data_loader_kwargs

    def train(self, train_dataset, val_dataset):
        train_dataset = TimeSeriesDatasetSampler(train_dataset, min_length=self._context_size * 2 + 1)
        val_dataset = TimeSeriesDatasetSampler(val_dataset, min_length=self._context_size * 2 + 1)

        train_loader = DataLoader(train_dataset, collate_fn=collate_fn_for_unsupervised_dataset,
                                  **self._data_loader_kwargs)
        val_loader = DataLoader(val_dataset, collate_fn=collate_fn_for_unsupervised_dataset,
                                **self._data_loader_kwargs)
        optimizer = torch.optim.Adam(self._model.parameters(), **self._optimizer_kwargs)
        loss = ContextReconstructionMSELoss()

        train_ae_feature_extractor(
            self._model,
            train_loader,
            val_loader,
            optimizer,
            loss,
            self._train_epoch_count,
            self._model_save_path,
            self._use_cuda)

    def extract(self, inputs, lengths):
        return self._model.get_features(inputs, lengths)


class AE2DFeatureExtractor(FeatureExtractorBase):
    def __init__(self, feature_size, train_epoch_count, model_save_path,
                 optimizer_kwargs, use_cuda, data_loader_kwargs, model_class, model_kwargs,
                 loss_kwargs, multigpu=True):
        super().__init__(feature_size)

        self.multigpu = multigpu
        self._model = load_class(model_class)(**model_kwargs)

        if use_cuda:
            self._model.cuda()
        
        if multigpu:
            self._model = torch.nn.DataParallel(self._model)
        
        self._use_cuda = use_cuda
        self._train_epoch_count = train_epoch_count
        self._model_save_path = model_save_path
        self._optimizer_kwargs = optimizer_kwargs
        self._data_loader_kwargs = data_loader_kwargs
        self._loss_kwargs = loss_kwargs

    def train(self, train_dataset, val_dataset):
        
        train_dataset = TimeSeriesDatasetSampler(train_dataset, min_length=1, max_length=1)
        val_dataset = TimeSeriesDatasetSampler(val_dataset, min_length=1, max_length=1)

        train_loader = DataLoader(train_dataset, collate_fn=collate_fn_for_unsupervised_dataset_with_straightener,
                                  **self._data_loader_kwargs)
        val_loader = DataLoader(val_dataset, collate_fn=collate_fn_for_unsupervised_dataset_with_straightener,
                                **self._data_loader_kwargs)
        optimizer = torch.optim.Adam(self._model.parameters(), **self._optimizer_kwargs)
        loss = AELoss(**self._loss_kwargs)
        train_ae_feature_extractor(
            self._model,
            train_loader,
            val_loader,
            optimizer,
            loss,
            self._train_epoch_count,
            self._model_save_path,
            self._use_cuda)

    def extract(self, inputs, lengths):
        if self.multigpu:
            res = self._model.module.get_features(inputs, lengths)
        else:
            res = self._model.get_features(inputs, lengths)
        return res
