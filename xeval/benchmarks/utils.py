import os
import logging
import json
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import benchmark_datasets as datasets
from ..core.utils import collate_fn_for_feature_extractor
from ..core.utils import load_class
from ..core.utils import move_data_to_device

LOGGER = logging.getLogger()


def prepare_features(extractor_instance, loader_instance, data_root, tmp_root, use_cuda=False):
    LOGGER.info("Extracting features from test set")
    for (data, lengths), fnames in tqdm(loader_instance):
        data = move_data_to_device(data, use_cuda)
        lengths = move_data_to_device(lengths, use_cuda)
        features = extractor_instance.extract(data, lengths)
        for feature, fname in zip(features, fnames):
            base_path = fname.replace(data_root, '').strip('/')
            new_path = os.path.join(tmp_root, base_path, 'features')
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            torch.save(feature.cpu().data, new_path)


def run_benchmarks(extractor_instance, config):
    splits = json.load(open(os.path.join(config['dataset_path'], "splits.json"), "r"))
    features_loader = DataLoader(datasets.ExtractorDataset(config['dataset_path'],
                                                           splits['test']['train'] + splits['test']['val'],
                                                           config['feature_extractor_train_dataset_kwargs']['use_images'],
                                                           config['feature_extractor_train_dataset_kwargs'].get('use_actions'),
                                                           config['feature_extractor_train_dataset_kwargs'].get('use_X'),
                                                           ),
                                 collate_fn=collate_fn_for_feature_extractor,
                                 **config['extractor_loader_kwargs'])

    start_time = time.time()
    prepare_features(extractor_instance=extractor_instance,
                     loader_instance=features_loader,
                     data_root=config['dataset_path'],
                     tmp_root=config['benchmark_tmp_root'],
                     use_cuda=config['feature_extractor_class_kwargs']['use_cuda'])
    inference_time = round(time.time() - start_time)

    benchmark_scores = {}
    for benchmark in config['benchmarks_list']:
        bname = benchmark['benchmark_class'].split('.')[-1]
        LOGGER.info(f"Running benchmark {bname}")
        benchmark_root = os.path.join(config['benchmark_tmp_root'], 'benchmarks', bname)
        os.makedirs(benchmark_root, exist_ok=True)
        bclass = load_class(benchmark['benchmark_class'])
        benchmark['model_kwargs']['feature_size'] = config['feature_extractor_class_kwargs']['feature_size']
        benchmark_scores[bname] = bclass(benchmark_root=benchmark_root,
                                         input_root=config['benchmark_tmp_root'],
                                         dataset_path=config['dataset_path'],
                                         model_kwargs=benchmark['model_kwargs'],
                                         loader_kwargs=benchmark['loader_kwargs'],
                                         train_epochs=benchmark['train_epochs']).run_benchmark()

    LOGGER.info(f"Inference time: {inference_time} seconds.")
    return benchmark_scores
