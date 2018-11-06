import os
import json
import logging

from ..core.logger import setup_logger
from ..benchmarks.utils import run_benchmarks
from ..core.utils import load_class

LOGGER = logging.getLogger()


def run_evaluation(config):
    logger_path = os.path.join(config['benchmark_tmp_root'], 'logger')
    os.makedirs(config['benchmark_tmp_root'], exist_ok=True)
    setup_logger(out_file=logger_path, stderr_level=logging.DEBUG)
    LOGGER.info(config)

    feature_extractor = load_class(config["feature_extractor_class"])(**config["feature_extractor_class_kwargs"])
    splits = json.load(open(os.path.join(config['dataset_path'], "splits.json"), "r"))

    train_dataset = load_class(
        config["feature_extractor_dataset_class"])(data_path=config['dataset_path'],
                                                   airports=splits['train'],
                                                   **config["feature_extractor_train_dataset_kwargs"])

    val_dataset = load_class(
        config["feature_extractor_dataset_class"])(data_path=config['dataset_path'],
                                                   airports=splits['val'],
                                                   **config["feature_extractor_train_dataset_kwargs"])

    try:
        if config['train_extractor']:
            feature_extractor.train(train_dataset, val_dataset)
        else:
            feature_extractor.load_weights()
    except NotImplementedError:
        pass

    scores = run_benchmarks(feature_extractor, config)
    LOGGER.info("Final Benchmark scores:\n" + str(scores))
