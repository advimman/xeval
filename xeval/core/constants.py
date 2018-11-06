import os
from os.path import dirname, abspath, join
import json

MIN_SEQ_LEN = 25
MAX_SEQ_LEN = 75
TIME_FIELD = 'sim/time/zulu_time_sec'

dataset_stats_path = abspath(dirname(dirname(dirname(__file__))))
dataset_stats_path = join(dataset_stats_path, "constants", "dataset.json")
dict_ = json.load(open(dataset_stats_path, "r"))
FEATURE_EXTRACTOR_INPUT_SIZE = dict_['FEATURE_EXTRACTOR_INPUT_SIZE']
ACTION_INDS = dict_['ACTION_INDS']
SUPERVISED_TASKS = dict_['SUPERVISED_TASKS']