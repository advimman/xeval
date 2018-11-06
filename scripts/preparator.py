import os
import glob
import argparse
import json

from tqdm import tqdm
import pandas as pd

FEATURE_SIZES = [32, 64, 128, 256]
use_configs = ["decouple_dynamics_X.json",
               "tml_config.json",
               "decouple_dynamics_im.json",
               "decouple_dynamics.json"]
static_config = []
bench_class_dict = {"BenchmarkWindRegression" : {"out": 1},
                    "BenchmarkTurbulenceRegression": {"out": 3},
                    "BenchmarkLandingRegression": {"out": 3},
                    "BenchmarkAutoRegression": {"out": 3},
                    "BenchmarkFailuresRegression": {"out": 12}}
for bench_class, bench_vals in bench_class_dict.items():
    dict_ = {
                "train_epochs": 50,
                "benchmark_class": f"xeval.benchmarks.benchmarks.{bench_class}",
                "model_kwargs": {
                    "use_cuda": True,
                    "output_size": bench_vals['out'],
                    "optim_lr": 0.003
                },
                "loader_kwargs": {
                    "batch_size": 128,
                    "shuffle": True,
                    "num_workers": 28
                }
            }
    static_config.append(dict_)


class Preparator(object):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.result_dict = {}
        self.save_path_model = "/Vol0/user/a.mashikhin/output/xplane/"

    def get_path(self):
        self.root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        self.save_path = os.path.join(self.root_dir, "constants", "dataset.json")

    def get_csv(self):
        folders = glob.glob(os.path.join(self.dataset_path, "*", "*"))
        self.example_path = [x for x in folders if os.path.isdir(x)][0]
        self.df = pd.read_csv(os.path.join(self.example_path, "X_modified.csv"))

    def get_width(self):
        width = self.df.values.shape[1]
        self.result_dict["FEATURE_EXTRACTOR_INPUT_SIZE"] = width

    def get_actions_inds(self):
        columns = []
        data_refs = ['engine', "throttle", 'angle', 'speed']
        for col in data_refs:
            columns += [x for x in self.df.columns if col in x.split("/")[-1]]

        inds = [self.df.columns.get_loc(c) for c in self.df.columns if c in columns]
        self.result_dict['ACTION_INDS'] = inds

    def get_supervised_tasks(self):
        tasks = glob.glob(os.path.join(self.example_path, "*"))
        tasks = [os.path.split(x)[-1] for x in tasks if os.path.isdir(x)]
        tasks = [x for x in tasks if "early_landing" not in x]
        self.result_dict['SUPERVISED_TASKS'] = tasks
        print(f"Found {len(tasks)} tasks: {tasks}.")

    def write_json(self):
        with open(self.save_path, "w") as f:
            json.dump(self.result_dict, f, indent=4)

    def update_benchmarks_config(self):
        self.modified_configs_names = []
        configs = [x for x in glob.glob(os.path.join(self.root_dir, "configs", "*.json"))]
        configs = [x for x in configs if os.path.split(x)[-1] in use_configs]
        for config in tqdm(configs):
            data = json.load(open(config, "r"))
            data['dataset_path'] = self.dataset_path
            for x in static_config:
                if x["benchmark_class"] == "xeval.benchmarks.benchmarks.BenchmarkAutoRegression":
                    x['model_kwargs']['output_size'] = self.result_dict["FEATURE_EXTRACTOR_INPUT_SIZE"] - 1
            data["benchmarks_list"] = static_config
            data['feature_extractor_class_kwargs']['data_loader_kwargs']['shuffle'] = True

            config_name = os.path.split(config)[-1].split(".")[0]
            for fs in FEATURE_SIZES:
                json_data = data.copy()
                json_data['feature_extractor_class_kwargs']['feature_size'] = fs
                json_data['feature_extractor_class_kwargs']['model_save_path'] = os.path.join(self.save_path_model, "models", config_name, str(fs), "weight.pth")
                json_data['benchmark_tmp_root'] = os.path.join(self.save_path_model, "features", config_name, str(fs))

                if fs != 128:
                    end_str = f"_{fs}.json"
                else:
                    end_str = ".json"
                modified_config_name = config_name + end_str
                self.modified_configs_names.append(modified_config_name)
                json_path = os.path.join(os.path.split(config)[0], modified_config_name)

                # save changes
                with open(json_path, "w") as f:
                    json.dump(json_data, f, indent=4)

        # generate script
        script_dir = os.path.split(os.path.split(config)[0])[0]
        text_file = open(os.path.join(script_dir, "train.sh"), "w")
        for name in self.modified_configs_names:
            str_ = f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -W ignore evaluate.py configs/{name}\n"
            text_file.write(str_)
        text_file.close()

    def __call__(self):
        self.get_path()
        self.get_csv()
        self.get_width()
        self.get_actions_inds()
        self.get_supervised_tasks()
        self.write_json()
        self.update_benchmarks_config()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", help="Path to dataset")

    args = parser.parse_args()
    pr = Preparator(args.dataset_path)
    pr()
