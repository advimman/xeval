import argparse
import json

from xeval.evaluation.utils import run_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config file")

    args = parser.parse_args()

    with open(args.config) as fin:
        config = json.load(fin)

    run_evaluation(config)


if __name__ == "__main__":
    main()
