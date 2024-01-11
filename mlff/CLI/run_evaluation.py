import argparse
import json
from mlff.config import from_config
from ml_collections import config_dict
import pathlib


def evaluate_so3krates_sparse():
    # Create the parser
    parser = argparse.ArgumentParser(description='Evaluate a SO3kratesSparse model.')
    parser.add_argument('--workdir', type=str, required=True, help='workdir')
    parser.add_argument('--num_test', type=int, required=False, default=None, help='Number of test points to use.')

    args = parser.parse_args()

    workdir = pathlib.Path(args.workdir).expanduser().absolute().resolve()

    with open(workdir / 'hyperparameters.json', 'r') as fp:
        x = json.load(fp)

    cfg = config_dict.ConfigDict(x)

    metrics = from_config.run_evaluation(config=cfg, num_test=args.num_test)
    print(metrics)


if __name__ == '__main__':
    evaluate_so3krates_sparse()
