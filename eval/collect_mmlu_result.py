import argparse
import numpy as np
from mobilellm.utils.io import json_load, json_save 

parser = argparse.ArgumentParser()
parser.add_argument('--result_path', type=str, default=None)
args = parser.parse_args()


def main():
    result = json_load(args.result_path)["results"]
    mmlu = []
    for k, v in result.items():
        if 'hendrycksTest' in k:
            mmlu.append(v['acc'])
    print('Mean accu:', np.mean(mmlu))


if __name__ == '__main__':
    main()