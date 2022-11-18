import argparse
import os
import subprocess
import tarfile
from datetime import datetime

from experiments import generate_experiment_cfgs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp',
        type=int,
        help='Experiment id as defined in experiment.py',
    )
    parser.add_argument(
        '--machine', type=str, default='local', choices=[
            'local',
        ])
    args = parser.parse_args()


    cfgs = generate_experiment_cfgs(args.exp)

    for i, cfg in enumerate(cfgs):
        if args.machine == 'local':
            print(f'Run config {i}/{len(cfgs)}:', cfg)
            sub_out = subprocess.run([f'cd {cfg["subfolder"]} && {cfg["EXEC_CMD"]}'], shell=True)
        else:
            raise NotImplementedError(args.machine)
