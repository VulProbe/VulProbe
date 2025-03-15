import os
import sys
import random
import logging

import torch
import numpy as np
from transformers import HfArgumentParser

from prettytable import PrettyTable

from args import ProgramArguments
from line_probe import train, test

def main(args):
    args.dataset_path = os.path.join(args.dataset_path, args.lang)
    
    if args.do_train:
        train(args=args)
    elif args.customize:
        test(args=args)

if __name__ == '__main__':
    parser = HfArgumentParser(ProgramArguments)
    args = parser.parse_args()
    
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    if args.seed > 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    
    console = logging.StreamHandler()
    console.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    if args.probe_saved_path is not None and args.probe_name is not None:
        args.output_path = os.path.join(args.probe_saved_path, args.probe_name)
        os.makedirs(args.output_path, exist_ok=True)

        file = logging.FileHandler(os.path.join(args.output_path, 'info.log'))
        file.setLevel(level=logging.INFO)
        formatter = logging.Formatter('[%(asctime)s | %(filename)s | line %(lineno)d] - %(levelname)s: %(message)s')
        file.setFormatter(formatter)
        logger.addHandler(file)

    logger.info('COMMAND: {}'.format(' '.join(sys.argv)))

    config_table = PrettyTable()
    config_table.field_names = ["Configuration", "Value"]
    config_table.align["Configuration"] = "l"
    config_table.align["Value"] = "l"
    for config, value in vars(args).items():
        config_table.add_row([config, str(value)])
    logger.info('Configuration:\n{}'.format(config_table))
    print(os.getcwd())
    main(args)