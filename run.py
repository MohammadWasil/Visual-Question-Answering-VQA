# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 01:13:44 2021

@author: wasil
"""

import os
import argparse
import yaml

def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser(description='Experiment Args')

    parser.add_argument(
        '--RUN_MODE', dest='RUN_MODE',
        choices=['DATA_PROCESSING', 'TRAIN'],
        help='{Data Preprocessing, training}',
        type=str, required=True
    )

    #parser.add_argument(
    #    '--CPU', dest='CPU',
    #    help='use CPU instead of GPU',
    #    action='store_true'
    #)

    #parser.add_argument(
    #    '--MODEL', dest='MODEL',
    #    help='upload trained model',
    #    type=int
    #)

    args = parser.parse_args()
    return args


class MainExec(object):
    def __init__(self,  configs):
        #self.args = args
        self.cfgs = configs

        #if self.args.CPU:
        #    self.device = torch.device("cpu")
        #else:
        #    self.device = torch.device(
        #        "cuda" if torch.cuda.is_available() else "cpu"
        #    )  # for failsafe
        #if self.args.RUN_MODE == 'test' or self.args.RUN_MODE =='bleu':
        #    if self.args.MODEL == None:
        #        raise Exception('Add a model number you need to evaluate, e.g Model_8.pickle, then pass 8 as an argument')
        print(self.cfgs['DATA'])

if __name__ == "__main__":
    args = parse_args()

    with open('./config.yml', 'r') as f:
        model_config = yaml.safe_load(f)
    
    exec = MainExec( model_config)
    #exec.run(args.RUN_MODE)