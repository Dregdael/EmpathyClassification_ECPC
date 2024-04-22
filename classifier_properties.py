import pickle
import pandas as pd
import torch
import os

import CEM as cem

from PBC4cip import PBC4cip
import os
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm, trange
from PBC4cip import PBC4cip
from PBC4cip.core.Evaluation import obtainAUCMulticlass
from PBC4cip.core.Helpers import get_col_dist, get_idx_val

def main():
    #setup of directories 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = current_dir + '/Experiments/outputs/Experiment 75/trained_pbc4cip.sav'

    #get model
    pbc = pickle.load(open(model_path, 'rb'))

    #predict with model
    print('model loaded')
    print(pbc.dataset.GetAttributeNames())
    print(len(pbc.EmergingPatterns))
    print(pbc.EmergingPatterns[0])
    print(pbc.EmergingPatterns[0].Counts)
        
    
if __name__ == '__main__':

    main()