import pickle
import pandas as pd
import torch
import os

from PBC4cip import PBC4cip
import os
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm, trange
from PBC4cip import PBC4cip
from PBC4cip.core.Evaluation import obtainAUCMulticlass
from PBC4cip.core.Helpers import get_col_dist, get_idx_val
import train_classifier as trainer
import test_classifier as tester
database_dir = '/processed_databases/EmpatheticConversationsExchangeFormat/'
#database_dir = '/processed_databases/EmpatheticExchanges/'

#trainer.train(database_dir,15)

tester.test(database_dir, 15)

