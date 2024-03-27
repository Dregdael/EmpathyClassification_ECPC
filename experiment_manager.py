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


current_dir = os.path.dirname(os.path.abspath(__file__))
#Available databases
database_dir_ec = '/processed_databases/EmpatheticConversationsExchangeFormat/'
database_dir_ex = '/processed_databases/EmpatheticExchanges/'

#Experiment parameters
experiment_number = 18
#whether to do training or use an already trained model
do_training = 0
#choose training database
train_database_dir = database_dir_ec
#choose testing database
test_database_dir = database_dir_ex
#already trained model
model_path = current_dir + '/Experiments/outputs/Experiment '+ str(experiment_number) + '/' + 'trained_pbc4cip.sav'
already_trained_model_path = current_dir + '/Experiments/outputs/Experiment '+ str(13) + '/' + 'trained_pbc4cip.sav'

feature_vector = [0,#database to classify 0 = empatheticconversations (old), 1 empatheticexchanges (new) 
                  1,#intent
                  1,#sentiment
                  1,#epitome
                  1,#vad lexicon
                  1,#length
                  1,#separated intent
                  1,#emotion
                  1,#emotion 32 -> 20
                  1,#emotion 32 -> 8
                  1,#emotion mimicry
                  ]




#Create path for experiment output
results_path = current_dir + '/Experiments/outputs/Experiment ' + str(experiment_number) + '/'
if os.path.exists(results_path):
    print('Path found')
else:
    os.mkdir(results_path)



if do_training == 1:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    trainFile = current_dir + train_database_dir + 'train.csv'
    data_train = pd.read_csv(trainFile)
    data_train["empathy"] = data_train["empathy"].astype('int')
    data_train["empathy"] = data_train["empathy"].astype('string')
    print(f'Features from the training database')
    print(data_train.columns)
    print(f'Number of datapoints in training database: {len(data_train)}')
else: 
    print(f'NO TRAINING, using model located at: {already_trained_model_path}')

testFile = current_dir + test_database_dir + 'test.csv'

data_test = pd.read_csv(testFile)

print(f'Features from the testing database')
print(data_test.columns)
print(f'Number of datapoints in testing database: {len(data_test)}')


if do_training == 1:
    trainer.train(experiment_number,data_train)
    model_path = current_dir + '/Experiments/outputs/Experiment '+ str(experiment_number) + '/' + 'trained_pbc4cip.sav'
    tester.test(experiment_number,data_test,model_path)
else: 
    tester.test(experiment_number,data_test,already_trained_model_path)







