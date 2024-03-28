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
import database_processing_package as data_processer

current_dir = os.path.dirname(os.path.abspath(__file__))
#Available databases
database_dir_ec = '/processed_databases/EmpatheticConversationsExchangeFormat/'
database_dir_ex = '/processed_databases/EmpatheticExchanges/'

#Experiment parameters
experiment_number = 22
#whether to do training or use an already trained model
do_training = 0
#choose training database
train_database_dir = database_dir_ex
#choose testing database
test_database_dir = database_dir_ex
#already trained model
model_path = current_dir + '/Experiments/outputs/Experiment '+ str(experiment_number) + '/' + 'trained_pbc4cip.sav'
already_trained_model_path = current_dir + '/Experiments/outputs/Experiment '+ str(20) + '/' + 'trained_pbc4cip.sav'
#whether to reprocess the database
reprocess_database = 1
#control vector for database processing
database_control_vector = [1,#database to classify 0 = empatheticconversations (old), 1 empatheticexchanges (new) 
                            1,#intent
                            1,#sentiment
                            1,#epitome
                            1,#vad lexicon
                            1,#length
                            0,#emotion 32
                            0,#emotion 20
                            0,#emotion 8
                            1,#emotion mimicry
                            ]


#If it is necessary to reprocess database, send instructions and carry out the procedure. 
if reprocess_database == 1:
    data_processer.process_database(database_control_vector)

print('')

#Create path for experiment output
results_path = current_dir + '/Experiments/outputs/Experiment ' + str(experiment_number) + '/'
if os.path.exists(results_path):
    print('WARNING: Experimental path found')
else:
    print('Creating Experimental path')
    os.mkdir(results_path)

print('')

#If you have to retrain the model, obtain training dataset
if do_training == 1:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    trainFile = current_dir + train_database_dir + 'train.csv'
    data_train = pd.read_csv(trainFile)
    data_train["empathy"] = data_train["empathy"].astype('int')
    data_train["empathy"] = data_train["empathy"].astype('string')
    data_train['mimicry'] = data_train['mimicry'].astype('category')
    print(f'Features from the training database')
    print(data_train.columns)
    print(f'Number of datapoints in training database: {len(data_train)}')
#Else, use an already trained model at a given path.
else: 
    print(f'NO TRAINING, using model located at: {already_trained_model_path}')

#Where the test file is located
testFile = current_dir + test_database_dir + 'test.csv'
#read test dataframe
data_test = pd.read_csv(testFile)


#Print features of the dataframe used for testing
print(f'Features from the testing database')
print(data_test.columns)

#Print ho wmany datapoints 
print(f'Number of datapoints in testing database: {len(data_test)}')

#If we have to train, carry out the procedure using a trainer and test the results using a test set 
if do_training == 1:
    trainer.train(experiment_number,data_train)
    model_path = current_dir + '/Experiments/outputs/Experiment '+ str(experiment_number) + '/' + 'trained_pbc4cip.sav'
    tester.test(experiment_number,data_test,model_path)
else: 
#If training is unnecessary, test the classifier on a test set 
    tester.test(experiment_number,data_test,already_trained_model_path)







