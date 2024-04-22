from ctypes import Array
import os
from sklearn.metrics import accuracy_score, classification_report, f1_score
import pandas as pd
import re
import CEM as cem
from tqdm import tqdm, trange
import logging
logging.disable(logging.WARNING)


def main():
    print('Getting predictions....')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    predictions_dir = '/Experiments/outputs/Experiment 87/'
    predictions_file = 'predictions.txt'
    preds_pbc = pd.read_csv(current_dir + predictions_dir + predictions_file, sep=" ", header=None)
    preds_pbc = preds_pbc.rename(columns={preds_pbc.columns[0]: "pred_val"})
    #print(preds_pbc.head())
    #print(preds_pbc.rename(columns={"0": "1"}))
    preds_pbc['pred_val'] += 1
    print(preds_pbc.head())

    print('Getting predictions from BERT.......')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    predictions_file_bert = '/BERT_predictions_3.txt'
    preds_bert = pd.read_csv(current_dir + predictions_file_bert, sep=" ", header=None)
    preds_bert = preds_bert.rename(columns={preds_bert.columns[0]: "pred_val"})
    #print(preds_bert.rename(columns={"0": "1"}))
    preds_bert['pred_val'] = preds_bert['pred_val'] + 1
    print(preds_bert.head())

    print('Actual labels')
    train_database_dir = '/processed_databases/EmpatheticExchanges/'
    testFile = current_dir + train_database_dir + 'EmpatheticExchanges_test.csv'
    df_test = pd.read_csv(testFile)
    df_test['empathy'] = df_test['empathy'].astype('int')
    print(df_test['empathy'].iloc[:5])

    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 0, 1]
    
    print('PBC f1 score')
    print(f1_score(df_test['empathy'], preds_pbc['pred_val'], average='weighted'))
    print('BERT f1 score')
    print(f1_score(df_test['empathy'], preds_bert['pred_val'], average='weighted'))



if __name__ == '__main__':

    main()