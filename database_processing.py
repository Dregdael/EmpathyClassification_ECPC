from math import isnan
import os, os.path
import pandas as pd
import torch
from torch import cuda
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer,pipeline

#sentiment
from classifiers.sentiment import sentiment_prediction as sp
#intent
from classifiers.empathetic_intent import intent_prediction as ip


def get_emp_intent(dataframe_row,mdl,tokenzr,dev):
    #dev = device
    if dataframe_row['is_response'] > 0:
        intent = dataframe_row['utterance']
        #print(dataframe_row['utterance'])
        intent = ip.get_empathetic_intent(str(dataframe_row['utterance']),mdl,tokenzr,dev)
        #print(dataframe_row['utterance'])
    else:
        intent = -1
    return intent


def get_sentiment_label(dataframe_row,mdl,tokenzr):
    #gets the sentiment in accordance to the label we want
    #0 - negative, 1 - neutral, 2 - positive
    label_val = ['negative','neutral', 'positive']
    sentiment_lst = sp.get_sentiment(str(dataframe_row['utterance']),mdl,tokenzr)
    index_max = max(range(len(sentiment_lst)), key=sentiment_lst.__getitem__)
    return label_val[index_max]



    

def is_responde(utterance_id):
    return int((utterance_id %2 == 0))

def fill_dataframe(dataframe):
    #This method fills empty spaces in the dataframe based on the conversation id, context, and prompt. 
    #Additionally, adds back 'utterance_idx' that says which utterance within the conversation it is, and is_response, which says if it is a response
    dataframe['utterance_idx'] = 0
    utt_idx = 0
    current_conv_id = ''
    current_context = ''
    current_evaluation = 0
    current_prompt = ''
    for i in range(len(dataframe)):
        if (str(dataframe.loc[i, 'conv_id']) != 'nan') and (str(dataframe.loc[i, 'context']) != 'nan') and (str(dataframe.loc[i, 'prompt']) != 'nan'):
            dataframe.loc[i, 'utterance_idx'] = 1
            utt_idx = 1
            current_conv_id = dataframe.loc[i, 'conv_id']
            current_context = dataframe.loc[i, 'context']
            current_prompt = dataframe.loc[i, 'prompt']
            current_evaluation = dataframe.loc[i, 'evaluation']
        else:
            utt_idx+=1
            dataframe.loc[i, 'conv_id'] = current_conv_id
            dataframe.loc[i, 'context'] = current_context
            dataframe.loc[i, 'prompt'] = current_prompt
            dataframe.loc[i, 'evaluation'] = current_evaluation
            dataframe.loc[i, 'utterance_idx'] = utt_idx
            #print(dataframe['conv_id'][i])
    
    dataframe['is_response'] = dataframe['utterance_idx'].apply(is_responde)
    return dataframe



def prepare_for_epitome(dataframe):
    dataframe['seeker_post'] = ''
    dataframe['response_post'] = ''
    conversation_ids = dataframe.conv_id.unique()
    epitome_df = pd.DataFrame()
    for i in conversation_ids:
        convo = dataframe[dataframe['conv_id'] == str(i)]

        #Ignore the last extra utterance from the speaker, it is unnecessary
        if len(convo)%2 != 0:
            convo = convo[:-1]
        #For every utterance in the index, if it is a "listener post", we get the exchange and annotate it. 
        for i in convo.index:
            if(convo.loc[i,'utterance_idx'] %2 == 0):
                convo.loc[i, 'seeker_post'] = convo.loc[i-1, 'utterance']
                convo.loc[i, 'response_post'] = convo.loc[i, 'utterance']
        epitome_df = pd.concat([epitome_df,convo])

    epitome_df = epitome_df[epitome_df['is_response'] != 0]
    dfcolumns = dataframe.columns.to_list()
    dfcolumns.remove('seeker_post')
    dfcolumns.remove('response_post')
    #print(dfcolumns)
    epitome_df = epitome_df.drop(columns=dfcolumns)

    return epitome_df



def main():
    print('Start!')

    #setup subdirectory of data samples
    dataSubDir = './data_samples/'
    empIntSubDir = './classifiers/empathetic_intent/'

    #get all files
    file_list = [name for name in os.listdir(dataSubDir) if os.path.isfile(dataSubDir+name)]

    #create empty dataframe
    df = pd.DataFrame()

    #get all datasets, process them, and join them
    for file in file_list:
        temp_df = pd.read_excel(dataSubDir+file, engine="odf")
        #set up from format given to evaluators to full dataframe
        temp_df = fill_dataframe(temp_df)

        #concatenate the datasets
        df = pd.concat([df,temp_df])
        df.reset_index(drop=True, inplace=True)
    
    
    #Check if there are any bad evaluations.
    if len(df[df['evaluation'].isin([1,2,3,4,5]) == False]) > 0:
        print('Error: Database contains bad evaluations, manually check the following conversations')
        print(df[df['evaluation'].isin([1,2,3,4,5]) == False])
        exit(1)

    #get empathetic intent
    model,tokenizer,device = ip.loadModelTokenizerAndDevice(empIntSubDir) #get model and parameters
    df['empathetic_intent'] = df.apply(get_emp_intent, axis=1, args = (model,tokenizer,device))  #apply empathetic intent extraction

    #sentiment labels
    sent_model, sent_tokenzr = sp.loadSentimentModel() #get model and tokenizer
    df['sentiment_label'] = df.apply(get_sentiment_label,axis = 1, args = (sent_model,sent_tokenzr)) #apply sentiment label extraction


    epitome_df = prepare_for_epitome(df)
    epitome_df.to_csv('EmpatheticConversations_forepitome.csv',index_label ='id')

    #Send full database to excel
    df.to_excel('EmpatheticConversations_withIntent.ods', engine="odf", index = False)
    print('Database processed successfully!')



if __name__ == '__main__':

    main()