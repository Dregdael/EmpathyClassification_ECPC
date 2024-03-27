
def get_reduced_label(vector,plutchik_8_vector,list_8_emotions,d):
    if vector.count(1) > 2:
        return d[plutchik_8_vector.index(vector)]
    elif vector.count(1) == 2:
        return d[plutchik_8_vector.index(vector)]
    else:
        return list_8_emotions[vector.index(1)]


def reduce_emotion_labels(emotion_column,dataframe):
    print(emotion_column)
    dataframe[str(emotion_column)] = dataframe[str(emotion_column)].astype('category')
    dataframe[emotion_column + "_encoded"] = dataframe[emotion_column].cat.codes
    c = dataframe[emotion_column].astype('category')
    dictionary = dict(enumerate(c.cat.categories))
    #print(d)
    #print(df_train.head())
    plutchik_emotions = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]

    plutchik_equivalencies = [[[0,0,1,0,0,0,0,0],2], #afraid
                            [[0,0,0,0,0,0,1,0],2], #angry
                            [[0,0,0,0,0,0,1,0],3], #annoyed
                            [[0,0,0,0,0,0,0,1],2], #anticipating
                            [[0,0,1,0,0,0,0,0],2], #anxious
                            [[0,0,1,0,0,0,0,0],3], #apprehensive
                            [[0,0,1,0,0,1,0,0],2], #ashamed
                            [[1,1,0,0,0,0,0,0],2], #caring
                            [[1,0,0,0,0,0,0,1],2], #confident
                            [[1,0,0,0,0,0,0,0],3], #content
                            [[0,0,0,1,1,1,0,0],1], #devastated
                            [[0,0,0,1,1,0,0,0],2], #disappointed
                            [[0,0,0,0,0,1,0,0],2], #disgusted
                            [[0,0,1,0,0,1,0,0],3], #embarassed
                            [[1,0,0,0,0,0,0,1],3], #excited
                            [[1,1,0,1,0,0,0,0],1], #faithful
                            [[0,0,0,0,0,0,1,0],1], #furious
                            [[1,1,0,1,0,0,0,0],2], #grateful
                            [[1,0,1,0,0,0,0,0],2], #guilty
                            [[0,1,0,0,0,0,0,1],2], #hopeful
                            [[0,1,0,1,0,0,0,0],1], #impressed
                            [[0,0,0,0,1,0,1,0],2], #jealous
                            [[1,0,0,0,0,0,0,0],2], #joyful
                            [[0,0,0,0,1,0,0,0],1], #lonely
                            [[1,0,0,0,1,0,0,0],2], #nostalgic
                            [[0,0,0,0,0,0,0,1],2], #prepared
                            [[1,0,0,0,0,0,1,0],2], #proud
                            [[0,0,0,0,1,0,0,0],2], #sad
                            [[0,1,0,0,0,0,0,0],2], #sentimental
                            [[0,0,0,1,0,0,0,0],2], #surprised
                            [[0,0,1,0,0,0,0,0],1], #terrified
                            [[0,1,0,0,0,0,0,0],2]] #trusting

    plutchik_equivalencies_wo_intensity = []
    for i in range(len(plutchik_equivalencies)):
        plutchik_equivalencies_wo_intensity.append(plutchik_equivalencies[i][0])

    dataframe['emotion_plutchik'] = dataframe[emotion_column + "_encoded"].apply(lambda x: plutchik_equivalencies[x][0])
    
    dataframe[emotion_column+'_reduced_labels'] = dataframe['emotion_plutchik'].apply(get_reduced_label, args = (plutchik_equivalencies_wo_intensity,plutchik_emotions,dictionary))

    dataframe = dataframe.drop(columns=['emotion_plutchik',emotion_column,emotion_column + "_encoded"])

    return dataframe

