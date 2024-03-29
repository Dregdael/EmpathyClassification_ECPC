import itertools
n=5
lst = list(map(list, itertools.product([0, 1], repeat=n)))


control_vector = [1,1,1,1,1,1,1,0,0,0,1]


'''
control_vector = [
                  1,    #database to classify 0 = empatheticconversations (old), 1 empatheticexchanges (new) 
                  1,    #intent
                  1,    #sentiment
                  1,    #epitome
                  1,    #vad lexicon
                  1,    #length
                  1,    #emotion 32
                  0,    #emotion 20
                  0,    #emotion 8
                  0,    #emotion mimicry
                  0    #reduced_empathy_labels
                  ]
'''
lst_of_control_vectors = []

for i in list(lst): 
    control_vector = [1]
    control_vector = control_vector + i[:-1] #combination of four characteristics
    control_vector.append(1) #have length
    control_vector = control_vector + [0,0,0] #have no emotion labels
    control_vector.append(i[-1]) #alternating mimicry
    control_vector.append(1) #reduced empathy labels
    lst_of_control_vectors.append(control_vector)
    #print (f'1, {i[:-1]},1 ,{' 0, 0, 0'}, {i[-1]}, 1') 
    print(control_vector)