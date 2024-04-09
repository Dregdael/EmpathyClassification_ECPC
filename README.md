# EmpathyClassification_ECPC

Empathy Classification of exchanging conversations with pattern classifiers 

This repository is focused on classification of empathetic exchanges in accordance to their empathy levels. We use empathy components found in the literature and pattern classifiers to carry out classification.

Current implementation requires manually modifying parameters through each stage of databse preparation, training, and testing. 

To use the code in this repository it is necessary to:

1. Setup a virtual conda environment using the environment.yml file. 
2. Train the intent classifier by running '/classifiers/empathetic_intent/train.py' This classifier gives a probability distribution that the response contains one of 8 empathetic intent categories, or a neutral class. 
3. Train the EPITOME classifier by running 'classifiers/EPITOME/train.py' This classifier marks whether there are certain communication mechanisms in the utterances. These mechanisms are related to empathy.
4. Use the experiment_managet.py file to select which features for the database, which database you want to use, and the classification algorithm's properties. This file must be used to obtain a trained model. 
5. Use the jupyter notebook demo to carry out empathy classification in a scenario of your choosing. It is necessary to specify the location of your trained model. 
