# EmpathyClassification_ECPC

Empathy Classification of exchanging conversations with pattern classifiers 

This repository is focused on classification of empathetic exchanges in accordance to their empathy levels. We use empathy components found in the literature and pattern classifiers to carry out classification.

Current implementation requires manually modifying parameters through each stage of databse preparation, training, and testing. 

To use the code in this repository it is necessary to:

1. Train the intent classifier in '/classifiers/empathetic_intent' by running train.py
2. Train the EPITOME classifier by running 'classifiers/EPITOME/train.py'
3. Select the database to use in database_processing.py 
4. Process the database by running database_processing.py
5. Train the model by running train.py
6. Test the model by running test.py
7. Get the results, patterns, model, and predictions from the database processed folder and move them to the experiments folder to a new folder called 'outputs' 

To do:

1. Streamline the experiment process.
2. Train new emotion classifier with less labels. 
3. Modify PBC4cip implementation to accelerate training. 

