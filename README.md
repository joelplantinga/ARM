# Advanced Research Methods


## Introduction

For this course we performed a fake news detection experiments using various methods. We wanted to not only test the ideal situation: english texts with plenty of training data. We also wanted to research questions that are relevant for businesses in the current digital landscape. We therefore did experiments with the Spanish language as well as we performed an experiment with limited training data.

We differentiated between using two models (i.e., Naive Bayes and Support Vector Machines) and two feature extraction methods (i.e., Count vectorization and TF-IDF). For every experiment we had results that can be analysed in order to see which is best in specific situations.

## Procedure

First using ```preprocessing.py``` data is read from the data directory, several preprocesisng techniques are applied and new files are created for the different feature extraction and language specifications.

Then using ```naive_bayes.py``` and ```svm.py``` we tuned the right hyper-parameters that can be used to to perform the experiments with. Note that ```optimizer.py``` is used as a utility file to tune models for all experiments.

With the right configured models, we could do our experiments in ```experiment.py```. these results are written to the results folder. 