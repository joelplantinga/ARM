from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.feature_selection import mutual_info_classif

from copy import deepcopy
from preprocessing import split_data
import pandas as pd
import numpy as np


def calculate_mut_info(X_train_bin,y_train_bin):

    mut_ind_score = mutual_info_classif(X_train_bin,y_train_bin, discrete_features=True)

    mutual_info = pd.Series(mut_ind_score)
    mutual_info.index = X_train.columns
    mutual_info = mutual_info.sort_values(ascending=False)

    return mutual_info


ITERATIONS = 1 # number of iterations per classifier - ngram pair

used_features = {
                  'naive_bayes':
                    {
                        'unigram': 735, 
                        'bigram': 1505 
                    },
                'logistic_regression':
                    {
                        'unigram': 881, 
                        'bigram': 4621
                    }, 
                'random_forest':
                    {
                        'unigram': 641, 
                        'bigram': 971
                    },
                'decision_tree':
                    {
                        'unigram': 1010, 
                        'bigram': 441
                    } 
                }   


classifiers = { 'naive_bayes': 
                    {
                        'unigram': MultinomialNB(),
                        'bigram': MultinomialNB()
                    },
                'logistic_regression':
                    {
                        'unigram': LogisticRegression(C=4451, penalty='l1', solver='liblinear'),
                        'bigram': LogisticRegression(C=1581, penalty='l1', solver='liblinear')
                    },
                'random_forest':
                    {
                        'unigram': RandomForestClassifier(max_depth=None, n_estimators=244,max_features=30),
                        'bigram': RandomForestClassifier(max_depth=70, n_estimators=150,max_features=10)
                    },
                'decision_tree':
                    {
                        'unigram': DecisionTreeClassifier(max_depth=20,max_features=11),
                        'bigram': DecisionTreeClassifier(max_depth=60,max_features=330)
                    }
            }


exp_df = pd.DataFrame()

for ngram in ['unigram', 'bigram']:


    dt = pd.read_csv(f"data/converted_count_{ngram}.csv")
    dt_binary = pd.read_csv(f"data/converted_binary_{ngram}.csv")

    X_train_bin, y_train_bin, X_test_bin, y_test_bin = split_data(dt_binary)
    X_train, y_train, X_test, y_test = split_data(dt)

    mutual_info = calculate_mut_info(X_train_bin,y_train_bin)


    for classif_name in ['naive_bayes', 'logistic_regression', 'random_forest', 'decision_tree']:

        num_feat = used_features[classif_name][ngram]
        selected = mutual_info[:num_feat]

        results = []

        for i in range(ITERATIONS):

            classifier = deepcopy(classifiers[classif_name][ngram])

            classifier.fit(X_train.loc[:,selected.index], y_train)

            y_pred = classifier.predict(X_test.loc[:,selected.index])

            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='binary')

            result = {
                        f"{classif_name}_{ngram}_accuracy": accuracy, 
                        f"{classif_name}_{ngram}_precision": precision, 
                        f"{classif_name}_{ngram}_recall": recall, 
                        f"{classif_name}_{ngram}_fscore": fscore
                    }
            
            results.append(result)


            # corr_num = 50
            # # neg_class_prob_sorted = classifier.feature_log_prob_[0, :].argsort()[::-1]
            # # pos_class_prob_sorted = classifier.feature_log_prob_[1, :].argsort()[::-1]
            # outcome = pd.Series(np.subtract(classifier.feature_log_prob_[1, :],
            #                       classifier.feature_log_prob_[0, :])).abs().argsort()[::-1]

            # outcome_val = sorted(np.subtract(classifier.feature_log_prob_[1, :],
            #                              classifier.feature_log_prob_[0, :]), 
            #                 reverse=True, key=abs)
            
            # # print(np.take(X_train.loc[:,selected.index].columns, outcome[:10]))
            # # print(outcome_val[:10])

            # test_df = pd.DataFrame(data={
            #     "name": np.take(X_train.loc[:,selected.index].columns, outcome[:corr_num]),
            #     "value":outcome_val[:corr_num]
            # })
            # print(test_df)

            # test_df.to_csv('results/feature_class_importance.csv')



            # exit(0)
        
        print(pd.DataFrame(results))
        exp_df = pd.concat([exp_df, pd.DataFrame(results)], axis=1)



exp_df.to_csv(f"results/experiment_results_{ITERATIONS}.csv", index=False)

aggregated_df = []

for column in exp_df.columns:

    print(f"average {column}: {round(exp_df[column].mean(), 3)}")

    row = {'name': column, 'value': round(exp_df[column].mean(), 3)}

    aggregated_df.append(row)

aggregated_df = pd.DataFrame(aggregated_df)
aggregated_df.to_csv(f"results/experiment_results_aggregated_{ITERATIONS}.csv", index=False)