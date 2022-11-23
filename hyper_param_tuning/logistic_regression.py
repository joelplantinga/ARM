from sklearn.linear_model import LogisticRegression
from preprocessing import split_data, create_CV
import pandas as pd
import optuna
from sklearn.metrics import confusion_matrix, classification_report
import sklearn
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
import math
#####
### Tuning for the best parameters
#####


NGRAM = "bigram"
CLASSIFIER = "lr"

dt = pd.read_csv(f"data/converted_count_{NGRAM}.csv")
dt_binary = pd.read_csv(f"data/converted_binary_{NGRAM}.csv")

X_train_bin, y_train_bin, X_test_bin, y_test_bin = split_data(dt_binary)

X_train, y_train, X_test, y_test = split_data(dt)

mut_ind_score = mutual_info_classif(X_train_bin,y_train_bin, discrete_features=True)

mutual_info = pd.Series(mut_ind_score)
mutual_info.index = X_train.columns
mutual_info = mutual_info.sort_values(ascending=False)

mutual_floor_length = ((math.floor(len(mutual_info) / 10.0)) * 10) +1

# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):


    C = trial.suggest_int("C", 1, 5001, 10)
    feat = trial.suggest_int("feat", 1, mutual_floor_length , 10)

    selected = mutual_info[:feat]

    X_selected = X_train.loc[:, selected.index]

    LogReg = LogisticRegression( penalty='l1', solver='liblinear', C=C)

    score = cross_val_score(LogReg, X_selected, y_train, n_jobs=-1, cv=create_CV())
    accuracy = score.mean()

    return accuracy


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)
found_param = study.trials_dataframe()
found_param.to_csv(f"results/preprocessing_{CLASSIFIER}_{NGRAM}_cv.csv")

print(study.best_trial)





#####
### Doing the actual experiment
#####


# dt = pd.read_csv('data/converted.csv')
# X_train, y_train, X_test, y_test = split_data(dt)





# dt = pd.read_csv('data/converted.csv')
# X_train, y_train, X_test, y_test = split_data(dt)

# c_values = [*range(1, 5000, 10)]
# scores = []

# for c in c_values:


#     clf_nb = LogisticRegression(random_state=0, penalty='l1', solver='liblinear', C=c)

#     scorelist = sklearn.model_selection.cross_val_score(clf_nb, X_train, y_train, cv=10)

#     print(f"{c} with average: {sum(scorelist) / len(scorelist)}")
#     scores.append(sum(scorelist) / len(scorelist))



# result = pd.DataFrame({"C": c_values, "avg_accuracy": scores})
# result.to_csv('data/lr_cv_results.csv')






