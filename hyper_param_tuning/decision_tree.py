
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
import optuna
import sklearn

from preprocessing import split_data, create_CV
import pandas as pd
import optuna
import sklearn
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
import math



NGRAM = "unigram"
CLASSIFIER = "dt"

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


    rf_max_depth = trial.suggest_categorical("rf_max_depth", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None])
    max_feat = trial.suggest_int("rf_max_feat", 10, 1010, 20)
    feat = trial.suggest_int("feat", 1, mutual_floor_length , 10)

    selected = mutual_info[:feat]

    X_selected = X_train.loc[:, selected.index]

    decTree = DecisionTreeClassifier(max_depth=rf_max_depth,max_features=max_feat)

    score = cross_val_score(decTree, X_selected, y_train, n_jobs=-1, cv=create_CV())
    accuracy = score.mean()

    return accuracy


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)
found_param = study.trials_dataframe()
found_param.to_csv(f"results/preprocessing_{CLASSIFIER}_{NGRAM}_cv.csv")

print(study.best_trial)