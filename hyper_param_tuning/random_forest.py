
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
import optuna
from preprocessing import split_data, create_CV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif


import math

NGRAM = "bigram"
CLASSIFIER = "rf"

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
# def objective(trial):

#     rf_max_depth = trial.suggest_categorical("rf_max_depth", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None])
#     max_feat = trial.suggest_int("rf_max_feat", 10, 1010, 20)
#     feat = trial.suggest_int("feat", 1, mutual_floor_length , 10)

#     selected = mutual_info[:feat]

#     X_selected = X_train.loc[:, selected.index]


#     rf = RandomForestClassifier(
#         max_depth=rf_max_depth, n_estimators=80,max_features=max_feat, oob_score=True
#     )
#     rf.fit(X_selected, y_train)
    
#     return rf.oob_score_



# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=200)
# found_param = study.trials_dataframe()
# found_param.to_csv(f"results/preprocessing_{CLASSIFIER}_{NGRAM}_cv.csv")

# print(study.best_trial)






selected = mutual_info[:971]

X_selected = X_train.loc[:, selected.index]

trees = [*range(5, 500, 10)]

scores = []

for tree in trees:


    rf = RandomForestClassifier(max_depth=70, n_estimators=tree,max_features=10)

    scorelist = cross_val_score(rf, X_selected, y_train, cv=create_CV())

    print(f"{tree} with average: {sum(scorelist) / len(scorelist)}")
    scores.append(sum(scorelist) / len(scorelist))


result = pd.DataFrame({"trees": trees, "avg_accuracy": scores})
result['cumulative_accuracy'] = result['avg_accuracy'].expanding().mean()
result['cumulative_max'] = result['avg_accuracy'].expanding().max()

result.to_csv(f"results/rf_{NGRAM}_estimators_plot.csv")


