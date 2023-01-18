from sklearn.naive_bayes import MultinomialNB
from preprocessing import split_data
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import numpy as np
from sklearn.model_selection import cross_val_score
import optuna

np.random.seed(0)







def optimize_nb(size, language, type):


    def objective(trial):

        feat = trial.suggest_int("feats", 5, len(X_train.columns)-1, log=True)

        selected = mutual_info[:feat]

        X_selected = X_train.loc[:, selected.index]

        # changed the solver to saga since it's the only one that supports l1 and multiclass problem
        clf_nb = MultinomialNB()

        # !! cross validation is called without shuffling on default in order to get the same results for every call
        score = cross_val_score(clf_nb, X_selected, y_train, n_jobs=-1)
        accuracy = score.mean()
        return accuracy


    if language == 'english':
        if size == 'normal':
            train_s = 800
        elif size == 'small':
            train_s = 100
    else:
        train_s = 800


    model = 'nb'

    dt = pd.read_csv(f"data/{language}_{type}.csv")

    X_train, y_train, X_test, y_test = split_data(dt, train_size=train_s)

    mut_ind_score = mutual_info_classif(X_train, y_train)

    mutual_info = pd.Series(mut_ind_score)
    mutual_info.index = X_train.columns
    mutual_info = mutual_info.sort_values(ascending=False)

    print("ready with  generating mutual info\n")


    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    found_params = study.trials_dataframe()
    found_params.to_csv(f"hyper_params/tuning_{language}_{type}_{size}_{model}.csv")
















# Test NB model



# selected = mutual_info[:1845]

# clf_nb = MultinomialNB()


# clf_nb.fit(X_train.loc[:,selected.index], y_train)
# print(clf_nb.score(X_test.loc[:,selected.index], y_test))


