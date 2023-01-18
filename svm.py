from preprocessing import split_data
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
import optuna
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

np.random.seed(0)


def optimize_svm(size, language, type):


    def objective(trial):

        C = trial.suggest_float("C", 0.001, 100000, log=True)
        gamma = trial.suggest_float('gamma', 10**-10, 1, log=True)
        kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf"])

        support_vector_machines = SVC(C=C, tol= 0.01, gamma=gamma, kernel=kernel, class_weight='balanced')

        # !! cross validation is called without shuffling on default in order to get the same results for every call
        score = cross_val_score(support_vector_machines, X_train, y_train, n_jobs=-1, cv=5)

        accuracy = score.mean()
        return accuracy



    if language == 'english':
        if size == 'normal':
            train_s = 800
        elif size == 'small':
            train_s = 100
    else:
        train_s = 800

    model = 'svm'



    print(f"model: {model}, train_s: {train_s}, language: {language}, size:{size}")


    dt = pd.read_csv(f"data/{language}_{type}.csv")

    X_train, y_train, X_test, y_test = split_data(dt, train_size=train_s)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)


    # print(f"columns x: {len(X_train.columns)}")


    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    found_params = study.trials_dataframe()
    found_params.to_csv(f"hyper_params/tuning_{language}_{type}_{size}_{model}.csv")





