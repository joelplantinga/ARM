import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_selection import mutual_info_classif
from preprocessing import split_data2
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from sklearn.preprocessing import StandardScaler

import time


np.random.seed(0)





def init_svm(language, type, size):

    tuning_svm = pd.read_csv(f"hyper_params/tuning_{language}_{type}_{size}_svm.csv")
    
    best_C = tuning_svm.loc[np.argmax(tuning_svm['value']), 'params_C']
    best_gamma = tuning_svm.loc[np.argmax(tuning_svm['value']), 'params_gamma']
    best_kernel = tuning_svm.loc[np.argmax(tuning_svm['value']), 'params_kernel']

    return SVC(C=best_C, gamma=best_gamma, kernel = best_kernel)

def get_nb_features(X_train, y_train, language, type, size):

    tuning_svm = pd.read_csv(f"hyper_params/tuning_{language}_{type}_{size}_nb.csv")
    num_features = tuning_svm.loc[np.argmax(tuning_svm['value']), 'params_feats']

    mut_ind_score = mutual_info_classif(X_train, y_train)

    mutual_info = pd.Series(mut_ind_score)
    mutual_info.index = X_train.columns
    mutual_info = mutual_info.sort_values(ascending=False)

    return mutual_info[:num_features]





def run_nb(language, size, ext):


    df = pd.read_csv(f"data/{language}_{ext}.csv")
    X_train, y_train, X_test, y_test = split_data2(df, train_size=800, test_size= 170)
    
    nb_feats = get_nb_features(X_train, y_train, language, ext, size)

    start = time.time()


    nb_count = MultinomialNB()
    

    nb_count.fit(X_train.loc[:, nb_feats.index], y_train)
    nb_count.predict(X_test.loc[:, nb_feats.index])

    end = time.time()
    return end - start


def run_svm(language, size, ext):


    df = pd.read_csv(f"data/{language}_{ext}.csv")

    X_train, y_train, X_test, y_test = split_data2(df, train_size=800, test_size= 170)
    
    start = time.time()


    svm = init_svm(language, ext, size)

    s = StandardScaler()


    svm.fit(s.fit_transform(X_train), y_train)
    svm.predict(s.transform(X_test))

    end = time.time()
    return end - start



def time_experiment(language, size):

    print('running the time experiment')

    results = []

    for i in range(20):

        print(f"running {i}'th iteration")

        result = {
                    f"naive_bayes_count": run_nb(language, size, 'count'), 
                    f"naive_bayes_tfidf": run_nb(language, size, 'tfidf'), 
                    f"svm_count": run_svm(language, size, 'count'), 
                    f"svm_tfidf": run_svm(language, size, 'tfidf'), 
                }
        results.append(result)

    results = pd.DataFrame(results)

    results.to_csv('results/time_experiment.csv')



def experiment(language, size):


    print(f"Running the experiment for\n LANGUAGE: {language}\n SIZE:{size}\n")

    if size == 'normal':
        train_s = 800
        test_s = 170
    elif size == 'small':
        train_s = 100
        test_s = 170

    df_count = pd.read_csv(f"data/{language}_count.csv")
    df_tfidf = pd.read_csv(f"data/{language}_count.csv")

    X_train_cnt, y_train_cnt, X_test_cnt, y_test_cnt = split_data2(df_count, train_size=train_s, test_size= test_s)
    X_train_tf, y_train_tf, X_test_tf, y_test_tf = split_data2(df_tfidf, train_size=train_s, test_size= test_s)

    nb_count_feats = get_nb_features(X_train_cnt, y_train_cnt, language, 'count', size)
    nb_tfidf_feats = get_nb_features(X_train_tf, y_train_tf, language, 'tfidf', size)

    nb_count = MultinomialNB()
    nb_tfidf = MultinomialNB()

    svm_count = init_svm(language, 'count', size)
    svm_tfidf = init_svm(language, 'tfidf', size)

    print('start fitting')

    s_1 = StandardScaler()
    s_2 = StandardScaler()



    nb_count.fit(X_train_cnt.loc[:, nb_count_feats.index], y_train_cnt)
    nb_tfidf.fit(X_train_tf.loc[:, nb_tfidf_feats.index], y_train_tf)

    svm_count.fit(s_1.fit_transform(X_train_cnt), y_train_cnt)
    svm_tfidf.fit(s_2.fit_transform(X_train_tf), y_train_tf)

    print('start predicting')

    res = pd.DataFrame.from_dict(
        {
            'svm_count' : svm_count.predict(s_1.transform(X_test_cnt)),
            'svm_tfidf' : svm_tfidf.predict(s_2.transform(X_test_tf)),
            'nb_count' : nb_count.predict(X_test_cnt.loc[:, nb_count_feats.index]),
            'nb_tfidf' : nb_tfidf.predict(X_test_tf.loc[:, nb_tfidf_feats.index]),
            'true' : y_test_cnt.values,
        }
    )

    res.to_csv(f"results/experiment_raw_{language}_{size}.csv")

    results = []

    for model in ['nb', 'svm']:

        for ext in ['count', 'tfidf']:

            accuracy = accuracy_score(res["true"], res[f"{model}_{ext}"])
            precision, recall, fscore, support = precision_recall_fscore_support(res["true"], res[f"{model}_{ext}"], average='binary')

            result = {
                        f"name": f"{model}_{ext}",
                        f"accuracy": accuracy, 
                        f"precision": precision, 
                        f"recall": recall, 
                        f"fscore": fscore
                    }
            
            results.append(result)

    results = pd.DataFrame(results)
    results.to_csv(f"results/experiment_aggregated_{language}_{size}.csv")





if __name__ == "__main__":

    time_experiment('english', 'normal')
    # experiment('english', 'normal')
    # experiment('spanish', 'normal')
    # experiment('english', 'small')





