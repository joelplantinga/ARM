from sklearn.naive_bayes import MultinomialNB
from preprocessing import split_data, create_CV
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import mutual_info_classif
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# np.random.seed(0)


NGRAM = "bigram"
CLASSIFIER = "nb"


dt = pd.read_csv(f"data/converted_count_{NGRAM}.csv")
dt_binary = pd.read_csv(f"data/converted_binary_{NGRAM}.csv")

X_train_bin, y_train_bin, X_test_bin, y_test_bin = split_data(dt_binary)
X_train, y_train, X_test, y_test = split_data(dt)

mut_ind_score = mutual_info_classif(X_train_bin, y_train_bin, discrete_features=True)

mutual_info = pd.Series(mut_ind_score)
mutual_info.index = X_train.columns
mutual_info = mutual_info.sort_values(ascending=False)

print("ready with  generating mutual info\n")

feats = [*range(5, len(mutual_info)-1, 10)]

scores = []

for feat in feats:

    selected = mutual_info[:feat]

    X_selected = X_train.loc[:, selected.index]

    clf_nb = MultinomialNB()

    scorelist = cross_val_score(clf_nb, X_selected, y_train, cv=create_CV())

    print(f"{feat} with average: {sum(scorelist) / len(scorelist)}")
    scores.append(sum(scorelist) / len(scorelist))



result = pd.DataFrame({"features": feats, "avg_accuracy": scores})
result.to_csv(f"results/preprocessing_nb_{NGRAM}_cv.csv")















# Test NB model



# selected = mutual_info[:1845]

# clf_nb = MultinomialNB()


# clf_nb.fit(X_train.loc[:,selected.index], y_train)
# print(clf_nb.score(X_test.loc[:,selected.index], y_test))


