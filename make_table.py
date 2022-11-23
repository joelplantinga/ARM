import pandas as pd




df = pd.read_csv('results/experiment_results_aggregated_1.csv')

result = []

for clas in ['naive_bayes', 'logistic_regression', 'random_forest', 'decision_tree']:


    for ngram in ['unigram', 'bigram']:

        row = {
            'Classifier': f"{clas}",
            'Ngram':f"{ngram}",
            'Accuracy': df.loc[df['name'] == f"{clas}_{ngram}_accuracy", 'value'].values[0],
            'Precision': df.loc[df['name'] == f"{clas}_{ngram}_precision", 'value'].values[0],
            'Recall': df.loc[df['name'] == f"{clas}_{ngram}_recall", 'value'].values[0],
            'Fscore': df.loc[df['name'] == f"{clas}_{ngram}_fscore", 'value'].values[0]
            }

        result.append(row)

result = pd.DataFrame(result)
result.to_csv('results/aggregated_results_table_format_1.csv', index=False)




