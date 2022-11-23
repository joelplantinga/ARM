from sklearn.feature_extraction.text import CountVectorizer
import nltk.stem
import pandas as pd


french_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([french_stemmer.stem(w) for w in analyzer(doc)])


def create_CV():

    df = pd.read_csv('data/original.csv')

    df = df.loc[df['type'] == 'train', :].reset_index()

    myCViterator = []
    for i in range(1,5):
        trainIndices = df[(df['fold']!=i) & (df['type'] == 'train')].index.values.astype(int)
        testIndices =  df[(df['fold']==i) & (df['type'] == 'train')].index.values.astype(int)
        myCViterator.append( (trainIndices, testIndices) )

    return myCViterator


def vectorize(dt, min_df=5, ngram_range=(1,2), binary=False):

    vectorizer = StemmedCountVectorizer(min_df=min_df, encoding='latin-1', ngram_range=ngram_range, stop_words='english', binary=binary)

    vec = vectorizer.fit_transform(dt['comment']).toarray()

    vectorized = pd.DataFrame(data=vec, columns=vectorizer.get_feature_names_out())

    vectorized.insert(0, 'class_label', dt['class'])
    vectorized.insert(1, 'set_type', dt['type'])
    vectorized.insert(2, 'original_file', dt['filename'])

    return vectorized


def split_data(dt):


    X_train = dt.loc[dt['set_type'] == 'train', ~dt.columns.isin(['set_type'])]
    y_train = X_train.pop('class_label')


    X_test = dt.loc[dt['set_type'] == 'test', ~dt.columns.isin(['set_type'])]
    y_test = X_test.pop('class_label')

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":


    binary_val = True
    ngram_val = (1,2)
    dt = pd.read_csv('data/original.csv')
    
    dt = vectorize(dt, ngram_range=ngram_val, binary=binary_val, min_df=3)
    
    dt['class_label'] = dt['class_label'].transform(lambda x: 0 if x == 'deceptive' else 1)

    dt = dt.drop(['original_file', 'fold'], axis=1)

    print(f"\npreprocessing produced a csv with dimennsions:\n{dt.shape}")

    if binary_val == True:
        binary_print = "binary"
    else:
        binary_print = "count"

    if ngram_val == (1,1):
        ngram_print = "unigram"
    else:
        ngram_print = "bigram"


    dt.to_csv(f"data/converted_{binary_print}_{ngram_print}.csv", index=False)