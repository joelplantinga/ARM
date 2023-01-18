from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk.stem
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

LANGUAGE = 'english'

french_stemmer = nltk.stem.SnowballStemmer(LANGUAGE)

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([french_stemmer.stem(w) for w in analyzer(doc)])

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
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


def vectorize(dt, type, min_df=5, ngram_range=(1,2), binary=False):

    if LANGUAGE == 'english':
        if type == 'count':
            vectorizer = StemmedCountVectorizer(min_df=min_df, encoding='latin-1', ngram_range=ngram_range, stop_words='english', binary=binary)
        else:
            vectorizer = StemmedTfidfVectorizer(min_df=min_df, encoding='latin-1', ngram_range=ngram_range, stop_words='english', binary=binary)
    elif LANGUAGE == 'spanish':
        if type == 'count':
            vectorizer = StemmedCountVectorizer(min_df=min_df, encoding='latin-1', ngram_range=ngram_range, stop_words=stopwords.words('spanish'), binary=binary)
        else:
            vectorizer = StemmedTfidfVectorizer(min_df=min_df, encoding='latin-1', ngram_range=ngram_range, stop_words=stopwords.words('spanish'), binary=binary)

    
    vec = vectorizer.fit_transform(dt['text'].values.astype('U')).toarray()

    vectorized = pd.DataFrame(data=vec, columns=vectorizer.get_feature_names_out())

    vectorized.insert(0, 'label', dt['label'])

    return vectorized

def split_data(dt, train_size):

    dt = dt.reset_index(drop=True)

    X = dt
    y = X.pop('label')

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

    return X_train, y_train, X_test, y_test


def split_data2(dt, train_size, test_size):

    dt = dt.reset_index(drop=True)

    X = dt
    y = X.pop('label')

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=0)

    return X_train, y_train, X_test, y_test


def create_count_english():

    print("start processing count english")


    dt = pd.read_csv('data/WELFake_Dataset.csv')
    dt = dt[ ['text', 'label'] ]
    dt = vectorize(dt, type='count', ngram_range=(1,2), min_df=0.03)
    dt.to_csv(f"data/english_count.csv")

def create_tfidf_english():

    print("start processing tfidf english")

    dt = pd.read_csv('data/WELFake_Dataset.csv')
    dt = dt[ ['text', 'label'] ]
    dt = vectorize(dt, type='tfidf', ngram_range=(1,2), min_df=0.03)
    dt.to_csv(f"data/english_tfidf.csv")

def create_spanish_data():
    dt = pd.read_excel('data/train.xlsx')
    dt = dt.rename(columns={'Category': 'label', 'Text': 'text'})
    dt.to_csv("data/spanish_data.csv")


def create_count_spanish():

    dt = pd.read_csv("data/spanish_data.csv")
    dt = dt[ ['text', 'label'] ]
    dt = vectorize(dt, type='count', ngram_range=(1,2), min_df=0.03)
    dt.to_csv(f"data/spanish_count.csv")

    print(dt.shape)


def create_tfidf_spanish():

    dt = pd.read_csv("data/spanish_data.csv")
    dt = dt[ ['text', 'label'] ]
    dt = vectorize(dt, type='tfidf', ngram_range=(1,2), min_df=0.03)
    dt.to_csv(f"data/spanish_tfidf.csv")


if __name__ == "__main__":

    # create_count_spanish()
    # create_tfidf_spanish()

    create_count_english()
    create_tfidf_english()
    # binary_val = True
    # ngram_val = (1,2)
    # dt = pd.read_csv('data/original.csv')
    
    # dt = vectorize(dt, ngram_range=ngram_val, binary=binary_val, min_df=3)
    
    # dt['class_label'] = dt['class_label'].transform(lambda x: 0 if x == 'deceptive' else 1)

    # dt = dt.drop(['original_file', 'fold'], axis=1)

    # print(f"\npreprocessing produced a csv with dimennsions:\n{dt.shape}")

    # if binary_val == True:
    #     binary_print = "binary"
    # else:
    #     binary_print = "count"

    # if ngram_val == (1,1):
    #     ngram_print = "unigram"
    # else:
    #     ngram_print = "bigram"


    # dt.to_csv(f"data/converted_{binary_print}_{ngram_print}.csv", index=False)