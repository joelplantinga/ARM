from svm import optimize_svm
from naive_bayes import optimize_nb
import numpy as np

np.random.seed(0)




optimize_nb(language='english', type='count', size='normal')
optimize_nb(language='english', type='tfidf', size='normal')

optimize_svm(language='english', type='count', size='normal')
optimize_svm(language='english', type='tfidf', size='normal')



optimize_nb(language='spanish', type='count', size='normal')
optimize_nb(language='spanish', type='tfidf', size='normal')

optimize_svm(language='spanish', type='count', size='normal')
optimize_svm(language='spanish', type='tfidf', size='normal')



optimize_nb(language='english', type='count', size='small')
optimize_nb(language='english', type='tfidf', size='small')

optimize_svm(language='english', type='count', size='small')
optimize_svm(language='english', type='tfidf', size='small')








LANGUAGE = 'english'
TYPE = 'count'
SIZE = 'normal'
MODEL = 'svm'
