from sklearn.datasets import fetch_20newsgroups
train = fetch_20newsgroups(subset='train', remove=('headers'))
test = fetch_20newsgroups(subset='test', remove=('headers'))

import sklearn.multiclass
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import nltk
import string
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer


stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    tokens = [i for i in tokens if i not in string.punctuation]
    stems = stem_tokens(tokens, stemmer)
    return stems

X = train.data
y = train.target


clf = pipeline = Pipeline([


    ('vect', CountVectorizer(ngram_range=(1, 3), stop_words='english', max_df=0.5)),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', SGDClassifier(penalty='l2', alpha=0.0001)),
])

clf.fit(X, y )

prediction = clf.predict(test.data)
labels = test.target

# generate a classification report for the test data
classif_report = sklearn.metrics.classification_report(
    labels, prediction, target_names=train.target_names
)

print("Test-data classification report")
print(classif_report)

test_acc = sklearn.metrics.accuracy_score(
    labels, prediction
)

print(test_acc)
