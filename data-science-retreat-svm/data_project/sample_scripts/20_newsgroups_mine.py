from sklearn.datasets import fetch_20newsgroups
Dockerfiletrain = fetch_20newsgroups(subset='train', remove=('headers'))
test = fetch_20newsgroups(subset='test', remove=('headers'))

import pickle as cPickle

import numpy as np
import sklearn.externals.joblib

import sklearn.multiclass
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

X = train.data
y = train.target

clf = pipeline = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 2), max_df=0.55, min_df=1, stop_words='english')),
    ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=False, norm='l2')),
    ('clf', SGDClassifier(penalty='l2', alpha=5e-05)),
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

sklearn.externals.joblib.dump(clf, 'pipeline.pkl')
sklearn.externals.joblib.dump(labels, 'label_encoder.pkl')
cPickle.dump(train.target_names, open('train_target_names.pkl','w'))