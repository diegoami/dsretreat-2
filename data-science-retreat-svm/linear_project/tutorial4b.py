
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
train = fetch_20newsgroups(subset='train', remove=('headers'))
test = fetch_20newsgroups(subset='test', remove=('headers'))

import sklearn.multiclass

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
max_acc = 0
obj_max = None

X = train.data
y = train.target

model_to_set = pipeline = Pipeline([
    ('vect', CountVectorizer(ngram_range= (1, 2),  max_df=0.55, stop_words='english')),
    ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=False, norm='l2')),
    ('clf', LogisticRegression(C=10)),
])
parameters = {
    'clf__solver': ('newton-cg', 'lbfgs', 'liblinear'),

    #'vect__max_features': (None, 5000, 10000, 50000),
    #'vect__ngram_range': ((1, 2), (1,3))  # unigrams or bigrams

   #  'tfidf__use_idf': (True, False),
   #  'tfidf__smooth_idf':  (True, False),
   #  'tfidf__norm': ('l1', 'l2'),
    #'clf__alpha': (0.00001, 0.0001, 0.001),
    #'clf__penalty': ('l2','elasticnet', 'l1'),
    # 'clf__n_iter': (10, 50, 80),
}


clf = GridSearchCV(model_to_set, param_grid=parameters,verbose=True)



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
best_parameters = clf.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))