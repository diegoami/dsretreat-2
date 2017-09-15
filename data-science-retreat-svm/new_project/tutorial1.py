
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

import sklearn.svm
train = fetch_20newsgroups(subset='train', remove=('headers'))
test = fetch_20newsgroups(subset='test', remove=('headers'))
import sklearn
import sklearn.multiclass


label_encoder = sklearn.preprocessing.LabelBinarizer()

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(train.data)
y = label_encoder.fit_transform(train.target)

import sklearn

from sklearn.naive_bayes import MultinomialNB

for alpha in [.01,.001, .0001, .00001]:
    print("alpha = "+str(alpha))
    clf = sklearn.multiclass.OneVsRestClassifier(MultinomialNB(alpha=alpha))

    #clf = sklearn.multiclass.OneVsRestClassifier(sklearn.svm.LinearSVC())


    print(X.shape, y.shape)
    clf.fit(X, y )

    pred = clf.predict(vectorizer.transform(test.data))
    lenc = label_encoder.fit_transform(test.target)

    # generate a classification report for the test data
    classif_report = sklearn.metrics.classification_report(

        label_encoder.transform(test.target),
        clf.predict(vectorizer.transform(test.data)),
        target_names=train.target_names
    )

    print("Test-data classification report")
    print(classif_report)

    test_acc = sklearn.metrics.accuracy_score(

        label_encoder.transform(test.target),
        clf.predict(vectorizer.transform(test.data))
    )

    print(test_acc)