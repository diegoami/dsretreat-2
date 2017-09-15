
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
train = fetch_20newsgroups(subset='train', remove=('headers'))
test = fetch_20newsgroups(subset='test', remove=('headers'))
# Display progress logs on stdout
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


import sklearn.multiclass
from sklearn.linear_model import SGDClassifier

max_acc = 0
obj_max = None
for max_df in [0.1]:

    for features in [ 8192]:
        print("features = "+str(features))

        vectorizer = TfidfVectorizer(max_features=features, stop_words='english', max_df=max_df, ngram_range=(1,2))
        label_encoder = sklearn.preprocessing.LabelBinarizer()

        X = vectorizer.fit_transform(train.data)
        y = label_encoder.fit_transform(train.target)

        Cs = [1]

        for C in Cs:

        #clf = sklearn.multiclass.OneVsRestClassifier(MultinomialNB(alpha=alpha))
            #clf = sklearn.multiclass.OneVsRestClassifier(SGDClassifier(alpha=alpha))

            clf = sklearn.multiclass.OneVsRestClassifier(sklearn.svm.LinearSVC(C=C))


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
            print(max_df, features, C)

            if (test_acc > max_acc):
                max_acc = test_acc
                obj_max = (max_df, features, C)

print(max_acc)
print(obj_max)