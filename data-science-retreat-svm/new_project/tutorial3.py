
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
train = fetch_20newsgroups(subset='train', remove=('headers'))
test = fetch_20newsgroups(subset='test', remove=('headers'))

import sklearn.multiclass
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

from sklearn.linear_model import SGDClassifier
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

from sklearn.ensemble import RandomForestClassifier

max_acc = 0
obj_max = None
for max_df in [0.1,0,2,0.3]:

    for features in [ 8192]:
        print("features = "+str(features))

        vectorizer = TfidfVectorizer(max_features=features, stop_words='english', max_df=max_df, ngram_range=(1,2))
        label_encoder = sklearn.preprocessing.LabelBinarizer()

        X = vectorizer.fit_transform(train.data)
        y = label_encoder.fit_transform(train.target)

        model_to_set = RandomForestClassifier(n_estimators=30)

        parameters = {
           # "estimator__C": [1, 2, 4, 8],
           # "estimator__kernel": ["poly", "rbf"],
           # "estimator__degree": [1, 2, 3, 4],
            "min_samples_leaf" : [1,2,3,4,5],
                "n_estimators": [30]
        }

        clf = GridSearchCV(model_to_set, param_grid=parameters,verbose=True)



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

        if (test_acc > max_acc):
            max_acc = test_acc
            obj_max = (max_df, features)

print(max_acc)
print(obj_max)