import sklearn.metrics
import sklearn.multiclass
from sklearn.datasets import fetch_20newsgroups
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer, Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline

def get_train_test():
    train = fetch_20newsgroups(subset='train', remove=('headers'))
    test = fetch_20newsgroups(subset='test', remove=('headers'))
    return train, test


def do_lsa_experiment(n_tfidf, n_svd, C):
    train, test = get_train_test()

    svd = TruncatedSVD(n_svd)
    normalizer = Normalizer(copy=False)
    vectorizer = TfidfVectorizer(ngram_range=(1,2),
                                 max_features=n_tfidf,
                                 stop_words='english')

    lsa = make_pipeline(vectorizer, svd, normalizer)
    # newgroup (target) to multi-class label
    label_encoder = sklearn.preprocessing.LabelBinarizer()

    # linear svc one vs rest classifier

    base_clf = LinearSVC(C=C)
    clf = sklearn.multiclass.OneVsRestClassifier(base_clf)

    # encode emails in design matrix
    X = lsa.fit_transform(train.data)
    # encode email newgroups in target matrix
    y = label_encoder.fit_transform(train.target)

    # fit soft-margin linear svms to X, y
    clf.fit(X, y)

    # generate a classification report for the test data
    classif_report = sklearn.metrics.classification_report(
        label_encoder.transform(test.target),
        clf.predict(lsa.transform(test.data)),
        target_names=train.target_names
    )

    print("Test-data classification report")
    print(classif_report)

    test_acc = sklearn.metrics.accuracy_score(
        label_encoder.transform(test.target),
        clf.predict(lsa.transform(test.data))
    )

    return test_acc, classif_report

Cs = [1e0, 1e1, 1e2, 1e3]

for C in Cs:
    test_acc, _ = do_lsa_experiment(2**14, 2**9, C)
    print("Test-data overall accuracy for C=%.3f: %.3f" % (C, test_acc))
