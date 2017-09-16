
# coding: utf-8

# # Evaluation
#
# The goal of this lab is to introduce you the most important techniques for evaluating your trained models. The motivation is to be able to select the model that has the best (expected) out-of-sample prediction and to assess the quality of the model.
#
# ## 1. Model Selection in a holdout setting
#
# We start with the <a href="https://en.wikipedia.org/wiki/Iris_flower_data_set">Iris</a> data set. In a nut shell the iris data set consists out of $4$ features (sepal length, sepal width, petal length, petal width) of three kinds of flowers in the iris family (iris setosa, iris versicolor, iris virginica). It was first used by Fisher to introduce linear discriminant analysis. Our version of the data set has 150 data points with 50 for each class.

# In[1]:


import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
print 'Loaded {} data points'.format(len(iris.data))

X, y = iris.data, iris.target

print 'Class labels: {}'.format(zip(range(3), iris.target_names))


# As a first example we try to classify the iris versicolor with the help of the first two features (that makes visualisation simpler as we do not know PCA yet).

# In[2]:


import numpy as np

X_versi = X[:, :2]

y_versi = np.zeros(len(y))
y_versi[y == 1] = 1


# In[3]:


plt.scatter(X_versi[:, 0], X_versi[:, 1], c=y_versi, s=70)


# The scatter plot shows that this is going to be a hard seperation problem as the classifier has to predict the red points.
#
# We split the data into a train and test (holdout) set.

# In[4]:


from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_versi, y_versi, random_state=3)


# The following function is a little visualization helper that draws the values of the decision function on a heat map given a matplotlib axe.

# In[5]:


def show_decision_function(clf, ax):
    xx, yy = np.meshgrid(np.linspace(4.5, 8, 200), np.linspace(1.5, 4.0, 200))
    try:
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    except AttributeError:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 0]

    Z = Z.reshape(xx.shape)
    ax.pcolormesh(xx, yy, Z, cmap=plt.cm.jet)
    ax.set_xlim(4.5, 8)
    ax.set_ylim(1.5, 4.0)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=100)


# In[6]:


from sklearn.svm import SVC


clf_svm = SVC(gamma=10, C=1)
clf_svm.fit(X_train, y_train)


# The scikit-learn metrics package offers the basic evaluation routines.

# In[7]:


from sklearn import metrics

y_pred = clf_svm.predict(X_test)

print "Precision: {}".format(metrics.precision_score(y_test, y_pred))
print "Recall: {}".format(metrics.recall_score(y_test, y_pred))
print "F-Score: {}".format(metrics.f1_score(y_test, y_pred))


# Recalling the definition of precision and recall the numbers mean that 2/3 of the positive predictions are correct and that 1/2 of the test iris versicolor has been found by the classifier. The F-Score is then just the arithmetic mean of both (7/12).
# 
# To plot the ROC curve the decision function needs to be explicitly evaluated. The following code block also contains a helper function to plot ROC curves.

# In[8]:


y_score = clf_svm.decision_function(X_test)



# <b>Ex. 1.1</b>: Train four different classifiers (on the train/test data we used in the prior example) and put them into the list 'clfs' (you can add elements to a list via the 'append' method. Analyse the visualisations that are created by the code blocks after (Is there a uniquely best classifier?). Set the $\gamma$ parameter to $1$ and vary the $C$. Before trying out values be sure to check the <a href="http://scikit-learn.org/stable/modules/svm.html">SVM</a> documentation in scikit-learn.
# 
# Hint: Set a name to your classifier, i.e. clf.name = "Some description" to keep track of what you have done.

# In[9]:


# Exercise 1.1
clfs = []


# In[10]:


# Solution Exercise 1.1

clfs = []
Cs = [0.1, 1, 10, 25]
for C in Cs:
    clf = SVC(gamma=1, C=C)
    clf.fit(X_train, y_train)
    clf.name = 'C = {}'.format(C)
    clfs.append(clf)


# In[11]:


# This code visualises the decision functions of the four different classifiers.

fig, axes = plt.subplots(2, 2, figsize=(20, 10))

for clf, ax in zip(clfs, axes.ravel()):
    show_decision_function(clf, ax)
    ax.set_title(clf.name)



    

# ## 2. Cross-Validation
# 
# Having only 150 samples it seems like a waste to waste 30% of the training samples into the holdout set. To avoid this we can use CV - as presented in the lecture - to trade computational power for a better use of our data. 
# 
# The following code creates a list of masks, where every mask can be used as an index set to select the test samples.

# In[13]:


def create_kfold_mask(num_samples, k):
    masks = []
    fold_size = num_samples / k
    
    for i in range(k):
        mask = np.zeros(num_samples, dtype=bool)
        mask[i*fold_size:(i+1)*fold_size] = True
        masks.append(mask)
        
    return masks

masks = create_kfold_mask(150, 10)


mask = masks[0]
print X_versi[mask] # selects the test sample
print len(X_versi[~mask]) # selects training sample, ~ is binary negation


# Since the data is sorted by the labels the $k$-fold CV will likely have trouble with class imbalances in the some cases. A random shuffle solves this problem.

# In[14]:


print y_versi
num_sample = len(X_versi)
np.random.seed(3)
permutation = np.random.permutation(num_sample)
X_versi, y_versi = X_versi[permutation], y_versi[permutation]


# <b>Ex. 2.1</b>: Implement the function scores = cv_k_fold_classifier(clf, k, X, y) that fits the classifier clf on the k-fold cvs of X and y. It returns two lists of scores: the training and test scores of each fold. Interpret the results of the code block after.

# In[15]:


# Exercise 2.1

def cv_k_fold_classifier(clf, k, X, y):
    training_scores = []
    test_scores = []
    mask_training_scores = create_kfold_mask(clf, le)
    mask = masks[k]


    return training_scores, test_scores
    

print(cv_k_fold_classifier(clf, 1, X, y))

# In[16]:
