#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 14:21:21 2020

@author: andrew
"""


import pandas as pd
from sklearn import datasets, linear_model, model_selection, neural_network
from sklearn.linear_model import LogisticRegression;  # importing logistic regression class
import matplotlib.pyplot as plt;
import itertools;
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, roc_curve, auc, plot_roc_curve; #  importing performance metrics
from sklearn.naive_bayes import GaussianNB; #  importing naive Bayes classifier
import numpy as np;  # importing numerical computing package

df = pd.read_csv("HTRU_2.csv")

print(df)

target = df["Class"]

df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)


print(df)
n = df.shape[0];
p = df.shape[1];


print(f'Number of records:{n}');
print(f'Number of attributes:{p}');

# Particionálás tanító és taszt adatállományra
X_train, X_test, y_train, y_test = model_selection.train_test_split(df, target, test_size=0.2, random_state=2020)

print(y_train)


# Fitting logistic regression for whole dataset
logreg = LogisticRegression(solver='liblinear', max_iter=100000);  # liblineár kicsire jó, newton-cg nagyon jobb (adatmennyiség)
logreg.fit(X_train, y_train);  #  fitting the model to data
intercept = logreg.intercept_[0]; #  intecept (constant) parameter
weight = logreg.coef_[0,:];   #  regression coefficients (weights)
logreg_score = logreg.score(df, target);  # accuracy of the model
logreg_score_train = logreg.score(X_train, y_train)
logreg_score_test = logreg.score(X_test, y_test)
yprobab_logreg = logreg.predict_proba(X_test);  #  prediction probabilities
ypred_logreg_train = logreg.predict(X_train)
cm_logreg_train = confusion_matrix(y_train, ypred_logreg_train)
ypred_logreg_test = logreg.predict(X_test)
cm_logreg_test= confusion_matrix(y_test, ypred_logreg_test)

# Fitting naive Bayes classifier
naive_bayes_classifier = GaussianNB();
naive_bayes_classifier.fit(X_train,y_train);
ypred_naive_bayes = naive_bayes_classifier.predict(X_train);  # spam prediction for train
cm_naive_bayes_train = confusion_matrix(y_train, ypred_naive_bayes); # train confusion matrix
ypred_naive_bayes = naive_bayes_classifier.predict(X_test);  # spam prediction
cm_naive_bayes_test = confusion_matrix(y_test, ypred_naive_bayes); # test confusion matrix 
yprobab_naive_bayes = naive_bayes_classifier.predict_proba(X_test);  #  prediction probabilities
naive_bayes_score_train = naive_bayes_classifier.score(X_train, y_train)
naive_bayes_score_test = naive_bayes_classifier.score(X_test, y_test)

def plot_cm(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

target_names=["non-pulsar", "pulsar"]

plt.figure(1);
plot_cm(cm_logreg_train, classes=target_names,
        title = 'Confusion matrix for training dataset (logistic regression)', normalize=True);
plt.show();


plt.figure(2);
plot_cm(cm_logreg_test, classes=target_names,
   title='Confusion matrix for test dataset (logistic regression)', normalize=True);
plt.show();

plt.figure(3);
plot_cm(cm_naive_bayes_train, classes=target_names,
    title='Confusion matrix for training dataset (naive Bayes)', normalize=True);
plt.show();

plt.figure(4);
plot_cm(cm_naive_bayes_test, classes=target_names,
   title='Confusion matrix for test dataset (naive Bayes)', normalize=True);
plt.show();

# ROC Curve

# Plotting ROC curve
plot_roc_curve(logreg, X_test, y_test);
plot_roc_curve(naive_bayes_classifier, X_test, y_test);

fpr_logreg, tpr_logreg, _ = roc_curve(y_test, yprobab_logreg[:,1]);
roc_auc_logreg = auc(fpr_logreg, tpr_logreg);

fpr_naive_bayes, tpr_naive_bayes, _ = roc_curve(y_test, yprobab_naive_bayes[:,1]);
roc_auc_naive_bayes = auc(fpr_naive_bayes, tpr_naive_bayes);

plt.figure(5);
lw = 2;
plt.plot(fpr_logreg, tpr_logreg, color='red',
         lw=lw, label='Logistic regression (AUC = %0.2f)' % roc_auc_logreg);
plt.plot(fpr_naive_bayes, tpr_naive_bayes, color='blue',
         lw=lw, label='Naive Bayes (AUC = %0.2f)' % roc_auc_naive_bayes);
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--');
plt.xlim([0.0, 1.0]);
plt.ylim([0.0, 1.05]);
plt.xlabel('False Positive Rate');
plt.ylabel('True Positive Rate');
plt.title('Receiver operating characteristic curve');
plt.legend(loc="lower right");
plt.show();