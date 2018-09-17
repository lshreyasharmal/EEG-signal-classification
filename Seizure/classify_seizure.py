import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

def accuracy(y_test,result,classifier):
    print "Accuracy with "+ classifier +" on seizure : " + str(accuracy_score(y_test,result))
    print
    print "CONFUSION MATRIX "
    tn, fp, fn, tp = confusion_matrix(y_test,result).ravel()
    total = tn+fp+fn+tp
    print "   | predicted:1 | predicted:0 |"
    print "|actual:1 " + "|tn =  %.3f  |fp =   %.3f |" %((tn),fp)
    print "|actual:0 " + "|fn =  %.3f  |tp =   %.3f |" %(fn,tp)
    scores = gnb.predict_log_proba(x_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, scores[:,1], pos_label=1)
    roc_auc = metrics.auc(fpr,tpr)
    plt.figure()
    plt.plot(fpr,tpr,color='r',label='ROC curve (area=%0.2f'%roc_auc)
    plt.plot([0,1],[0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc ='lower right')
    plt.title("ROC CURVE")
    plt.savefig(classifier+".jpg")
    plt.show()
    print
x_train = np.load("x_train_balanced.npy")
x_test = np.load("x_test_balanced.npy")
y_train = np.load("y_train_balanced.npy")
y_test = np.load("y_test_balanced.npy")


gnb = GaussianNB()
result = gnb.fit(x_train,y_train).predict(x_test)
accuracy(y_test,result,"Naive Bayes")

gnb = LinearDiscriminantAnalysis()
result = gnb.fit(x_train,y_train).predict(x_test)
accuracy(y_test,result,"LDA")

classifier = MLPClassifier(hidden_layer_sizes = 10, max_iter = 100, activation='logistic',solver='adam',learning_rate_init=0.01,early_stopping=True)
result = classifier.fit(x_train,y_train).predict(x_test)
accuracy(y_test,result,"Nnet")
