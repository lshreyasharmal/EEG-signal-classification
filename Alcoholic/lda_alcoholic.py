import numpy as np
import random
from numpy import linalg as LA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def accuracy(y_test,result,scores):
    print str(accuracy_score(y_test,result))
    print "Accuracy with LDA-implemented on seizure : " + str(accuracy_score(y_test,result))
    print
    print "CONFUSION MATRIX "
    tn, fp, fn, tp = confusion_matrix(y_test,result).ravel()
    total = tn+fp+fn+tp
    print "	  | predicted:1 | predicted:0 |"
    print "|actual:1 " + "|tn =  %.3f  |fp =   %.3f |" %((tn),fp)
    print "|actual:0 " + "|fn =  %.3f  |tp =   %.3f |" %(fn,tp)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, scores, pos_label=1)
    roc_auc = metrics.auc(fpr,tpr)
    plt.figure()
    plt.plot(fpr,tpr,color='r',label='ROC curve (area=%0.2f'%roc_auc)
    plt.plot([0,1],[0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc ='lower right')
    plt.title("ROC CURVE")
    plt.savefig("LDA implemented.jpg")
    plt.show()
    print

x_0 = []
x_1 = []

x_train = np.load("x_train_balanced_alc.npy")
y_train = np.load("y_train_balanced_alc.npy")
x_test = np.load("x_test_balanced_alc.npy")
y_test = np.load("y_test_balanced_alc.npy")


for i in range(len(y_train)):
    if y_train[i] == 0:
        x_0.append(x_train[i])
    else:
        x_1.append(x_train[i])

x_0 = np.array(x_0)
x_1 = np.array(x_1)

mean_0 = np.mean(x_0,axis=0)
mean_1 = np.mean(x_1,axis=0)

dataset = np.concatenate((x_0,x_1))
mean = np.mean(dataset,axis =0)
Sw0= np.matmul((x_0-mean_0).T,(x_0-mean_0))
Sw1 = np.matmul((x_1-mean_1).T,(x_1-mean_1))
Sb = np.cov(dataset.T)-Sw0-Sw1
matrix0 = np.dot(LA.pinv(Sw0),Sb)
matrix1 = np.dot(LA.pinv(Sw1),Sb)
eig_vals0,eig_vecs0 = LA.eigh(matrix0)
eig_vals1,eig_vecs1 = LA.eigh(matrix1)
eiglist0 = [(eig_vals0[i], eig_vecs0[:, i]) for i in range(len(eig_vals0))]
eiglist1 = [(eig_vals1[i], eig_vecs1[:, i]) for i in range(len(eig_vals1))]
eiglist0 = sorted(eiglist0, key = lambda x : x[0], reverse = True)
eiglist1 = sorted(eiglist1, key = lambda x : x[0], reverse = True)
w0 = np.array([eiglist0[i][1] for i in range(1)])
w1 = np.array([eiglist1[i][1] for i in range(1)])
#1,178

x_0 = np.dot(x_0,w0.T)
x_1 = np.dot(x_1,w1.T)
result = []
scores = []
transposed_mean_0 = np.mean(x_0,axis=0)
transposed_mean_1 = np.mean(x_1,axis=0)

for test in x_test:
    proj_0 = np.dot(test,w0.T)
    proj_1 = np.dot(test,w1.T)
    if abs(proj_0-transposed_mean_0) < abs(proj_1-transposed_mean_1):
        result.append(0)
    else:
        result.append(1)
    scores.append((proj_0-transposed_mean_0))

result = np.array(result)
scores = np.array(scores)

print scores.shape
accuracy(y_test,result,scores)
