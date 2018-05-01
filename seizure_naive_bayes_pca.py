import numpy as np
from sklearn.naive_bayes import GaussianNB
from numpy import genfromtxt
from sklearn.metrics import accuracy_score
import random
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
dataset = genfromtxt('pca_Seizure.csv', delimiter=',')
labels = genfromtxt('labels.csv',delimiter=',')
i_train = genfromtxt('indices.csv',delimiter=',')

trainX = []
trainY = []
testX = []
testY = []

for i in range(len(dataset)):
	if i in i_train:
		trainX.append(dataset[i])
		trainY.append(labels[i])
	else:
		testX.append(dataset[i])
		testY.append(labels[i])
np.savetxt("trainX_pca.csv", trainX, delimiter=",")
np.savetxt("trainY_pca.csv",trainY,delimiter=',')
np.savetxt("testX_pca.csv", testX, delimiter=",")
np.savetxt("testY_pca.csv",testY,delimiter=',')
gnb = GaussianNB()
result = gnb.fit(trainX,trainY).predict(testX)
print "Accuracy : " + str(accuracy_score(testY,result))

print "CONFUSION MATRIX "
tn, fp, fn, tp = confusion_matrix(testY,result).ravel()
total = tn+fp+fn+tp
print "	  | predicted:1 | predicted:0 |"
print "|actual:1 " + "|tn =  %.3f  |fp =   %.3f |" %((tn*1.0/total),fp*1.0/total)
print "|actual:0 " + "|fn =  %.3f  |tp =   %.3f |" %(fn*1.0/total,tp*1.0/total)
scores = gnb.predict_log_proba(testX)
# print scores[:,1]
fpr, tpr, thresholds = metrics.roc_curve(testY, scores[:,1], pos_label=1)
roc_auc = metrics.auc(fpr,tpr)
plt.figure()
plt.plot(fpr,tpr,color='r',label='ROC curve (area=%0.2f'%roc_auc)
plt.plot([0,1],[0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc ='lower right')
plt.title("ROC CURVE")
plt.show()
