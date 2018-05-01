from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.naive_bayes import GaussianNB
from numpy import genfromtxt
from sklearn.metrics import accuracy_score
import random
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt

feature_dataset = []
label = []
with open("seizure.csv") as file:
	i = 0
	for line in file:
		i+=1
		if i == 1:
			continue;
		line = line.strip("\n")
		line = line.strip("")
		line = line.split(",")
		formatted_features = [int(x) for x in line[1:len(line)-1]]
		feature_dataset.append(formatted_features)
		if (line[len(line)-1]=='2' or line[len(line)-1]=='3' or line[len(line)-1]=='4' or line[len(line)-1]=='5'):
			label.append(0)
		else:
			label.append(1)


feature_dataset = np.array(feature_dataset)

# i_train = random.sample(range(0, feature_dataset.shape[0]), int(0.7*(feature_dataset.shape[0])))
# print i_train
i_train = np.load("i.npy")
trainX = []
trainY = []
testX = []
testY = []
np.savetxt('indices.csv',i_train,delimiter=',')
for i in range(len(feature_dataset)):
	if i in i_train:
		trainX.append(feature_dataset[i])
		trainY.append(label[i])
	else:
		testX.append(feature_dataset[i])
		testY.append(label[i])
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
