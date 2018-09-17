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
#
feature_dataset = []
label = []
with open("unbalanced_seizure.csv") as file:
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
feature_dataset = StandardScaler().fit_transform(feature_dataset)
print feature_dataset[0]

pca = PCA(n_components = 178)
principalComponents = pca.fit_transform(feature_dataset)
print principalComponents.shape
std_deviation_of_principal_components = np.std(principalComponents,axis = 0)
print std_deviation_of_principal_components[0]
variance_of_principal_components = np.array([x**2 for x in std_deviation_of_principal_components])
variance_of_principal_components = np.array(variance_of_principal_components)
print variance_of_principal_components.shape
print variance_of_principal_components[0]
sum_variance_of_principal_components = np.sum(variance_of_principal_components)
print sum_variance_of_principal_components
proportion_variance_principal_components = np.array([(float(x)/sum_variance_of_principal_components) for x in variance_of_principal_components])
print proportion_variance_principal_components
print np.sum(proportion_variance_principal_components)
cumulative_scores = []
number_of_components = []
epsilon = 0.002
score = 0
num_principal_components = 0
for i in range(178):
	score+=proportion_variance_principal_components[i]
	if proportion_variance_principal_components[i]<epsilon and num_principal_components==0:
		num_principal_components=i+1
	number_of_components.append(i+1)
	cumulative_scores.append(score)
plt.plot(number_of_components,cumulative_scores)

plt.title("Plot to obtain optimal number of components")
plt.ylabel("cumulative proportion of variance of components")
plt.xlabel("Number of principal components")
plt.show()

print num_principal_components

pca = PCA(n_components = num_principal_components)
final_principalComponents = pca.fit_transform(feature_dataset)

x_train = np.load("x_train_balanced.npy")
x_test = np.load("x_test_balanced.npy")

x_train = pca.fit_transform(x_train)
x_test = pca.fit_transform(x_test)

np.save("x_train_transformed_balanced",x_train)
np.save("x_test_transformed_balanced",x_test)