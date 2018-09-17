from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
feature_dataset = np.load('data_alcoholic.npy')
label = np.load('labels_alcoholic.npy')
p = np.random.permutation(11074)
feature_dataset = feature_dataset[p]
label = label[p]
l = (11074*7)/10
feature_dataset = StandardScaler().fit_transform(feature_dataset)


pca = PCA(n_components = 2000)
principalComponents = pca.fit_transform(feature_dataset)
# print principalComponents.shape
std_deviation_of_principal_components = np.std(principalComponents,axis = 0)
# print std_deviation_of_principal_components[0]
variance_of_principal_components = np.array([x**2 for x in std_deviation_of_principal_components])
# variance_of_principal_components = np.array(variance_of_principal_components)
# print variance_of_principal_components.shape
# print variance_of_principal_components[0]
sum_variance_of_principal_components = np.sum(variance_of_principal_components)
# print sum_variance_of_principal_components
proportion_variance_principal_components = np.array([(float(x)/sum_variance_of_principal_components) for x in variance_of_principal_components])
# print proportion_variance_principal_components
# print np.sum(proportion_variance_principal_components)
cumulative_scores = []
number_of_components = []
epsilon = 0.0001
score = 0
num_principal_components = 0
for i in range(2000):
	score+=proportion_variance_principal_components[i]
	if proportion_variance_principal_components[i]<epsilon and num_principal_components==0:
		# print i
		num_principal_components=i+1		
	number_of_components.append(i+1)
	cumulative_scores.append(score)

plt.plot(number_of_components,cumulative_scores)
plt.show()

pca = PCA(n_components = num_principal_components)
final_principalComponents = pca.fit_transform(feature_dataset)
np.save('pca_Alcoholic.npy',final_principalComponents)
# Training_class1 = []
# Training_class2 = []


# for i in range(len(final_principalComponents)):
# 	for j in range(len(final_principalComponents[i])):
# 		if final_principalComponents[i][j]<0:
# 			final_principalComponents[i][j]=-1
# 		else:
# 			final_principalComponents[i][j]=1
# for i in range(len(final_principalComponents)):
# 	if(label[i]==1):
# 		Training_class2.append(final_principalComponents[i])
# 	else:
# 		Training_class1.append(final_principalComponents[i])
# Training_class2=np.array(Training_class2)
# Training_class1=np.array(Training_class1)

# O=2
# Q=2
# prior1 = normalise(rand(Q,1))
# transmat1 = mk_stochastic(rand(Q,Q))
# obsmat1 = mk_stochastic(rand(Q,O))
# [LL, prior2, transmat2, obsmat2] = dhmm_em(Training_class1, prior1, transmat1, obsmat1, 'max_iter', 5)
# loglik = dhmm_logprob(Training_class1, prior2, transmat2, obsmat2)
# print loglik