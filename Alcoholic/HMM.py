import numpy as np
import pickle
import copy
import math
from sklearn.model_selection import train_test_split

def viterbi(initial, q, p, sequence):
	K = initial.shape[0]
	T = sequence.shape[0]
	v = np.zeros((T,K))
	for k in range(K):
		# print initial[k]
		# print q[k][sequence[0]]
		v[0][k] = (initial[k]*q[k][sequence[0]])
		# print v[0][k]
	flag=False
	for i in range(1,T):
		for k in range(K):
			maxx = float('-inf')
			for j in range(K):
				if(p[j][k]*v[i-1][j] > maxx):
					maxx = p[j][k]*v[i-1][j]
			v[i][k] = (q[k][sequence[i]]*maxx)
			if(v[i][k]<1E-200):
				flag=True
		if(flag==True):
			for k in range(K):
				v[i][k]*=1E+200
			flag=False

	state_sequence = np.zeros(T, dtype=int)
	maxx = float('-inf')
	s = -1
	for k in range(K):
		if(v[T-1][k]>maxx):
			maxx = v[T-1][k]
			s = k
	state_sequence[T-1] = s
	for t in range(T-2,-1,-1):
		maxx = float('-inf')
		s = -1
		for k in range(K):
			if(v[t][k]*p[k][state_sequence[t+1]]>maxx):
				maxx = v[t][k]*p[k][state_sequence[t+1]]
				s = k
		state_sequence[t] = s
	return state_sequence

data_test = np.load('x_test_balanced_alc.npy')
data = np.load('x_train_balanced_alc.npy')
labels_test = np.load('y_test_balanced_alc.npy')
labels = np.load('y_train_balanced_alc.npy')
for i in range(data.shape[0]):
	for j in range(data.shape[1]):
		data[i][j] = int(data[i][j])
for i in range(data_test.shape[0]):
	for j in range(data_test.shape[1]):
		data_test[i][j] = int(data_test[i][j])
minn = float('inf')
maxx = float('-inf')
for i in range(data.shape[0]):
	for j in range(data.shape[1]):
		if data[i][j]<minn:
			minn = data[i][j]
		if data[i][j]>maxx:
			maxx = data[i][j]
for i in range(data_test.shape[0]):
	for j in range(data_test.shape[1]):
		if data_test[i][j]<minn:
			minn = data_test[i][j]
		if data_test[i][j]>maxx:
			maxx = data_test[i][j]


# print labels
alcoholic_counts={}
non_alcoholic_counts={}
for i in range(int(minn),int(maxx)+1):
	alcoholic_counts[i] = 0
	non_alcoholic_counts[i]=0
data_non_alcoholic = []
data_alcoholic=[]
for i in range(labels.shape[0]):
	if labels[i]==0:
		data_non_alcoholic.append(data[i,:])
	else:
		data_alcoholic.append(data[i,:])
data_non_alcoholic = np.array(data_non_alcoholic)
data_alcoholic=np.array(data_alcoholic)
start_probabilities = np.zeros(2, dtype = float)
start_probabilities[0]=(data_non_alcoholic.shape[0])
start_probabilities[1]=(data_alcoholic.shape[0])
start_probabilities=np.array(start_probabilities)
for i in range(start_probabilities.shape[0]):
	start_probabilities[i]/=float(data.shape[0])
transition_probs = np.array([[1,0],[0,1]])
for i in range(data_alcoholic.shape[0]):
	for j in range(data_alcoholic.shape[1]):
		alcoholic_counts[data[i][j]]+=1
for i in range(data_non_alcoholic.shape[0]):
	for j in range(data_non_alcoholic.shape[1]):
		non_alcoholic_counts[data[i][j]]+=1
for key in alcoholic_counts:
	alcoholic_counts[key]/=float(data_alcoholic.shape[0]*data_alcoholic.shape[1])
for key in non_alcoholic_counts:
	non_alcoholic_counts[key]/=float(data_non_alcoholic.shape[0]*data_non_alcoholic.shape[1])
observation_probs = {}
observation_probs[0] = non_alcoholic_counts
observation_probs[1] = alcoholic_counts
np.save('start.npy',start_probabilities)
np.save('tranition.npy',transition_probs)
pickle_out = open('observation_probs.pickle','wb')
pickle.dump(observation_probs,pickle_out)
pickle_out.close()
pickle_in = open('observation_probs.pickle','r')
observation_probs = pickle.load(pickle_in)
pickle_in.close()
start_probabilities = np.load('start.npy')
transition_probs = np.load('tranition.npy')
accuracy = 0
tp = 0
tn = 0
fp = 0
fn = 0
for i in range(data_test.shape[0]):
	k = viterbi(start_probabilities,observation_probs,transition_probs,data_test[i,:])[0]
	if k==labels_test[i]:
		accuracy+=1
		if(labels_test[i]==1):
			tp+=1
		else:
			tn+=1
	else:
		if(labels_test[i]==0):
			fp+=1
		else:
			fn+=1

		# print labels[i]


print accuracy/float(data_test.shape[0])
print tp,fp,tn,fn
