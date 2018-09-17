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

x_test = np.load('x_test_transformed_balanced.npy')
x_train = np.load('x_train_transformed_balanced.npy')
y_test = np.load('y_test_balanced.npy')
y_train = np.load('y_train_balanced.npy')
for i in range(x_train.shape[0]):
	for j in range(x_train.shape[1]):
		x_train[i][j] = int(x_train[i][j])
for i in range(x_test.shape[0]):
	for j in range(x_test.shape[1]):
		x_test[i][j] = int(x_test[i][j])
minn = float('inf')
maxx = float('-inf')
for i in range(x_train.shape[0]):
	for j in range(x_train.shape[1]):
		if x_train[i][j]<minn:
			minn = x_train[i][j]
		if x_train[i][j]>maxx:
			maxx = x_train[i][j]
for i in range(x_test.shape[0]):
	for j in range(x_test.shape[1]):
		if x_test[i][j]<minn:
			minn = x_test[i][j]
		if x_test[i][j]>maxx:
			maxx = x_test[i][j]
non_seizures_count = {}
seizures_count = {}
x_train_seizure=[]
x_train_non_seizure=[]
for i in range(int(minn),int(maxx)+1):
	seizures_count[i]=0
	non_seizures_count[i]=0
for i in range(x_train.shape[0]):
	if(y_train[i]==0):
		x_train_non_seizure.append(x_train[i,:])
	else:
		x_train_seizure.append(x_train[i,:])
x_train_seizure=np.array(x_train_seizure)
x_train_non_seizure=np.array(x_train_non_seizure)
start_probabilities = np.zeros(2,dtype = float)
start_probabilities[0] = x_train_non_seizure.shape[0]/float(x_train.shape[0])
start_probabilities[1] = x_train_seizure.shape[0]/float(x_train.shape[0])
transition_probs = np.array([[1,0],[0,1]])
for i in range(x_train_seizure.shape[0]):
	for j in range(x_train_seizure.shape[1]):
		seizures_count[x_train_seizure[i][j]]+=1
for i in range(x_train_non_seizure.shape[0]):
	for j in range(x_train_non_seizure.shape[1]):
		non_seizures_count[x_train_non_seizure[i][j]]+=1
for key in seizures_count:
	seizures_count[key]/=float(x_train_seizure.shape[0]*x_train_seizure.shape[1])
for key in non_seizures_count:
	non_seizures_count[key]/=float(x_train_non_seizure.shape[0]*x_train_non_seizure.shape[1])
observation_probs = {}
observation_probs[0] = non_seizures_count
observation_probs[1] = seizures_count
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
for i in range(x_test.shape[0]):
	k = viterbi(start_probabilities,observation_probs,transition_probs,x_test[i,:])[0]
	if k==y_test[i]:
		accuracy+=1
		if(y_test[i]==1):
			tp+=1
		else:
			tn+=1
	else:
		if(y_test[i]==0):
			fp+=1
		else:
			fn+=1

		# print labels[i]


print accuracy/float(x_test.shape[0])
print tp,fp,tn,fn
