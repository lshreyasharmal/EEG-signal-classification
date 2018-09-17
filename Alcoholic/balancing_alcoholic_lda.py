import numpy as np
from sklearn.preprocessing import StandardScaler
import random
from sklearn.model_selection import train_test_split
label_0 = []
label_1 = []
Output = np.load("labels_alcoholic.npy")
# PCA :-
# with open("pca_Alcoholic.csv") as file:
#     i = 0
#     for line in file:
#         line = line.strip("\n")
#         line = line.strip("")
#         line = line.split()
#         line = [float(x) for x in line]
#
#         if Output[i] == 0:
#             line.append(0)
#             label_0.append(line)
#         else:
#             line.append(1)
#             label_1.append(line)
#         i+=1

#Not PCA :-
data = np.load("data_alcoholic_LDA.npy")


for i in range(data.shape[0]):
    temp=[]
    if data[i].shape[0]==0:
        continue
    data[i]=data[i].reshape((1,))
    if Output[i]==0:
        temp.append(data[i])
        temp.append(0)
        label_0.append(temp)
    else:
        temp.append(data[i])
        print temp
        temp.append(1)
        label_1.append(temp)


label_0 = np.array(label_0)
label_1 = np.array(label_1)
print label_0.shape
print label_1.shape
# label_1 = random.sample(label_1,2000)
# label_1 = np.array(label_1)
# label_0 = random.sample(label_0,2000)
# label_0 = np.array(label_0)
dataset = np.concatenate((label_0,label_1))
dataset = np.asarray(dataset)
print dataset[0].shape
labels = dataset[:,dataset.shape[1]-1]
dataset = np.delete(dataset,dataset.shape[1]-1,1)
print dataset[0].shape
d = []
for i in range(dataset.shape[0]):
    d.append(dataset[i][0])
# print d[0]
d = np.array(d)
# d = dataset
dataset = StandardScaler().fit_transform(d)
print dataset.shape

print labels
x_train, x_test, y_train, y_test = train_test_split(dataset,labels,test_size = 0.3, random_state=42)
np.save("x_train_balanced_alc",x_train)
np.save("x_test_balanced_alc",x_test)
np.save("y_train_balanced_alc",y_train)
np.save("y_test_balanced_alc",y_test)
print x_train.shape
print y_test