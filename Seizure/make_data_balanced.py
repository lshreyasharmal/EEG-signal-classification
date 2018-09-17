import numpy as np
from sklearn.preprocessing import StandardScaler
import random
from sklearn.model_selection import train_test_split
label_0 = []
label_1 = []
with open("unbalanced_seizure.csv") as file:
    i = 0
    for line in file:
        i+=1
        if i == 1:
            continue;
        line = line.strip("\n")
        line = line.strip("")
        line = line.split(",")
        formatted_features = [float(x) for x in line[1:len(line)-1]]
        if (line[len(line)-1]=='2' or line[len(line)-1]=='3' or line[len(line)-1]=='4' or line[len(line)-1]=='5'):
            formatted_features.append(0)
            label_0.append(formatted_features)
        else:
            formatted_features.append(1)
            label_1.append(formatted_features)

label_0 = random.sample(label_0,2300)
label_0 = np.array(label_0)
dataset = np.concatenate((label_0,label_1))
dataset = np.asarray(dataset)
labels = dataset[:,dataset.shape[1]-1]
dataset = np.delete(dataset,dataset.shape[1]-1,1)
dataset = StandardScaler().fit_transform(dataset)
x_train, x_test, y_train, y_test = train_test_split(dataset,labels,test_size = 0.3, random_state=42)
np.save("x_train_balanced",x_train)
np.save("x_test_balanced",x_test)
np.save("y_train_balanced",y_train)
np.save("y_test_balanced",y_test)
print x_train.shape
