# from rarfile import RarFile 
import gzip
from os import listdir
import os
import tarfile
import numpy as np
def main():
	pathtofolder = '/Users/aayushimalik/Downloads/eeg_full'
	pathto = '/Users/aayushimalik/Downloads/alldata'
	for f in listdir(pathtofolder):
		if (f.endswith("tar.gz")):
			tar = tarfile.open(str(pathtofolder+'/'+f),'r:gz')
			tar.extractall(pathto)
			tar.close()
	labels = []
	X = []
	for f in listdir(pathto):
		pathtof=str(pathto+'/'+f)
		if(os.path.isdir(pathtof)):
			for subfile in listdir(pathtof):
				if (subfile.endswith(".gz")):
					f1 = gzip.open(str(pathtof+'/'+subfile), 'rb')
					label = f1.readline()
					if(label[5]=='a'):
						labels.append(1)
					else:
						labels.append(0)
					data = []
					for line in f1.readlines():
						if line[0]!='#':
							arr = line.split(' ')
							data.append(float(arr[-1].strip()))
					X.append(np.array(data))
					f1.close()
	
	labels = np.array(labels)
	X = np.array(X)
	np.save('data_alcoholic',X)
	np.save('labels_alcoholic',labels)
	X = np.load('data_alcoholic.npy')
	labels = np.load('labels_alcoholic.npy')
	data = np.zeros((X.shape[0],X[0].shape[0]))
	for i in range(X.shape[0]):
		if X[i].shape[0]==16384:
			for j in range(X[0].shape[0]):
				data[i][j] = X[i][j]
	np.save('data',data)
	data = np.load('data.npy')
	labels = np.load('labels_alcoholic.npy')
	print data[0]



if __name__=="__main__":
	main()

