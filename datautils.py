import os
import torch 
import numpy as np
cwd = os.getcwd()
path=cwd+'/Resnet50Torch/'
def train_test_data(d_train,d_test,train_categories,test_categories):
	for i in train_categories:
		
		for x in os.listdir(path+i+'/'):
		

			a=np.load(path+i+'/'+x)
			k=np.mean(a,axis=0)
			d_train[x.rstrip('.npy')]=[torch.from_numpy(k),0]
	

	for i in test_categories:
		
		for x in os.listdir(path+i+'/'):
			
			a=np.load(path+i+'/'+x)
			k=np.mean(a,axis=0)
			d_test[x.rstrip('.npy')]=[torch.from_numpy(k),0]
	return (d_train,d_test)

if(__name__=="__main__"):
    print("in utility file")