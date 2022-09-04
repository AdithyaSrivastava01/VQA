import os
import torch 
import torchvision 
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from scipy.io import loadmat
from model import Model
from train import training
from prediction import prediction
from datautils import train_test_data
from data_dmos import testdata_dmos,traindata_dmos
from copy import deepcopy
import sklearn
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy import stats
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
cwd = os.getcwd()#gets the current working directory of the file


path=cwd+'/Resnet50Torch/'
categories=[x for x in os.listdir(path)]
categories.remove("live_vqc")
categories=['ecvq','live_vqa','epfl_cif', 'evvq', 'epfl_4cif', 'live_mobile', 'csiq']
# categories=["evvq","live_mobile"]
print(categories)
l_ecvq=[]
l_evvq=[]
l_epfl=[]
l_vqa=[]
l_mobile=[]
l_csiq=[]
l_cif=[]
l_cif4=[]

for i in range(3):
	for x in (categories):
		# print("categoies are: ",categories)
		print("x is: ",x)
		d_train={}
		d_test={}
		# copy=categories
		copy=deepcopy(categories)

		train_categories=[]
		test_categories=[]

		
		if(x=="evvq"):
			print("Testing for evvq")
			test_categories.append(x)
			copy.remove(x)
			train_categories=copy
			# # print(train_categories)
			d_train,d_test=train_test_data(d_train,d_test,train_categories,test_categories)

			d_test=testdata_dmos(test_categories,d_test)
			d_train=traindata_dmos(train_categories,d_train)
			# print(d_train['bf_t134'])
			# for i in d_train:
			# 	print(i,":",d_train[i][1])
			train_data=list(d_train.values())
			test_data=list(d_test.values())
			trainX =[train_data[i][0].numpy() for i in range(len(train_data))]
			trainY =np.array([train_data[i][1] for i in range(len(train_data))],dtype=float)



			testX =[test_data[i][0].numpy() for i in range(len(test_data))]

			testY =np.array([test_data[i][1] for i in range(len(test_data))],dtype=float)


			train_dl=DataLoader(train_data,batch_size=128,shuffle=True)

			test_dl=DataLoader(test_data,batch_size=32)
			model=Model()
			opt = torch.optim.Adam(model.parameters(),lr=1e-5)
			loss_fn = nn.L1Loss()
			# training(220,train_dl,model,loss_fn,opt)
			valid,loss=training(35,train_dl,model,loss_fn,opt,test_data,testY)
			valid_cor=list(valid.values())
			valid_it=list(valid.keys())
			for x in range(len(valid_it)):
				writer.add_scalar("evvq",valid_cor[x],valid_it[x])

			# print(valid_cor,valid_it)
			writer.flush()
			# model = SVR(kernel = 'rbf', C=1, epsilon=0.02, verbose=True)
			# model.fit(trainX, trainY)
			# preds = model.predict(testX)
			# # print(type(testY[0]))
			# print("SVR evvq: ",stats.spearmanr(preds, testY))
			l_evvq.append(prediction(model,test_data,testY))
			
		elif(x=="csiq"):
			print("Testing for "+x)
			test_categories=[x]
			copy.remove(x)
			train_categories=copy
			# print(train_categories)
			d_train,d_test=train_test_data(d_train,d_test,train_categories,test_categories)

			d_test=testdata_dmos(test_categories,d_test)
			d_train=traindata_dmos(train_categories,d_train)
			train_data=list(d_train.values())
			test_data=list(d_test.values())
			trainX =[train_data[i][0].numpy() for i in range(len(train_data))]
			trainY =np.array([train_data[i][1] for i in range(len(train_data))],dtype=float)



			testX =[test_data[i][0].numpy() for i in range(len(test_data))]

			testY =np.array([test_data[i][1] for i in range(len(test_data))],dtype=float)


			# testY =np.array([test_data[i][1] for i in range(len(test_data))])


			train_dl=DataLoader(train_data,batch_size=32,shuffle=True)

			test_dl=DataLoader(test_data,batch_size=32)
			# model = SVR(kernel = 'rbf', C=1, epsilon=0.02, verbose=True)
			# model.fit(trainX, trainY)
			# preds = model.predict(testX)
			# print(type(testY[0]))
			# print("SVR csiq: ",stats.spearmanr(preds, testY))
			model=Model()
			opt = torch.optim.Adam(model.parameters(),lr=1e-3)
			loss_fn = nn.L1Loss()
			valid,loss=training(64,train_dl,model,loss_fn,opt,test_data,testY)
			valid_cor=list(valid.values())
			valid_it=list(valid.keys())
			for x in range(len(valid_it)):
				writer.add_scalar("csiq",valid_cor[x],valid_it[x])

			# print(valid_cor,valid_it)
			writer.flush()
			l_csiq.append(prediction(model,test_data,testY))
			
			# break
		elif(x=="live_mobile"):
			print("Testing for "+x)
			test_categories=[x]
			copy.remove(x)
			train_categories=copy
			# print(train_categories)
			d_train,d_test=train_test_data(d_train,d_test,train_categories,test_categories)

			d_test=testdata_dmos(test_categories,d_test)
			d_train=traindata_dmos(train_categories,d_train)
			train_data=list(d_train.values())
			test_data=list(d_test.values())
			trainX =[train_data[i][0].numpy() for i in range(len(train_data))]
			trainY =np.array([train_data[i][1] for i in range(len(train_data))],dtype=float)



			testX =[test_data[i][0].numpy() for i in range(len(test_data))]

			testY =np.array([test_data[i][1] for i in range(len(test_data))],dtype=float)


			# testY =np.array([test_data[i][1] for i in range(len(test_data))])


			train_dl=DataLoader(train_data,batch_size=32,shuffle=True)

			test_dl=DataLoader(test_data,batch_size=32)
			model=Model()
			opt = torch.optim.Adam(model.parameters(),lr=1e-5)
			loss_fn = nn.L1Loss()
			valid,loss=training(200,train_dl,model,loss_fn,opt,test_data,testY)
			# model = SVR(kernel = 'rbf', C=1, epsilon=0.02, verbose=True)
			# model.fit(trainX, trainY)
			# preds = model.predict(testX)
			# print(type(testY[0]))
			# print("SVR live_mobile: ",stats.spearmanr(preds, testY))
			valid_cor=list(valid.values())
			valid_it=list(valid.keys())
			loss_cor=list(loss.values())
			loss_it=list(loss.keys())
			for x in range(len(valid_it)):
				writer.add_scalar("live_mobile",valid_cor[x],valid_it[x])

			# print(valid_cor,valid_it)
			writer.flush()
			for x in range(len(loss_it)):
				writer.add_scalar("live_mobile_loss",loss_cor[x],loss_it[x])

			# print(valid_cor,valid_it)
			writer.flush()
			l_mobile.append(prediction(model,test_data,testY))
			
		elif(x=='ecvq'):
			print("Testing for "+x)
			test_categories=[x]
			copy.remove(x)
			train_categories=copy
			# print(train_categories)
			d_train,d_test=train_test_data(d_train,d_test,train_categories,test_categories)

			d_test=testdata_dmos(test_categories,d_test)
			d_train=traindata_dmos(train_categories,d_train)
			train_data=list(d_train.values())
			test_data=list(d_test.values())


			# testY =np.array([test_data[i][1] for i in range(len(test_data))])
			trainX =[train_data[i][0].numpy() for i in range(len(train_data))]
			trainY =np.array([train_data[i][1] for i in range(len(train_data))],dtype=float)



			testX =[test_data[i][0].numpy() for i in range(len(test_data))]

			testY =np.array([test_data[i][1] for i in range(len(test_data))],dtype=float)


			train_dl=DataLoader(train_data,batch_size=64,shuffle=True)

			test_dl=DataLoader(test_data,batch_size=32)
			# model = SVR(kernel = 'rbf', C=1, epsilon=0.02, verbose=True)
			# model.fit(trainX, trainY)
			# preds = model.predict(testX)
			# print(type(testY[0]))
			# print("SVR ecvq: ",stats.spearmanr(preds, testY))

			model=Model()
			opt = torch.optim.Adam(model.parameters(),lr=1e-3)
			loss_fn = nn.L1Loss()
			# training(220,train_dl,model,loss_fn,opt,test_data,testY)
			valid,loss=training(350,train_dl,model,loss_fn,opt,test_data,testY)
			valid_cor=list(valid.values())
			valid_it=list(valid.keys())
			for x in range(len(valid_it)):
				writer.add_scalar("ecvq",valid_cor[x],valid_it[x])

			# print(valid_cor,valid_it)
			writer.flush()
			l_ecvq.append(prediction(model,test_data,testY))
			# # break
		elif(x=="live_vqa"):
			print("Testing for "+x)
			test_categories=[x]
			copy.remove(x)
			train_categories=copy
			# print(train_categories)
			d_train,d_test=train_test_data(d_train,d_test,train_categories,test_categories)

			d_test=testdata_dmos(test_categories,d_test)
			d_train=traindata_dmos(train_categories,d_train)
			
			
			train_data=list(d_train.values())
			test_data=list(d_test.values())


			# testY =np.array([test_data[i][1] for i in range(len(test_data))])
			trainX =[train_data[i][0].numpy() for i in range(len(train_data))]
			trainY =np.array([train_data[i][1] for i in range(len(train_data))],dtype=float)



			testX =[test_data[i][0].numpy() for i in range(len(test_data))]

			testY =np.array([test_data[i][1] for i in range(len(test_data))],dtype=float)


			train_dl=DataLoader(train_data,batch_size=32,shuffle=True)

			test_dl=DataLoader(test_data,batch_size=32)
			# model = SVR(kernel = 'rbf', C=1, epsilon=0.02, verbose=True)
			# model.fit(trainX, trainY)
			# preds = model.predict(testX)
			# print(type(testY[0]))
			# print("SVR live_vqa: ",stats.spearmanr(preds, testY))



			train_dl=DataLoader(train_data,batch_size=32,shuffle=True)

			test_dl=DataLoader(test_data,batch_size=32)
			# # model1=Model()
			# # model1.load_state_dict(torch.load('VQAweights.pth'))
			# #TRAINING
			model=Model()
			opt = torch.optim.Adam(model.parameters(),lr=1e-3)
			loss_fn = nn.L1Loss()
			# training(120,train_dl,model,loss_fn,opt,test_data,testY)
			valid,loss=training(370,train_dl,model,loss_fn,opt,test_data,testY)
			valid_cor=list(valid.values())
			valid_it=list(valid.keys())
			for x in range(len(valid_it)):
				writer.add_scalar("live_vqa",valid_cor[x],valid_it[x])

			# print(valid_cor,valid_it)
			writer.flush()
			l_vqa.append(prediction(model,test_data,testY))
			# # break
		elif(x=="epfl_cif"):
			print("Testing for "+x)
			test_categories=[x]
			copy=[ele for ele in copy if ele not in test_categories]

			# categories.remove(x)
			train_categories=copy
			# print(train_categories)
			d_train,d_test=train_test_data(d_train,d_test,train_categories,test_categories)

			d_test=testdata_dmos(test_categories,d_test)
			d_train=traindata_dmos(train_categories,d_train)
			
			train_data=list(d_train.values())
			
			test_data=list(d_test.values())
			# print("testlist: ",test_data)


			trainX =[train_data[i][0].numpy() for i in range(len(train_data))]
			trainY =np.array([train_data[i][1] for i in range(len(train_data))],dtype=float)



			testX =[test_data[i][0].numpy() for i in range(len(test_data))]

			testY =np.array([test_data[i][1] for i in range(len(test_data))],dtype=float)


			train_dl=DataLoader(train_data,batch_size=64,shuffle=True)

			test_dl=DataLoader(test_data,batch_size=32)
			# model = SVR(kernel = 'rbf', C=1, epsilon=0.02, verbose=True)
			# model.fit(trainX, trainY)
			# preds = model.predict(testX)
			# # print(type(testY[0]))
			# print("SVR epfl: ",stats.spearmanr(preds, testY))



			
			model=Model()
			opt = torch.optim.Adam(model.parameters(),lr=1e-4)
			loss_fn = nn.L1Loss()
			valid,loss=training(320,train_dl,model,loss_fn,opt,test_data,testY)
			valid_cor=list(valid.values())
			valid_it=list(valid.keys())
			for x in range(len(valid_it)):
				writer.add_scalar("epfl_cif",valid_cor[x],valid_it[x])

			# print(valid_cor,valid_it)
			writer.flush()


			l_cif.append(prediction(model,test_data,testY))
		elif(x=="epfl_4cif"):
			print("Testing for "+x)
			test_categories=[x]
			copy=[ele for ele in copy if ele not in test_categories]

			# categories.remove(x)
			train_categories=copy
			# print(train_categories)
			d_train,d_test=train_test_data(d_train,d_test,train_categories,test_categories)

			d_test=testdata_dmos(test_categories,d_test)
			d_train=traindata_dmos(train_categories,d_train)
			
			train_data=list(d_train.values())
			
			test_data=list(d_test.values())
			# print("testlist: ",test_data)


			trainX =[train_data[i][0].numpy() for i in range(len(train_data))]
			trainY =np.array([train_data[i][1] for i in range(len(train_data))],dtype=float)



			testX =[test_data[i][0].numpy() for i in range(len(test_data))]

			testY =np.array([test_data[i][1] for i in range(len(test_data))],dtype=float)


			train_dl=DataLoader(train_data,batch_size=64,shuffle=True)

			test_dl=DataLoader(test_data,batch_size=32)
			# model = SVR(kernel = 'rbf', C=1, epsilon=0.02, verbose=True)
			# model.fit(trainX, trainY)
			# preds = model.predict(testX)
			# # print(type(testY[0]))
			# print("SVR epfl: ",stats.spearmanr(preds, testY))



			
			model=Model()
			opt = torch.optim.Adam(model.parameters(),lr=1e-4)
			loss_fn = nn.L1Loss()
			valid,loss=training(320,train_dl,model,loss_fn,opt,test_data,testY)
			valid_cor=list(valid.values())
			valid_it=list(valid.keys())
			for x in range(len(valid_it)):
				writer.add_scalar("epfl_4cif",valid_cor[x],valid_it[x])

			# print(valid_cor,valid_it)
			writer.flush()


			l_cif4.append(prediction(model,test_data,testY))
			# # break
		





x=(sum(l_cif)/len(l_cif)+sum(l_cif4)/len(l_cif4))/2


print("ECVQ: ",sum(l_ecvq)/len(l_ecvq))
print("EVVQ: ",sum(l_evvq)/len(l_evvq))
print("LIVE MOBILE: ",sum(l_mobile)/len(l_mobile))
print("LIVE VQA: ",sum(l_vqa)/len(l_vqa))
print("CSIQ: ",sum(l_csiq)/len(l_csiq))
print("EPFL: ",x)











		

# print(train_categories)

# scores_path=cwd+"/Scores/"
# scores=[x for x in os.listdir(scores_path)]
# # print(scores)

# score_path=cwd+'/Scores/'

# print(train_categories)


	

# MOS dictionary

# dmos_mobile={}

# mobile_names=[]
# mobile_dmos=[]
# vqa_dmos=[]
# vqa_names=[]
# cif_names=[]
# cif_dmos=[]
# cif4_names=[]
# cif4_dmos=[]
# csiq_names=[]
# csiq_dmos=[]
# evvq_dmos=[]
# evvq_names=[]
# ecvq_dmos=[]
# ecvq_names=[]


# score_path=cwd+'/Scores/'
# for x in os.listdir(score_path):
	
# 	if(x=='dmos_final.mat'):
# 		l=[]
# 		mat1=loadmat(score_path+'dmos_final.mat')
# 		x=min(mat1['dmos_mobile'])
# 		y=max(mat1['dmos_mobile'])
# 		for en,i in enumerate(mat1['dmos_mobile']):
# 			l.append(((i)/(y))[0])
# 		mobile_dmos=l
# 	elif(x=='live_mobile_names.mat'):
		
# 		mat=loadmat(score_path+x)
# 		for t in mat['dist_names'][0]:
# 			mobile_names.append(t[0])
		
# 		with open(score_path+"remove.txt",'r') as f:
# 			x=f.readlines()
		
# 		for en,i in enumerate(x):
# 			mobile_names.remove(i.rstrip(".yuv\n"))
			
# 	elif(x=="live_vqa_dmos.txt"):
	
# 		with open(score_path+x,'r') as f:
# 			x=f.readlines()
# 			for i in x:
# 				i=i.split('\t')
				
# 				vqa_dmos.append(float(i[0]))
# 			y=max(vqa_dmos)
			
# 			for j in range(len(vqa_dmos)):
				
# 				d_train[vqa_names[j]][1] = (vqa_dmos[j])/(y) 
	
# 	elif(x=="live_vqa_names.txt"):
# 		with open(score_path+x,'r') as f:
# 			x=f.readlines()
# 			for en,(i) in enumerate(x):
# 				vqa_names.append(i.rstrip(".yuv\n"))
	
# 	elif(x=='csiq_dmos.txt'):
	
# 		with open(score_path+x,'r') as f:
# 			x=f.readlines()
# 			for i in x:
# 				i=i.split("\t")
				

# 				csiq_names.append(i[0].rstrip('.yuv'))
# 				csiq_dmos.append(float(i[1]))
				
			
# 			y=max(csiq_dmos)
			
# 			for j in range(len(csiq_dmos)):
				
# 				d_train[csiq_names[j]][1]=(1-csiq_dmos[j]/y)
			
			
# 	elif(x=='evvq_dmos.txt'):
	
		
# 		with open(score_path+x,'r') as f:
# 			x=f.readlines()
# 			for i in x:
			
# 				i=i.split("\t")
# 				i=list(filter(lambda a: a != "", i))
# 				i=list(filter(lambda a: a != '       ', i))
				

			
# 				evvq_names.append(i[0])
# 				evvq_dmos.append(float(i[2]))
				
			
# 			y=max(evvq_dmos)
			
# 			for j in range(len(evvq_dmos)):
				
# 				d_train[evvq_names[j]][1]=(1-evvq_dmos[j]/y)
			
		

# 	elif(x=='ecvq_dmos.txt'):
		
# 		with open(score_path+x,'r') as f:
# 			x=f.readlines()
# 			for i in x:
# 				i=i.split("\t")
# 				i=list(filter(lambda a: a != "", i))
# 				i=list(filter(lambda a: a != '       ', i))
				
				
# 				ecvq_names.append(i[0])
# 				ecvq_dmos.append(float(i[2]))
			
# 			y=max(ecvq_dmos)
			
# 			for j in range(len(ecvq_dmos)):
# 				d_test[ecvq_names[j]][1]=(1-ecvq_dmos[j]/y)
			
# 	elif(x=='epfl_4cif_mos.txt'):
		
# 		with open(score_path+x,'r') as f:
# 			x=f.readlines()
# 			for i in x:
# 				i=i.split("\t")
# 				i=list(filter(lambda a: a != "", i))
				

# 				i_new=i[1].split(" ")
# 				x=5-float(i_new[0])
				
# 				cif4_names.append(i[0])
# 				cif4_dmos.append(x)
			
# 			y=max(cif4_dmos)
			
# 			for j in range(len(cif4_dmos)):
				
# 				d_train[cif4_names[j]][1]=((1-cif4_dmos[j])/y)
			

# 	elif(x=='epfl_cif_mos.txt'):
	
# 		with open(score_path+x,'r') as f:
# 			x=f.readlines()
# 			for i in x:
# 				i=i.split("\t")
# 				i=list(filter(lambda a: a != "", i))
				

# 				i_new=i[1].split(" ")
# 				x=5-float(i_new[0])
				
# 				cif_names.append(i[0])
# 				cif_dmos.append(x)
			
# 			y=max(cif_dmos)
			
# 			for j in range(len(cif_dmos)):
				
# 				d_train[cif_names[j]][1]=((1-cif_dmos[j])/y)
		







# 	dmos_mobile=dict(zip(mobile_names,mobile_dmos))
	
# for i in dmos_mobile:
# 	d_train[i][1]=dmos_mobile[i]

# train_data=list(d_train.values())
# test_data=list(d_test.values())


# testY =np.array([test_data[i][1] for i in range(len(test_data))])


# train_dl=DataLoader(train_data,batch_size=32,shuffle=True)

# test_dl=DataLoader(test_data,batch_size=32)








#TRAINING
# model=Model()
# opt = torch.optim.Adam(model.parameters(),lr=1e-3)
# loss_fn = nn.L1Loss()
# training(220,train_dl,model,loss_fn,opt)

#PREDICTION
# model1=Model()
# model1.load_state_dict(torch.load('VQAweights.pth'))
# print(prediction(model1,test_data,testY))