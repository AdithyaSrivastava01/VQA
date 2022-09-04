import os
from scipy.io import loadmat
cwd = os.getcwd()



score_path=cwd+'/Scores/'

def testdata_dmos(test_categories,d_test):
	dmos_mobile={}

	mobile_names=[]
	mobile_dmos=[]
	vqa_dmos=[]
	vqa_names=[]
	cif_names=[]
	cif_dmos=[]
	cif4_names=[]
	cif4_dmos=[]
	csiq_names=[]
	csiq_dmos=[]
	evvq_dmos=[]
	evvq_names=[]
	ecvq_dmos=[]
	ecvq_names=[]
	for category in test_categories:
		if(category=='live_mobile'):
			l=[]
			mat1=loadmat(score_path+'dmos_final.mat')
			x=min(mat1['dmos_mobile'])
			y=max(mat1['dmos_mobile'])
			for en,i in enumerate(mat1['dmos_mobile']):
				l.append((i/y)[0])

			mobile_dmos=l

			
			mat=loadmat(score_path+'live_mobile_names.mat')
			for t in mat['dist_names'][0]:
				mobile_names.append(t[0])
			
			with open(score_path+"remove.txt",'r') as f:
				x=f.readlines()
			
			for en,i in enumerate(x):
				mobile_names.remove(i.rstrip(".yuv\n"))
			dmos_mobile=dict(zip(mobile_names,mobile_dmos))

			for i in dmos_mobile:
				d_test[i][1]=dmos_mobile[i]
				
		elif(category=="live_vqa"):

			with open(score_path+"live_vqa_dmos.txt",'r') as f:
				x=f.readlines()
				for i in x:
					i=i.split('\t')
					
					vqa_dmos.append(float(i[0]))
				y=max(vqa_dmos)

				with open(score_path+"live_vqa_names.txt",'r') as f:
					x=f.readlines()
				for en,(i) in enumerate(x):
					vqa_names.append(i.rstrip(".yuv\n"))
				
				for j in range(len(vqa_dmos)):
					
					d_test[vqa_names[j]][1] =vqa_dmos[j]/y 


			

		elif(category=='csiq'):

			with open(score_path+'csiq_dmos.txt','r') as f:
				x=f.readlines()
				for i in x:
					i=i.split("\t")
					

					csiq_names.append(i[0].rstrip('.yuv'))
					csiq_dmos.append(float(i[1]))
					
				
				y=max(csiq_dmos)
				
				for j in range(len(csiq_dmos)):
					
					d_test[csiq_names[j]][1]=csiq_dmos[j]/y
				
				
		elif(category=='evvq'):

			
			with open(score_path+'evvq_dmos.txt','r') as f:
				x=f.readlines()
				for i in x:
				
					i=i.split("\t")
					i=list(filter(lambda a: a != "", i))
					i=list(filter(lambda a: a != '       ', i))
					

				
					evvq_names.append(i[0])
					evvq_dmos.append(float(i[2]))
					
				
				y=max(evvq_dmos)
				
				for j in range(len(evvq_dmos)):
					
					d_test[evvq_names[j]][1]=1.0-evvq_dmos[j]/y
					# d_test[evvq_names[j]][1]=evvq_dmos[j]
				
			

		elif(category=='ecvq'):
			
			with open(score_path+'ecvq_dmos.txt','r') as f:
				x=f.readlines()
				for i in x:
					i=i.split("\t")
					i=list(filter(lambda a: a != "", i))
					i=list(filter(lambda a: a != '       ', i))
					
					
					ecvq_names.append(i[0])
					ecvq_dmos.append(float(i[2]))
				
				y=max(ecvq_dmos)
				
				for j in range(len(ecvq_dmos)):
					d_test[ecvq_names[j]][1]=1.0-ecvq_dmos[j]/y
				return d_test
				
		elif(category=='epfl_4cif'):
			
			with open(score_path+'epfl_4cif_mos.txt','r') as f:
				x=f.readlines()
				# print("x ix : ",x)
				for i in x:
					i=i.split("\t")
					# print("i is: ",i)
					# i=list(filter(lambda a: a != "", i))
					

					# i_new=i[1].split(" ")
					# x=5-float(i_new[0])
					remove=['\n']
					for sub in remove:
					    i[1] = i[1].replace(sub, ' ')
					res = " ".join(i[1].split())
					cif4_names.append(i[0])
					cif4_dmos.append(float(res))
				
				y=max(cif4_dmos)
				
				for j in range(len(cif4_dmos)):
				# 	print(type(cif4_dmos[j]))
					d_test[cif4_names[j]][1] =1.0-cif4_dmos[j]/y
				

		elif(category=='epfl_cif'):

			with open(score_path+'epfl_cif_mos.txt','r') as f:
				x=f.readlines()
				# print("x ix : ",x)
				for i in x:
					i=i.split("\t")
					# i=i.split(" ")

					
					# i=list(filter(lambda a: a != "  ", i))
					i[1] = i[1].split('\n')[0]
					i[1] = i[1].split(' ')[0]
					# print("i is: ",i)
					

					# i_new=i[1].split(" ")
					# x=5-float(i_new[0])
					
					cif_names.append(i[0])
					cif_dmos.append(float(i[1]))
				
				y=max(cif_dmos)
				
				# print(cif_dmos)
				for j in range(len(cif_dmos)):
				# 	print(type(cif4_dmos[j]))
					d_test[cif_names[j]][1] =1.0-cif_dmos[j]/y
	return d_test
	


def traindata_dmos(train_categories,d_train):
	dmos_mobile={}

	mobile_names=[]
	mobile_dmos=[]
	vqa_dmos=[]
	vqa_names=[]
	cif_names=[]
	cif_dmos=[]
	cif4_names=[]
	cif4_dmos=[]
	csiq_names=[]
	csiq_dmos=[]
	evvq_dmos=[]
	evvq_names=[]
	ecvq_dmos=[]
	ecvq_names=[]
	for category in train_categories:
		
		if(category=='live_mobile'):
			l=[]
			mat1=loadmat(score_path+'dmos_final.mat')
			x=min(mat1['dmos_mobile'])
			y=max(mat1['dmos_mobile'])
			for en,i in enumerate(mat1['dmos_mobile']):
				l.append((i/y)[0])
			mobile_dmos=l

			
			mat=loadmat(score_path+'live_mobile_names.mat')
			for t in mat['dist_names'][0]:
				mobile_names.append(t[0])
			
			with open(score_path+"remove.txt",'r') as f:
				x=f.readlines()
			
			for en,i in enumerate(x):
				mobile_names.remove(i.rstrip(".yuv\n"))
			dmos_mobile=dict(zip(mobile_names,mobile_dmos))
			# print(dmos_mobile)
			# print(mobile_names[10])
			# print(mobile_dmos[10])

			for i in dmos_mobile:
				d_train[i][1]=dmos_mobile[i]
			
				
		elif(category=="live_vqa"):

			with open(score_path+"live_vqa_dmos.txt",'r') as f:
				x=f.readlines()
				for i in x:
					i=i.split('\t')
					
					vqa_dmos.append(float(i[0]))
				y=max(vqa_dmos)

				with open(score_path+"live_vqa_names.txt",'r') as f:
					x=f.readlines()
				for en,(i) in enumerate(x):
					vqa_names.append(i.rstrip(".yuv\n"))
				
				for j in range(len(vqa_dmos)):
					
					d_train[vqa_names[j]][1] =vqa_dmos[j]/y 
					# d_train[vqa_names[j]][1] =vqa_dmos[j]

				


			

		elif(category=='csiq'):

			with open(score_path+'csiq_dmos.txt','r') as f:
				x=f.readlines()
				for i in x:
					i=i.split("\t")
					

					csiq_names.append(i[0].rstrip('.yuv'))
					csiq_dmos.append(float(i[1]))
					
				
				y=max(csiq_dmos)
				
				for j in range(len(csiq_dmos)):
					# d_train[csiq_names[j]][1] =csiq_dmos[j]
					
					d_train[csiq_names[j]][1]=csiq_dmos[j]/y
				
				
		elif(category=='evvq'):

			
			with open(score_path+'evvq_dmos.txt','r') as f:
				x=f.readlines()
				for i in x:
				
					i=i.split("\t")
					i=list(filter(lambda a: a != "", i))
					i=list(filter(lambda a: a != '       ', i))
					

				
					evvq_names.append(i[0])
					evvq_dmos.append(float(i[2]))
					
				
				y=max(evvq_dmos)
				
				for j in range(len(evvq_dmos)):

					
					d_train[evvq_names[j]][1]=1.0-evvq_dmos[j]/y
				
			

		elif(category=='ecvq'):
			
			with open(score_path+'ecvq_dmos.txt','r') as f:
				x=f.readlines()
				for i in x:
					i=i.split("\t")
					i=list(filter(lambda a: a != "", i))
					i=list(filter(lambda a: a != '       ', i))
					
					
					ecvq_names.append(i[0])
					ecvq_dmos.append(float(i[2]))
				
				y=max(ecvq_dmos)
				
				for j in range(len(ecvq_dmos)):
					# d_train[ecvq_names[j]][1] =ecvq_dmos[j]
					d_train[ecvq_names[j]][1]=1.0-ecvq_dmos[j]/y
				
		elif(category=='epfl_4cif'):
			
			with open(score_path+'epfl_4cif_mos.txt','r') as f:
				x=f.readlines()
				# print("x ix : ",x)
				for i in x:
					i=i.split("\t")
					# print("i is: ",i)
					# i=list(filter(lambda a: a != "", i))
					

					# i_new=i[1].split(" ")
					# x=5-float(i_new[0])
					remove=['\n']
					for sub in remove:
					    i[1] = i[1].replace(sub, ' ')
					res = " ".join(i[1].split())
					cif4_names.append(i[0])
					cif4_dmos.append(float(res))
				
				y=max(cif4_dmos)
				
				for j in range(len(cif4_dmos)):
				# 	print(type(cif4_dmos[j]))
					d_train[cif4_names[j]][1] =1.0-cif4_dmos[j]/y
					
					# d_train[cif4_names[j]][1]=1-cif4_dmos[j]/y
	# print("check: ",cif4_dmos)

		elif(category=='epfl_cif'):

			with open(score_path+'epfl_cif_mos.txt','r') as f:
				x=f.readlines()
				# print("x ix : ",x)
				for i in x:
					i=i.split("\t")
					# i=i.split(" ")

					
					# i=list(filter(lambda a: a != "  ", i))
					i[1] = i[1].split('\n')[0]
					i[1] = i[1].split(' ')[0]
					# print("i is: ",i)
					

					# i_new=i[1].split(" ")
					# x=5-float(i_new[0])
					
					cif_names.append(i[0])
					cif_dmos.append(float(i[1]))
				
				y=max(cif_dmos)
				
				# print(cif_dmos)
				for j in range(len(cif_dmos)):
				# 	print(type(cif4_dmos[j]))
					d_train[cif_names[j]][1] =1.0-cif_dmos[j]/y
					
					# d_train[cif_names[j]][1]=1-cif_dmos[j]/y

	return d_train
	









if(__name__=="__main__"):
    print("in dmos file")