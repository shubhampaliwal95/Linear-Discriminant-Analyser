#Perceptron
import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt
import math
import scipy.stats as stats
def perceptron(filename):
	dataset=pd.read_csv(filename,header=None)
	dataset.columns=["seq","x","y","class"]
	coordinates=dataset.iloc[:,1:3].values
	member_class=dataset.iloc[:,3].values
	coordinates_class1=dataset[dataset['class']==0][["x","y"]].values
	coordinates_class2=dataset[dataset['class']==1][["x","y"]].values
	#plotting scatter plot
	plt.title(filename)
	plt.xlabel('Dimension_1')
	plt.ylabel('Dimension_2')
	plt.plot(coordinates_class1[:,0],coordinates_class1[:,1],'.',color='r',alpha=0.5,label='Class 0')
	plt.plot(coordinates_class2[:,0],coordinates_class2[:,1],'*',color='b',alpha=0.5,label='Class 1')
	color=	plt.legend(loc='best')
	plt.show()
	#Applying perceptron
	W = [1,1,1]
	for i in range(0,1000):
		
		ones = np.ones(coordinates_class1.shape[0],int,1)
		ones = ones.reshape(coordinates_class1.shape[0],1)
		#print(coordinates_class1.shape)
		coordinates_class1_1=np.column_stack([ones,coordinates_class1])	#1 W0 W1
		coordinates_class2_1=np.column_stack([ones,coordinates_class2])
		#print(coordinates_class1_1.shape)
		projection_class1=np.dot(coordinates_class1_1,W)
		projection_class2=np.dot(coordinates_class2_1,W)
		#print(projection_class1.shape)
		error_projection_1=0
		error_projection_2=0
		for i in range(0,projection_class1.shape[0]):
			if(projection_class1[i]<0):
				error_projection_1+=(projection_class1[i])
		for i in range(0,projection_class2.shape[0]):
			if(projection_class2[i]>=0):
				error_projection_2+=(projection_class2[i])
		net_error=abs(error_projection_1)+abs(error_projection_2)
		print(net_error)
		del_E_by_del_W0=0
		del_E_by_del_W1=0
		del_E_by_del_W2=0
		for i in range(0,projection_class1.shape[0]):
			if(projection_class1[i]<0):
				del_E_by_del_W0+=(coordinates_class1_1[i][0])
				del_E_by_del_W1+=(coordinates_class1_1[i][1])
				del_E_by_del_W2+=(coordinates_class1_1[i][2])
		for i in range(0,projection_class2.shape[0]):
			if(projection_class2[i]>=0):
				del_E_by_del_W0+=(coordinates_class2_1[i][0])*-1
				del_E_by_del_W1+=(coordinates_class2_1[i][1])*-1
				del_E_by_del_W2+=(coordinates_class2_1[i][2])*-1
		#print(del_E_by_del_W0)
		W[0]=W[0]+1*del_E_by_del_W0
		W[1]=W[1]+1*del_E_by_del_W1
		W[2]=W[2]+1*del_E_by_del_W2
		#print(W)

	
perceptron('dataset_3.csv')
#perceptron('dataset_2.csv')
#perceptron('dataset_3.csv')