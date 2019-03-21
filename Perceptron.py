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
	#plt.show()
	#Applying perceptron
	W = [1,2,3]
	ones = np.ones(coordinates_class1.shape[0],int,1)
	ones = ones.reshape(coordinates_class1.shape[0],1)
	print(coordinates_class1.shape)
	coordinates_class1_1=np.column_stack([ones,coordinates_class1])
	coordinates_class2_1=np.column_stack([ones,coordinates_class2])
	print(coordinates_class1_1.shape)
	
	
perceptron('dataset_1.csv')
perceptron('dataset_2.csv')
perceptron('dataset_3.csv')