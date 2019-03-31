#linear Discriminant analysis
import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt
import math
from scipy.stats import norm
import scipy.stats as stats


def fisher(filename):

	plt.title(filename)
	plt.xlabel('Dimension_1')
	plt.ylabel('Dimension_2')

	dataset=pd.read_csv(filename,header=None)
	dataset.columns=["seq","x","y","class"]
	coordinates=dataset.iloc[:,1:3].values						#taking dataset into arrays
	member_class=dataset.iloc[:,3].values
	
	#finding mean vectors
	mean_vectors = []
	for cl in range(0,2):
		mean_vectors.append(np.mean(coordinates[member_class==cl], axis=0))
		#print(mean_vectors[cl])
	#finding within class covariance matrix
	S_W = np.zeros((2,2))
	for cl,mv in zip(range(0,2), mean_vectors):
		class_sc_mat = np.zeros((2,2))
		for row in coordinates[member_class==cl]:
			row, mv = row.reshape(2,1), mv.reshape(2,1)
			class_sc_mat += (row-mv).dot((row-mv).T)
		S_W += class_sc_mat
	#print('within class scatter matrix:\n', S_W)
	#Between class covariance matrix
	#overall_mean = np.mean(coordinates, axis=0)
	#S_B = np.zeros((2,2))
	#S_B = (mean_vectors[1]-mean_vectors[0]).dot((mean_vectors[1]-mean_vectors[0]).T)
	S_W_inv = np.linalg.inv(S_W)
	W = S_W_inv.dot(mean_vectors[1]-mean_vectors[0])
	W = W/np.linalg.norm(W)
	print("W is:",W)
	#print(W.shape)
	#print(coordinates.shape)

	#Plotting the points
	coordinates_class1=dataset[dataset['class']==0][["x","y"]].values
	coordinates_class2=dataset[dataset['class']==1][["x","y"]].values
	#print(coordinates_class1.shape)
	plt.plot(coordinates_class1[:,0],coordinates_class1[:,1],'.',color='r',alpha=0.25)
	plt.plot(coordinates_class2[:,0],coordinates_class2[:,1],'*',color='g',alpha=0.25)
	projection_class1=np.dot(coordinates_class1,W)
	projection_class2=np.dot(coordinates_class2,W)

	#Getting the class points in 2d
	proj_vec_class1=np.stack([projection_class1,projection_class1],axis=1)*W  #projected point with magnitude in direction of W
	proj_vec_class2=np.stack([projection_class2,projection_class2],axis=1)*W

	#print(proj_vec_class1.shape)
	
	plt.plot(proj_vec_class1[::1,0],proj_vec_class1[::1,1],'.',color='r',alpha=0.5,label='class 0')	#::5 means every 5th
	plt.plot(proj_vec_class2[::1,0],proj_vec_class2[::1,1],'*',color='g',alpha=0.5,label='class 1')
	plt.legend(loc='best')


	#Fitting projected points in normal distribution
	mean_projected_class1=np.mean(projection_class1)
	mean_projected_class2=np.mean(projection_class2)
	var_projected_class1=np.var(projection_class1)
	var_projected_class2=np.var(projection_class2)
	print("mean 1 is: ",mean_projected_class1)
	print("variance 1 is:",var_projected_class1)
	print("mean 2 is:",mean_projected_class2)
	print("variance 2 is:",var_projected_class2)
	#Finding point of intersection of normal distribution
	def solve(m1,m2,std1,std2):
	  a = 1/(2*std1**2) - 1/(2*std2**2)
	  b = m2/(std2**2) - m1/(std1**2)
	  c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
	  return np.roots([a,b,c])

	result = solve(mean_projected_class1,mean_projected_class2,math.sqrt(var_projected_class1),math.sqrt(var_projected_class2))
	W_perpendicular = np.array([-W[1],W[0]])
	#printing normal distribution of points

	z=np.linspace(-3,2,500)
	points=np.column_stack([z,z])
	
	projection_points=points*W
	normal_point_class1=norm.pdf(z,mean_projected_class1,math.sqrt(var_projected_class1))
	normal_point_class2=norm.pdf(z,mean_projected_class2,math.sqrt(var_projected_class2))
	normal_projection_vec_class1=np.column_stack([normal_point_class1,normal_point_class1])*W_perpendicular
	normal_projection_vec_class2=np.column_stack([normal_point_class2,normal_point_class2])*W_perpendicular
	normal_class1=projection_points+normal_projection_vec_class1
	normal_class2=projection_points+normal_projection_vec_class2
	plt.plot(normal_class1[::1,0],normal_class1[::1,1],'-',color='b',alpha=0.5)
	plt.plot(normal_class2[::1,0],normal_class2[::1,1],'-',color='b',alpha=0.5)

	#print(result)
	result_0=result[0]
	#Taking that root as intersection point which lies between means of both classes
	if((result[0]>mean_projected_class1 and result[0]<mean_projected_class2) or (result[0]>mean_projected_class2 and result[0]<mean_projected_class1)):
		seperation_point=np.asarray([result[0],result[0]])
	elif((result[1]>mean_projected_class1 and result[1]<mean_projected_class2) or (result[1]>mean_projected_class2 and result[1]<mean_projected_class1)):
		seperation_point=np.asarray([result[1],result[1]])

	point=(seperation_point.T)*W
	print("Seperation point is :",point)
	#Finding equation of seperating line
	x = np.linspace(-0.1,0.1)
	y=(-W[0]/W[1])*x-point[0]*(-W[0]/W[1])+point[1]
	plt.plot(x,y,'b')						#plotting seperating line
	#plt.savefig("fisher@"+filename+".png")	#Saving results
	plt.close()
	#plt.show()
fisher('dataset_1.csv')
fisher('dataset_2.csv')
fisher('dataset_3.csv')
