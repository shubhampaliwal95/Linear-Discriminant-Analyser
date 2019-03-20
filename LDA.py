#linear Discriminant analysis
import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt
import math
#taking datasets into arrays
dataset=pd.read_csv('dataset_1.csv')
coordinates=dataset.iloc[:,1:3].values
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
S_B = np.zeros((2,2))
S_B = (mean_vectors[1]-mean_vectors[0]).dot((mean_vectors[1]-mean_vectors[0]).T)
S_W_inv = np.linalg.inv(S_W)
W = S_W_inv.dot(mean_vectors[1]-mean_vectors[0])
W = W/np.linalg.norm(W)
print(W)
print(coordinates.shape)
plt.plot(coordinates,'*')

#projected_points_1 = []
#for i in range(0,1):
#	projected_points_1.append(W.dot((coordinates[member_class==i]).T))
#plt.plot(projected_points_1,'^')
plt.show() 
'''for i,mean_vec in enumerate(mean_vectors):
	n = coordinates[member_class==i,:].shape[0]
	mean_vec = mean_vec.reshape(2,1)
	overall_mean = overall_mean.reshape(2,1)
	S_B+= n * (mean_vec - overall_mean).dot((mean_vec-overall_mean).T)
print('between-class scatter matrix', S_B)
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:,i].reshape(2,1)   
    print('\nEigenvector {}: \n{}'.format(i, eigvec_sc.real))
    print('Eigenvalue {:}: {:.2e}'.format(i, eig_vals[i].real))
for i in range(len(eig_vals)):
    eigv = eig_vecs[:,i].reshape(2,1)
    np.testing.assert_array_almost_equal(np.linalg.inv(S_W).dot(S_B).dot(eigv),
                                         eig_vals[i] * eigv,
                                         decimal=6, err_msg='', verbose=True)
print('ok')

#eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
#eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
#print('Eigenvalues in decreasing order:\n')
#for i in eig_pairs:
#   print(i[0])
#print('Variance explained:\n')
#eigv_sum = sum(eig_vals)
#for i,j in enumerate(eig_pairs):
#    print('eigenvalue {0:}: {1:.2%}'.format(i, (j[0]/eigv_sum).real))
#W = np.hstack((eig_pairs[0][1].reshape(2,1)))
print(W.shape)
print(coordinates.shape)
X_lda = coordinates.dot(W)
print(X_lda.shape)
#plt.plot(W,'r')
l1, = plt.plot(W,'r')
l2, = plt.plot(X_lda,'*')
#plt.show()
plt.show()

print('Done')
'''