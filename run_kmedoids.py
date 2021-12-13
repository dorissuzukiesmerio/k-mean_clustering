# K-Mean Clustering Algorithm : WRITING FUNCTION

import pandas
from sklearn.cluster import kmedoids # python -m pip install scikit-learn
from sklean_extra.cluster import KMedoids

import matplotlib.pyplot as pyplot
import numpy

from sklearn.metrics import silhouette_score

data = pandas.read_csv("dataset.csv")

print(data)
data = data.values # Transformation: two values in one array, array of arrays ; loads to sklearn better; though looses some functionalities such as matrix operation
print(data)


# Visualization: 
pyplot.scatter(data[:,0], data[:,1]) # row, column (all rows, first column)
pyplot.savefig("scatterplot.png") # no need for this if using Jupyter Notebook


def run_kmeans(n, data): 	
	machine = KMeans(n_clusters = n) 
	machine.fit(data)
	results = machine.predict(data)
	centroids = machine.cluster_centers_ # internal use
	ssd = machine.inertia_ #sum of square deviations
	# print(ssd)
	silhouette = 0
	if n>1:
		silhouette = silhouette_score(data, machine.labels_, metric = 'euclidean')
	pyplot.scatter(data[:,0], data[:,1], c = results) # row, column (all rows, first column)
	pyplot.scatter(centroids[:,0], centroids[:,1], c='red', marker= "*", s=200)
	pyplot.savefig("scatterplot_kmeans_colors_"+ str(n) + ".png") # no need for this if using Jupyter Notebook
	pyplot.close()
	return ssd, silhouette



def run_KMedoids(n, data): 	
	machine = KMedoids(n_clusters = n) 
	machine.fit(data)
	results = machine.predict(data)
	centroids = machine.cluster_centers_ # internal use
	ssd = machine.inertia_ #sum of square deviations
	# print(ssd)
	silhouette = 0
	if n>1:
		silhouette = silhouette_score(data, machine.labels_, metric = 'euclidean')
	pyplot.scatter(data[:,0], data[:,1], c = results) # row, column (all rows, first column)
	pyplot.scatter(centroids[:,0], centroids[:,1], c='red', marker= "*", s=200)
	pyplot.savefig("scatterplot_kmedoids_colors_"+ str(n) + ".png") # no need for this if using Jupyter Notebook
	pyplot.close()
	return ssd, silhouette


#KMeans results:
result = [run_kmeans(i+1, data) for i in range(7)][1:]
print(result)

ssd_result = [ i[0] for i in result] 
silhouette_result = [ i[1] for i in result][1:]

pyplot.plot(range(1,8), result)
pyplot.savefig("ssd.png")
pyplot.close() # Interpretaion: see almost linear flat curve after n=4


result_diff = [result[i-1] - x for i,x in enumerate(results)] [1:] # [1:] is the way to drop the first element
print(result_diff)

pyplot.plot(range(1,8), silhouette_result)
pyplot.savefig("silhouette.png")
pyplot.close()

print("\nssd: \n", ssd_result)
print("\nssd differences: \n", ssd_result_diff)


print("\nsilhouette scores: \n", silhouette_result)
print("\nmax silhouette scores: \n", max(silhouette_result))
print("\nnumber of cluster with max silhouette scores: \n", silhouette_result.index(max(silhouette_result))+2)


#KMedoids

result = [run_kmedoids(i+1, data) for i in range(7)][1:]
print(result)

ssd_result = [ i[0] for i in result]  # Getting first item of the array
silhouette_result = [ i[1] for i in result][1:] # Getting second item of the array

pyplot.plot(range(1,8), result)
pyplot.savefig("ssd.png")
pyplot.close() # Interpretaion: see almost linear flat curve after n=4


result_diff = [result[i-1] - x for i,x in enumerate(results)] [1:] # [1:] is the way to drop the first element
print(result_diff)

pyplot.plot(range(2,8), silhouette_result)
pyplot.savefig("silhouette.png")
pyplot.close()

print("\nssd: \n", ssd_result)
print("\nssd differences: \n", ssd_result_diff)


print("\nsilhouette scores: \n", silhouette_result)
print("\nmax silhouette scores: \n", max(silhouette_result))
print("\nnumber of cluster with max silhouette scores: \n", silhouette_result.index(max(silhouette_result))+2)

