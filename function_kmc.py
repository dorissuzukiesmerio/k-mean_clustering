# K-Mean Clustering Algorithm : WRITING FUNCTION

import pandas
from sklearn.cluster import KMeans # python -m pip install scikit-learn
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
	pyplot.savefig("scatterplot_colors_"+ str(n) + ".png") # no need for this if using Jupyter Notebook
	pyplot.close()
	return ssd, silhouette

# run_means(4, data)

# #Now:
# for number in range(4):
# 	run_means(number)

# for i in range(7):
# 	ssd = run_kmeans(i+1, data)
# 	result.append(ssd)

result = [run_kmeans(i+1, data) for i in range(7)][1:]
print(result)

pyplot.plot(range(1,8), result)
pyplot.savefig("ssd.png")
pyplot.close() # Interpretaion: see almost linear flat curve after n=4

# result_diff = []
# for i,x in enumerate(results)
# 	diff = result[i-1] - x
# 	result_diff.append(diff)

result_diff = [result[i-1] - x for i,x in enumerate(results)] [1:] # [1:] is the way to drop the first element
print(result_diff)

# Remembering:
# The closest center of gravity 
# Two conditions that need to be satisfied

# INTERPRETATION:
# ssd : sum of distance given the number of clusterings - > so you cannot compare the numbers 
# Decreasing from 4144 to 1900 to 1100 to 600 to 500 (highest decrease)


# Silhouette Score: the highest peak in the graph
# Find the highest number
# A = mean intra-cluster distance
# B = mean nearest-cluster distance

# S = B - A / max(A, B)

# ssd_result_diff = [ ssd_result[i-1] - x for i,x  in enumerate(ssd_result)][1:]

pyplot.plot(range(2,8), silhouette_result)
pyplot.savefig("silhouette.png")
pyplot.close()

print("\nssd: \n", ssd_result)
print("\nssd differences: \n", ssd_result_diff)


print("\nsilhouette scores: \n", silhouette_result)
print("\nmax silhouette scores: \n", max(silhouette_result))
print("\nnumber of cluster with max silhouette scores: \n", silhouette_result.index(max(silhouette_result))+2)

