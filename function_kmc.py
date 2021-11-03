# K-Mean Clustering Algorithm : WRITING FUNCTION

import pandas
from sklearn.cluster import KMeans # python -m pip install scikit-learn
import matplotlib.pyplot as pyplot

from sklearn.metrics import 

data = pandas.read_csv("dataset.csv")

print(data)
data = data.values # two values in one array, array of arrays ; loads to sklearn better; though looses some functionalities such as matrix operation
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
		print(silhouette_score(data, machine.labels_, metric = 'euclidean'))
	pyplot.scatter(data[:,0], data[:,1], c = results) # row, column (all rows, first column)
	pyplot.scatter(centroids[:,0], centroids[:,1], c='red', marker= "*", s=200)
	pyplot.savefig("scatterplot_colors_"+ str(n) + ".png") # no need for this if using Jupyter Notebook
	pyplot(close)
	return ssd, silhouette_score

run_means(4)

# #Now:
# for number in range(4):
# 	run_means(number)

# for i in range(7):
# 	ssd = run_kmeans(i+1, data)
# 	result.append(ssd)

result = [run_kmeans(i+1, data) for i in range(7)][1:]
print(result)

pyplot.plot(range(7), result)
pyplot.savefig("ssd.png")
pyplot.close()

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


# Silhouette Score:
# Find the highest number
# A = mean intra-cluster distance
# B = mean nearest-cluster distance

# S = B - A / max(A, B)


