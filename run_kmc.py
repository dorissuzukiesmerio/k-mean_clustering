# K-Mean Clustering Algorithm

import pandas
from sklearn.cluster import KMeans # python -m pip install scikit-learn
import matplotlib.pyplot as pyplot

data = pandas.read_csv("dataset.csv")

print(data)

# Example 1: x1 = Density of tree , x2 =  number of shops
# Example 2: x1 = Number of tickets if individual, x2 = occupation of individual

# Transformation: Sklearn doesn't support DataFrame format very well -> so there is a need to transform to 
data = data.values # two values in one array, array of arrays ; loads to sklearn better; though looses some functionalities such as matrix operation
print(data)


# Visualization: 
# Scatterplot (x, y)
pyplot.scatter(data[:,0], data[:,1]) # row, column (all rows, first column)
pyplot.savefig("scatterplot.png") # no need for this if using Jupyter Notebook


###  IMPLEMENTING ALGORITHM ###
# Step 1 - Construct the Machine:
machine = KMeans(n_clusters = 4) 
# KMeans() is from the imported package
# specify the number of clusters

# Step 2 - Fit the data:
machine.fit(data)

# Step 3 - Predict the data:
machine.predict(data)

# Interpretation:
# 400 dots; 400 observations ; 400 group numbers assigned to the result
# first observation: assigned to group 1
# second : group 3 
# ..... and so on.....

# Visualization of results: 
# Scatterplot (x, y)
pyplot.scatter(data[:,0], data[:,1], c = results) # row, column (all rows, first column)
pyplot.savefig("scatterplot2.png") # no need for this if using Jupyter Notebook
