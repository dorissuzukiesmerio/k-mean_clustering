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

