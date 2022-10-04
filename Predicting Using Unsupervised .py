#!/usr/bin/env python
# coding: utf-8

# Name : Merna alaa elden
# 
# Track : Data science & Business Analytics.
# 
# Task 2 : Predicting Using Unsupervised ML (K-means Algorithm) to predict the optimum number of clusters and represent it visually. 

# In[11]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# In[13]:


iris = pd.read_csv("D:\DATA SCIENCE\sparks\Iris.csv")
iris


# In[14]:


iris.isnull().sum()


# In[15]:


iris.describe()


# In[16]:


iris.info()


# In[17]:


iris.columns


# In[18]:


iris.corr()


# In[19]:


iris.drop(['Id'] , axis=1, inplace = True)


# In[20]:


iris.head()


# In[23]:


#split the data into training and test sets
iris_data = iris.iloc[:, [0, 1, 2, 3]].values
# Finding the optimum number of clusters for k-means classification
wcss = []
for i in range (1 , 11):
    kmeans = KMeans(n_clusters = i , init = 'k-means++' , max_iter = 300 , n_init = 10 , random_state =10)
    kmeans.fit(iris_data)
    wcss.append(kmeans.inertia_)
# Plotting the results onto a line graph, 
# allowing us to observe The elbow
plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()


# the optimum clusters is where the elbow occurs so we will choose the number of clusters as 3

# In[25]:


#creating kmeans classifier
kmeans = KMeans (n_clusters = 3 , init ='k-means++' , max_iter = 300 , n_init =10 , random_state = 10)
y_kmeans = kmeans.fit_predict(iris_data)


# In[26]:


#visualise the clusters
plt.scatter(iris_data[y_kmeans == 0, 0] , iris_data[y_kmeans == 0 ,1],
           s = 100 , c = 'purple' , label = 'Iris-setosa')
plt.scatter(iris_data[y_kmeans == 1, 0], iris_data[y_kmeans == 1, 1], 
            s = 100, c = 'yellow', label = 'Iris-versicolour')
plt.scatter(iris_data[y_kmeans == 2, 0], iris_data[y_kmeans == 2, 1],
            s = 100, c = 'black', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# In[ ]:




