#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 18:10:10 2020

@author: andrew
"""

import pandas as pd
import numpy as np;  # Numerical Python library
import matplotlib.pyplot as plt;  # Matlab-like Python module
from sklearn.cluster import KMeans; # Class for K-means clustering
from sklearn.metrics import davies_bouldin_score;  # function for Davies-Bouldin goodness-of-fit 
from sklearn.decomposition import PCA; #  Class for Principal Component analysis


df = pd.read_csv("HTRU_2.csv")

print(df)

target = df["Class"]

df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)

print(df)


# Default parameters
n_c = 2; # number of clusters

# Kmeans clustering
kmeans = KMeans(n_clusters=n_c, random_state=2020);  # instance of KMeans class
kmeans.fit(df);   #  fitting the model to data
pulsar_labels = kmeans.labels_;  # cluster labels
pulsar_centers = kmeans.cluster_centers_;  # centroid of clusters
sse = kmeans.inertia_;  # sum of squares of error (within sum of squares)
score = kmeans.score(df);  # negative error
distX = kmeans.transform(df);
dist_center = kmeans.transform(pulsar_centers);
# both sse and score measure the goodness of clustering

# PCA with limited components
pca = PCA(n_components=2);
pca.fit(df);
pulsar_pc = pca.transform(df);  #  data coordinates in the PC space
centers_pc = pca.transform(pulsar_centers);  # the cluster centroids in the PC space

# Visualizing of clustering in the principal components space
fig = plt.figure(1);
plt.title('Clustering of the Pulsar data after PCA');
plt.xlabel('PC1');
plt.ylabel('PC2');
plt.scatter(pulsar_pc[:,0],pulsar_pc[:,1],s=50,c=pulsar_labels);  # data
plt.scatter(centers_pc[:,0],centers_pc[:,1],s=200,c='red',marker='X');  # centroids
plt.legend();
plt.show();

# Visualizing of clustering in the distance space
fig = plt.figure(2);
plt.title('Pulsar data in the distance space');
plt.xlabel('Cluster 1');
plt.ylabel('Cluster 2');
plt.scatter(distX[:,0],distX[:,1],s=50,c=pulsar_labels);  # data
plt.scatter(dist_center[:,0],dist_center[:,1],s=200,c='red',marker='X');  # centroids
plt.legend();
plt.show();


# Finding optimal cluster number
Max_K = 31;  # maximum cluster number
SSE = np.zeros((Max_K-2));  #  array for sum of squares errors
DB = np.zeros((Max_K-2));  # array for Davies Bouldin indeces
for i in range(Max_K-2):
    n_c = i+2;
    kmeans = KMeans(n_clusters=n_c, random_state=2020);
    kmeans.fit(df);
    pulsar_labels = kmeans.labels_;
    SSE[i] = kmeans.inertia_;
    DB[i] = davies_bouldin_score(df,pulsar_labels);

# Visualization of SSE values    
fig = plt.figure(3);
plt.title('Sum of squares of error curve');
plt.xlabel('Number of clusters');
plt.ylabel('SSE');
plt.plot(np.arange(2,Max_K),SSE, color='red')
plt.show();

# Visualization of DB scores
fig = plt.figure(4);
plt.title('Davies-Bouldin score curve');
plt.xlabel('Number of clusters');
plt.ylabel('DB index');
plt.plot(np.arange(2,Max_K),DB, color='blue')
plt.show();
