# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:51:00 2020
"""

from sklearn.cluster import KMeans
from math import isclose
import density_push as dp
import numpy as np

def sdot_cluster(target_points, target_measure, n):
    """
    sdot_cluster

    Parameters
    ----------
    target_points : np.array
        DESCRIPTION.
    target_measure : np.array
        DESCRIPTION.
    n : int
        DESCRIPTION.

    Returns
    -------
    kmeans, target_measure_clusters

    """
    kmeans = KMeans(n_clusters=n, init="k-means++").fit(target_points)
    labels = np.unique(kmeans.labels_) # sorted unique labels
    target_measure_clusters = np.empty(len(labels))
    
    for i in labels: # clusters 0...n
        idx = np.where(kmeans.labels_ == i) # cluster i
        # sigma additivity for discrete measure
        target_measure_clusters[i] = sum(target_measure[idx])

    return kmeans, target_measure_clusters


# def sdot_asgd_2layer(target_points, target_measure,
#                      target_points_clustered, target_measure_clusters, 
#                      source_points_sample, C=1, n_clustered=10):
def sdot_asgd_2layer(S0, S0_labels, Nu0, S1, Nu1, x_sample, C=1, 
                     W0 = None, W1 = None):
    """
    sdot_asgd_2layer
    
    Parameters
    ----------
    target_points : np.array
        DESCRIPTION.
    target_measure : np.array
        DESCRIPTION.
    target_points_clusters : np.array
        DESCRIPTION.
    target_measure_clusters : np.array
        DESCRIPTION.
    source_points_sample : np.array
        DESCRIPTION.
    C : float, optional
        DESCRIPTION. The default is 1.
    n_clusters : int, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    W1, W0

    """
    # initialize weight vectors for the first layer
    # TODO: S1 enthaelt cluster-label fuer ALLE indizes
    W1 = np.zeros(S1.shape[0]) # (10, 0)
    W1_tmp = np.zeros(S1.shape[0])
    
    # initialize weight vectors for the second layer
    W0 = np.zeros(S0.shape[0]) # (500, 0)
    W0_tmp = np.zeros(S0.shape[0])
    # initialize number of visits to the power cells Eta1
    Eta1 = np.zeros(S1.shape[0])
    
    # iterate over source distribution samples
    n_iter = np.shape(x_sample)[0]
    for t in range(n_iter):
        if (t+1) % 10000 == 0:
            print("Iteration: {}".format(t+1))
        # LAYER 1
        # Take sample x
        x = x_sample[t]

        # Compute argmin
        # !assumes S1 (kmeans.cluster_centers_) are ordered by cluster, or
        # kmeans.cluster_centers_[i] = <centroid of cluster i>
        r = np.sum(np.square(x - S1), axis=1) - W1
        s1_index = np.argmin(r) # index of s_tilde in S1
        
        # Update number of visits to the power cells
        Eta1[s1_index] = Eta1[s1_index] + 1
        
        # Compute gradient approximation
        grad = np.copy(Nu1)
        grad[s1_index] = grad[s1_index] - 1 # (10, 0)

        # Update weight vector (gradient ascent)
        W1_tmp =  W1_tmp + C/np.sqrt(t+1)*grad # t+1 because it starts from 0
        W1 = t/(t+1)*W1 + 1/(t+1)*W1_tmp # t+1 because it starts from 0

        # # Evaluate empirical Reward
        # r2 = np.sum(np.square(x-y) , axis=1) - W  # |x-y|^2 - W_tmp (900, )
        # h = np.min(r2) + np.dot(W,nu) 
        # h_save = np.hstack((h_save,h))

        # LAYER 2
        # Compute preimage pi_{0}^{-1} for s1_index
        pi_index = np.nonzero(S0_labels == s1_index)[0]
        pi_preim = S0[pi_index]

        r2 = np.sum(np.square(x - pi_preim), axis=1) - W0[pi_index]
        s2_index = np.argmin(r2)

        # Compute gradient (nu0 - lenght)
        g_padding = np.zeros(S0.shape[0])
        g_padding[pi_index] = Nu0[pi_index] 
        g_padding[pi_index[s2_index]] -= 1  
    
        # Compute gradient approximation
#        grad2 = np.zeros(len(pi_preim))
#        grad2[pi_index] = Nu0[pi_index] - 1      
#        s2 = pi_preim[s2_index]
#        comp_index = np.nonzero(pi_preim != s2)
#        grad2[comp_index] = Nu0[comp_index]
        
        # Update weight vector (Gradient ascent) and average weights
        W0_tmp= W0_tmp + C/np.sqrt(Eta1[s1_index])* g_padding
        W0[pi_index] = 1/Eta1[s1_index]*W0_tmp[pi_index] + (Eta1[s1_index]-1)/Eta1[s1_index]*W0[pi_index]

        # result
    return W0, W1


def main(): # tests
    # cluster fuer source_points (stetig verteilt)
    niter = 2000 # amount of sampled points
    source_density = dp.get_density_by_name("banana")  # Density of source distribution
    source_points = source_density.sample_from(niter).numpy()
    np.save("sample_points_source_density", source_points)
    
    # 10 clusters: 0...9
    # indices for x vector => np.where
    kmeans = KMeans(n_clusters=10, init="k-means++").fit(source_points)
    # urbild von pi_0 (beispiel)
    idx = np.where(kmeans == 5)
    print(source_points[idx])
    
    # cluster fuer target_points (diskret verteilt)
    target_density = dp.get_density_by_name("uniform")
    target_points = target_density.sample_from(500)
    np.save("sample_points_target_density", target_points)
    
    target_measure = np.ones(target_points.shape[0]) / target_points.shape[0]
    n_clusters = 10
    
    target_kmeans, target_measure_clusters = sdot_cluster(
        target_points, target_measure, n_clusters)
    print(target_measure)
    print(target_measure_clusters)
    print(target_kmeans.labels_)
    print(target_kmeans.cluster_centers_)
    
    # 2-layer algorithmus
    S1 = target_kmeans.cluster_centers_
    S1_labels = target_kmeans.labels_
    S0 = source_points


if __name__ == "__main__":
    main()