# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 00:20:47 2020
"""

# Import external dependencies
import density_push as dp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import seaborn as sns
import torch

# Import 1-layer algorithm
from sdot_1layer import sdot_asgd
from sdot_2layer import sdot_asgd_2layer, sdot_cluster

def create_grid(x_min, x_max, y_min, y_max, x_n, y_n):
    x = np.linspace(x_min, x_max, num=x_n)
    y = np.linspace(y_min, y_max, num=y_n)
    A = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    return(A)


def scatter_plot_coords(grid):
    x_col = np.zeros(len(grid))
    y_col = np.zeros(len(grid))
    
    for i, point in enumerate(grid):
        x_col[i] = point[0]
        y_col[i] = point[1]

    # TODO: normalize weights for s (size)
    return x_col, y_col


def generate_data():
    # target distribution
    y = create_grid(-1.5, 3, -1, 4.5, 40, 40)
    fn = "sample_points_grid.npy"
    np.save(fn, y)
    nu = np.ones(y.shape[0]) / y.shape[0]   # Uniform distribution for nu

    for n_iter in [10**0, 10**3, 10**4, 10**5, 2*10**6]: # iterations for 10 clusters
#    for n_iter in [10**3, 10**4, 5*10**4, 2*10**5]: # iterations for 200 clusters
        # source distribution
        source_density = dp.get_density_by_name("banana")
        x_sample = source_density.sample_from(n_iter).numpy()
        fn = "sample_points_source_density_{}.npy".format(n_iter)
        np.save(fn, x_sample)
        
        # optimal transport (1-layer)
        # def sdot_asgd(y, nu, niter, C, x_sample):
        #W, h_save = sdot_asgd_2layer(y, nu, n_iter, 1, x_sample)
        n_clusters = 10
        kmeans, nu1 = sdot_cluster(y, nu, n_clusters)
        y_labels, y1 = kmeans.labels_, kmeans.cluster_centers_
        W, W1 = sdot_asgd_2layer(y, y_labels, nu, y1, nu1, x_sample, C=1)
        fn = "weights_grid_cluster_{}_{}.npy".format(n_clusters, n_iter)
        np.save(fn, W)
        #fn = "h_save_grid_{}.npy".format(n_iter)
        #np.save(fn, h_save)


def get_source_density(name):
    density = dp.get_density_by_name(name)
    id_projection = torch.eye(density.dim)[-2:, :]
    marginal_density = density.marginal(id_projection)
    return density, marginal_density


def main():
 #   generate_data()
    # load data
    G = np.load("sample_points_grid.npy")
    #W = np.load("weights_grid.npy")
    #h_save = np.load("h_save_grid.npy")
    x_col, y_col = scatter_plot_coords(G)

    # create scatter plot with legend
    cmap = sns.cubehelix_palette(as_cmap=True)
    # f, ax = plt.subplots()
    # points = ax.scatter(x_col, y_col, c=W, cmap=cmap)
    # f.colorbar(points)
    # ax.axis('off')
    # plt.show(f)
    # f.savefig("grid_1600_300dpi.png", dpi=300)
    
    # create plot of continous density
    density, marginal_density = get_source_density("banana")
    dp.vis.density_contours(marginal_density, resolution=300, fill=True)
    #plt.savefig("banana_300dpi.png", dpi=300)
    
    # create subplots for multiple iterations
    # load data
    # 10 cluster
#    W0 = np.load("weights_grid_cluster_10_1000.npy")
#    W1 = np.load("weights_grid_cluster_10_10000.npy")
#    W2 = np.load("weights_grid_cluster_10_100000.npy")
#    W3 = np.load("weights_grid_cluster_10_2000000.npy")

#    # 200 cluster
    W0 = np.load("weights_grid_cluster_200_1000.npy")
    W1 = np.load("weights_grid_cluster_200_10000.npy")
    W2 = np.load("weights_grid_cluster_200_50000.npy")
    W3 = np.load("weights_grid_cluster_200_200000.npy")
#    
    fig, ax = plt.subplots(2, 2)
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)
    ax[0, 0].scatter(x_col, y_col, c=W0, cmap=cmap)
    #ax[0, 0].scatter(x_col, y_col, s=0.05, cmap=cmap)
    ax[0, 0].set_title('k = $10^3$')
    ax[0, 0].axis('equal')
    ax[0, 0].axis('off')
    ax[0, 1].scatter(x_col, y_col, c=W1, cmap=cmap)
    #ax[0, 1].scatter(x_col, y_col, s=0.05, cmap=cmap)
    ax[0, 1].set_title('k = $10^4$')
    ax[0, 1].axis('equal')
    ax[0, 1].axis('off')
#    ax[1, 0].scatter(x_col, y_col, c=W2, cmap=cmap)
    ax[1, 0].scatter(x_col, y_col, c=W2, cmap=cmap) # 10 cluster
#    ax[1, 0].set_title('k = $10^5$')
    ax[1, 0].set_title('k = $5 * 10^4$')
    ax[1, 0].axis('equal')
    ax[1, 0].axis('off')
    ax[1, 1].scatter(x_col, y_col, c=W3, cmap=cmap)
    #ax[1, 1].scatter(x_col, y_col, s=0.05, cmap=cmap)
#    ax[1, 1].set_title('k = $2*10^6$')  # 10 cluster
    ax[1, 1].set_title('k = $2*10^5$')
    ax[1, 1].axis('equal')
    ax[1, 1].axis('off')
    fig.savefig("grid_k_cluster_300dpi.jpg", dpi=300, bbox_inches='tight')
    
    

if __name__ == "__main__":
    main()
