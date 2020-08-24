
# Import external dependencies
import density_push as dp
import math
import matplotlib.pyplot as plt
import torch
import scipy.io
import numpy as np

def sdot_asgd(y, nu, C, x_sample, W = None):
    """
         Semi discrete regularized optimal transport
           by Average stochastic gradient descent.
        
         Input:
         - y: set of Dirac locations of target distribution
         - nu: masses of Diracs
         - C: gradient step
         - x_sample_points: sampled points from source distribution
        
         Ouput:
         - W: Weights for the Optimal Transport 

         NB: 
           - we use c(x,y) = |x-y|^2 as cost function
    """
    # if W == None: W = np.zeros(y.shape[0]) else: assert(W.shape[0] == y.shape[0])
    W = np.zeros(y.shape[0])       # (500, 0)
    W_tmp = np.copy(W)
    #source_density = dp.get_density_by_name(name_source)  # Density of source distribution
    h_save = np.empty_like(0)
    # Print iteration status
    niter = np.shape(x_sample)[0]
    for t in range(niter):
        if (t+1) % 10000 == 0:
            print("Iteration: {}".format(t+1))
    
        # Sample from source distribution
        #x = source_density.sample_from(1).numpy()
        x = x_sample[t]

        # Gradient Step
        r = np.sum(np.square(x-y) , axis=1) - W_tmp  # |x-y|^2 - W_tmp (900, )
        indx_min = np.argmin(r)
        grad = np.copy(nu)
        grad[indx_min] = grad[indx_min] - 1  # (900, )

        # Evaluate empirical Reward
        r2 = np.sum(np.square(x-y) , axis=1) - W  # |x-y|^2 - W_tmp (900, )
        h = np.min(r2) + np.dot(W,nu) 
        h_save = np.hstack((h_save,h))

        # Gradient Ascent 
        W_tmp = W_tmp + C/np.sqrt(t+1) *grad # t+1 because it starts from 0
        W = t/(t+1) *W + 1/(t+1)*W_tmp  # t+1 because it starts from 0
        # W = W / np.max(np.abs(W))    

    return W, h_save


if __name__ == "__main__":
    # Define target distribution
    source_density = dp.get_density_by_name("uniform")  # Density of source distribution
    y = source_density.sample_from(500).numpy() # (500,2) # 2 dimensional
    nu = np.ones(y.shape[0])/y.shape[0]

    # Number of iterations 
    niter= 1000000

    # Gradient step
    C = 1

    def module_items_with_prefix(module, prefix): 
        return [module.__dict__[export] for export in dir(module) if export.startswith(prefix)] 
    distribution_names = module_items_with_prefix(dp, "TOY_DISTRIB")   

    W, h_vect = sdot_asgd(y, nu, "banana", niter, C)
 #   np.save('./Figures/h_vect_1.npy', h_vect)
    
    #print(W)    
