## A Stochastic Approach for Two-layer Semi-discrete Optimal Transport

### Abstract
The semi-discrete optimal transport problem arises in numerous applications as a natural way to 
calculate differences between two given measures. However, the computational burden
to handle this problem tends to hinder practicable realizations, especially when 
working in high-dimensional spaces or with target measures that have large supports.  
In this work, we want to cope with this problem studying a stochastic optimization approach that uses the benefits
of a hierarchical multi-scale strategy in order to approximate optimal transport maps.
We investigate the two-layer approach as an unsupervised learning problem that seeks to find the best
approximation by maximizing the two-layer expected reward. 
Our main goal will consist in understanding whether the numerical 
advantages that we might gain using this strategy compensate the possible deviation of these approximations from
the optimal solution.
For this, we study the underlying geometry of the two-layer power maps and use them as the class of functions in which 
the best possible approximation is to be found.
Understanding them will give us a better intuition and insights on the optimization problem that arises when using this approach.
We conclude this work by analyzing and motivating a version of the Average Stochastic Gradient 
Ascent algorithm which will turn out as a very efficient strategy to solve the two-layer stochastic optimization problem.
