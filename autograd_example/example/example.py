import autograd.numpy as np
from autograd import grad
X, w = np.random.randn(1000, 10), np.random.randn(10) # 1000 vectors of 10 features
Y, W = np.sin(np.dot(X,w)/10), np.random.randn(10+1, 50) # w: defined dependency between X and Y, 50: # neurons
def NN_Obj(W): # shallow net of the form : max( X^T W[:-1,] )^T W[-1,]
    return 0.001*np.linalg.norm(np.dot(np.maximum( np.dot( X, W[:-1,] ), 0 ) , W[-1,]) - Y, 2) ** 2; # MSE 
gradient = grad(NN_Obj) # Obtain gradient of L2 regression with shallow nn 
for i in range(10000): # simple gradient descent for number of epochs 
    W = W - 0.01 * gradient(W); # step with gradient 
    print NN_Obj(W) # compute objective 