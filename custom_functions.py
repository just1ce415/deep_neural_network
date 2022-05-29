import numpy as np

def sigmoid(Z):
    return (1 / (1 + np.exp(-Z)), Z)

def relu(Z):
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    cache = Z 
    return A, cache

def sigmoid_backward(dA, activation_cache):
    Z = activation_cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def relu_backward(dA, activation_cache):
    Z = activation_cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ


if __name__ == '__main__':
    print(relu_backward(np.array([1, 2, 6]), np.array([3, -1, 4])))