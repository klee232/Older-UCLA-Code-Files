import numpy as np

from nndl.layers import *
from nndl.conv_layers import *
from cs231n.fast_layers import *
from nndl.layer_utils import *
from nndl.conv_layer_utils import *

import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=64, filter_size=3,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=True):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize the weights and biases of a three layer CNN. To initialize:
    #     - the biases should be initialized to zeros.
    #     - the weights should be initialized to a matrix with entries
    #         drawn from a Gaussian distribution with zero mean and 
    #         standard deviation given by weight_scale.
    # ================================================================ #
    self.bn_params = {}
    # get the parameters from input_dim
    C = input_dim[0]
    H = input_dim[1]
    W = input_dim[2]
  
    # modified version from fc_net from homework 4
    # convolutional layer
    self.params['W1'] = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))
    self.params['b1'] = np.ones(shape = num_filters)
    
    # padding the output size
    pad = (filter_size-1)/2
    wc = W - filter_size + 1 + 2*pad
    hc = H - filter_size + 1 + 2*pad
    
    # 2*2 max-pooling layer
    wp = 2
    hp = 2
    pstride = 2
    wp = (wc - wp)/pstride + 1
    hp = (hc - hp)/pstride + 1
    p_dim = int(num_filters*wp*hp)
    # weights for the second layer
    self.params['W2'] = np.random.normal(0, weight_scale,(p_dim, hidden_dim))
    self.params['b2'] = np.ones(shape = hidden_dim)

    # weights for the third relu layer
    self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
    self.params['b3'] = np.ones(shape =  num_classes)
    
    # implementing batchnormalization
    if (self.use_batchnorm == True):
        # copied from the previous homework
        self.bn_params['bn_param1'] = {'mode': 'train', 'running_mean': np.zeros(num_filters), 'running_var': np.zeros(num_filters)}
        self.params['gamma1'] = np.ones(shape = num_filters)
        self.params['beta1'] = np.zeros(shape = num_filters)
        # copied from the previous homework
        self.bn_params['bn_param2'] = {'mode': 'train', 'running_mean': np.zeros(hidden_dim), 'running_var': np.zeros(hidden_dim)}
        self.params['gamma2'] = np.ones(shape = hidden_dim)
        self.params['beta2'] = np.zeros(shape = hidden_dim)

    
    
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the three layer CNN.  Store the output
    #   scores as the variable "scores".
    # ================================================================ #
    # Copy from neural_net.py
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'
    
    if self.use_batchnorm is True:
        
        bn_param1 = self.bn_params['bn_param1']
        bn_param2 = self.bn_params['bn_param2']

        beta1 = self.params['beta1']
        beta2 = self.params['beta2']
        
        gamma1 = self.params['gamma1']
        gamma2 = self.params['gamma2']


    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    
    
    # inserting the input to convolutional-relu-max-pooling layer
    if(self.use_batchnorm == True):
        out1, cache11 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out1, cache12 = spatial_batchnorm_forward(out1, gamma1, beta1, bn_param1)
    else:
        out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    
    # reshaping the convolutional output
    N, F, H, W = out1.shape
    out1.reshape(N, F*H*W)
    
    # inserting the output from previous layer to affine-relu layer
    if(self.use_batchnorm == True):
        out2, cache21 = affine_forward(out1, W2, b2) 
        out2, cache22 = batchnorm_forward(out2, gamma2, beta2, bn_param2)
        out2, cache23 = relu_forward(out2)
    else:
        out2, cache2 = affine_relu_forward(out1, W2, b2)
    
    # inserting the output from the previous layer to the last affine-relu layer
    scores, cache3 = affine_forward(out2, W3, b3)

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    if y is None:
      return scores
    
    loss, grads = 0, {}
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of the three layer CNN.  Store the grads
    #   in the grads dictionary, exactly as before (i.e., the gradient of 
    #   self.params[k] will be grads[k]).  Store the loss as "loss", and
    #   don't forget to add regularization on ALL weight matrices.
    # ================================================================ #
    # calculate the loss 
    loss, gd_x = softmax_loss(scores,y)
    loss += 0.5*(self.reg*np.sum(W1**2) + self.reg*np.sum(W2**2)+self.reg*np.sum(W3**2))
    
    # calculate the gradient
    dmax, dw3, db3 = affine_backward(gd_x, cache3)
    if(self.use_batchnorm == True):   
        dmax = relu_backward(dmax, cache23)
        dmax, dgamma2, dbeta2 = batchnorm_backward(dmax, cache22)
        dmax, dw2, db2  = affine_backward(dmax, cache21)        
    else:
        dmax, dw2, db2 = affine_relu_backward(dmax, cache2)
        
    if(self.use_batchnorm == True):
        dmax, dgamma1, dbeta1  = spatial_batchnorm_backward(dmax, cache12)
        dx, dw1, db1 = conv_relu_pool_backward(dmax, cache11)
    else:
        dx, dw1, db1 = conv_relu_pool_backward(dmax, cache1)
    
    #add the cost
    dw3 += self.reg*W3
    dw2 += self.reg*W2
    dw1 += self.reg*W1
    
    # store the values
    grads['W3'] = dw3
    grads['b3'] = db3
    grads['W2'] = dw2
    grads['b2'] = db2
    grads['gamma2'] = dgamma2
    grads['beta2'] = dbeta2
    grads['W1'] = dw1
    grads['b1'] = db1
    grads['gamma1'] = dgamma1
    grads['beta1'] = dbeta1
    
     
    

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss, grads
  
  
pass
