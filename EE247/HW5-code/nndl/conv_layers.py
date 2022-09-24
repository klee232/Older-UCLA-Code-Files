import numpy as np
from nndl.layers import *
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

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #

  # return the scales of the input and the filter
  N,C,H,W = x.shape
  F,C,Hf,Wf = w.shape
  
  # padding the input
  padding_x = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant', constant_values=0)
  # changes the output size
  N,C,Hp,Wp = padding_x.shape
  
  # convolutional layers
  # setup the output array for convolutional layer
  Hc = (H-Hf+2*pad)/stride + 1
  Wc = (W-Wf+2*pad)/stride + 1
  conv_out = np.zeros((N,F, int(Hc), int(Wc)))

  # loop through each input
  
  for d in range(0, N):
    # reset wi everytime it finishes a dataset
    counter_h = -1
    counter_w = -1
    for hi in range(0,Hp-Hf+1,stride):
        #print("Wp-Wf:", Wp-Wf)
        #print("stride:", stride)
        #print(wi)
        # refresh counter for h 
        counter_h += 1
        for wi in range(0,Wp-Wf+1,stride):
            #print("Hp-Hf:", Hp-Hf)
            #print("stride:", stride)
            #print(hi)
            # refresh counter for w
            counter_w += 1
            for fi in range(0, F):
                #print(fi)
                #print(d,fi,hi,wi)
                # conv_out[d,fi,counter_h,counter_w] = np.sum(w[fi]*padding_x[d, :, hi*stride:hi*stride+Hf, wi*stride:wi*stride+Wf]) + b[fi]
                conv_out[d,fi,counter_h,counter_w] = np.sum(w[fi]*padding_x[d, :, hi:hi+Hf, wi:wi+Wf]) + b[fi]
        
        counter_w = -1
  #print(conv_out)
  out = conv_out
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #
  N,C,H,W = x.shape
  F,C,Hf,Wf = w.shape
  
    
  dx = np.zeros((x.shape))
  dw = np.zeros((w.shape))
  db = np.zeros((b.shape))
  
  # the derivative of xpad
  # setup the output array for convolutional layer
  N,C,Hp,Wp = xpad.shape
  dxpad = np.zeros((xpad.shape))
  # get the parameters for padding
  Hc = (H-Hf+2*pad)/stride + 1
  Wc = (W-Wf+2*pad)/stride + 1
  
  # the derivative of b
  for fi in range(F):
    db[fi] = np.sum(dout[:,fi,:,:])
  
  # the derivative of w
  for fi in range(F):
    for ci in range(C):
        for hi in range(Hf):
            for wi in range(Wf):
                dtemp = dout[:,fi,:,:]*xpad[:,ci, hi:hi+out_height * stride:stride, wi:wi+out_width * stride:stride]
                dw[fi,ci,hi,wi] = np.sum(dtemp)
  
  
  # derive the derivative of padding x
  for d in range(N):
    for fi in range(F):
        for hi in range(int(Hc)):
            for wi in range(int(Wc)):
                dxpad[d, :, hi*stride:hi*stride+Hf, wi*stride:wi*stride+Wf] += w[fi]*dout[d,fi,hi,wi]
  
  # derive the derivative of x
  N,C,H,W = x.shape
  dx = dxpad[:, :, pad:pad+H, pad:pad+W]
  
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """

  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #
  # get the parameters for input x
  N,C,H,W = x.shape
  
  # derive the parameters (copied from the latter part)
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride'] 
  
  # set up variable for storing output
  Hmax = (H-pool_height)/stride + 1
  Wmax = (W-pool_width)/stride + 1
  Hmax = int(Hmax)
  Wmax = int(Wmax)
  out = np.zeros((N, C, Hmax, Wmax))
  
  for d in range(N):
    counter_h = -1
    counter_w = -1
    for hi in range(0, H-pool_height+1, stride):
        counter_h += 1
        for wi in range(0, W-pool_width+1, stride):
            counter_w += 1
            for ci in range(C):
                out[d,ci,counter_h,counter_w] = np.max(x[d,ci,hi:hi+pool_height, wi:wi+pool_width])
        counter_w = -1

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """

  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  # set up variable for storing dx
  dx = np.zeros((x.shape))
  
  # get the parameters for input x
  N,C,H,W = x.shape
  
  # set up variable for storing output
  Hmax = int((H - pool_height)/stride + 1)
  Wmax = int((W - pool_width)/stride + 1)
  
  for d in range(N):
    for hi in range(0, Hmax):
        for wi in range(0, Wmax):
            for ci in range(C):
                xn = x[d,ci,hi*stride:hi*stride+pool_height, wi*stride:wi*stride+pool_width]
                xmax = np.max(xn)
                dx[d,ci,hi*stride:hi*stride+pool_height, wi*stride:wi*stride+pool_width] = dout[d,ci,hi,wi]*(xn==xmax)
                
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  # get the parameters for input x
  N,C,H,W = x.shape
  
  # transpose x
  xt = np.transpose(x,(0,2,3,1))
  
  # reshape the x into [N*H*W, C]
  reshaped = xt.reshape(N*H*W, C)
  
  # insert the reshaped x into batchnorm function
  out, cache = batchnorm_forward(reshaped, gamma, beta, bn_param)

  # change it back to the original shape and order 
  out = out.reshape(N, H, W, C)
  # rearranging the columns
  out = np.transpose(out,(0,3,1,2))
  

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  # get the parameters for input dout
  N,C,H,W = dout.shape
  
  # transpose dout 
  dt = np.transpose(dout,(0,2,3,1))
  # reshape the dout
  dt = dt.reshape(N*H*W,C)
  
  # insert the input to batchnorm
  output, dgamma, dbeta = batchnorm_backward(dt, cache)
  
  # derive the derivative of x
  output = output.reshape(N,H,W,C)
  # rearranging the columns
  dx = np.transpose(output,(0,3,1,2))
  

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta