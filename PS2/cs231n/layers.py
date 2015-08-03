import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  N = x.shape[0]
  out = np.reshape(x, (N, -1)).dot(w) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  db = np.sum(dout, axis=0)
  
  dx = dout.dot(w.T)
  dx = np.reshape(dx, x.shape)
  
  N = x.shape[0]
  dw = np.reshape(x, (N,-1)).T.dot(dout)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.maximum(0, x)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dx = dout * (x > 0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


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
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  (N, C, H, W) = x.shape
  (F, _, HH, WW) = w.shape
  
  stride = conv_param['stride']
  pad = conv_param['pad']
  
  Hp = 1 + (H + 2*pad - HH) / stride
  Wp = 1 + (W + 2*pad - WW) / stride

  out = np.empty((N, F, Hp, Wp))
  # loop over the inputs
  for n in xrange(N):
      # pad the image
      im = np.pad(x[n], ((0,0), (pad,pad), (pad,pad)), mode='constant', constant_values=0)
      # loop over the filters
      for f in xrange(F):
          # loop over the image windows
          for i in xrange(Hp):
              istart = i*stride
              iend = istart + HH
              for j in xrange(Wp):
                  jstart = j*stride
                  jend = jstart + WW
                  # get the *convolution* (actually correlation)
                  out[n, f, i, j] = np.sum(w[f] * im[:,istart:iend,jstart:jend]) + b[f]      
      
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives. (N, F, Hp, Wp)
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x (NxCxHxW)
  - dw: Gradient with respect to w (FxCxHHxWW)
  - db: Gradient with respect to b (F,)
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  # read cache
  x, w, b, conv_param = cache

  # sizes
  N, F, Hp, Wp = dout.shape
  
  (_, C, H, W) = x.shape
  (_, _, HH, WW) = w.shape
  
  stride, pad = conv_param['stride'], conv_param['pad']
  
  
  # allocate
  dx = np.zeros_like(x)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)

  # loop over the inputs
  for n in xrange(N):
      # pad the image
      im = np.pad(x[n], ((0,0), (pad,pad), (pad,pad)), mode='constant', constant_values=0)
      # padded derivative, remove padding later
      dxn = np.zeros_like(im)
      # loop over the filters
      for f in xrange(F):
          # loop over the image windows
          for i in xrange(Hp):
              istart = i*stride
              iend = istart + HH
              for j in xrange(Wp):
                  jstart = j*stride
                  jend = jstart + WW
                  # now we have the input gradient for this neuron in out[n,f,i,j]
                  # and we want to distribubte that over b, x, w that contributed
                  # to it throug the equation
                  # out[n, f, i, j] = np.sum(w[f] * im[:,istart:iend,jstart:jend]) + b[f]
                  dd = dout[n,f,i,j]
                  
                  # wrt b
                  db[f] += dd
                  
                  # wrt w
                  dw[f] += dd * im[:,istart:iend,jstart:jend]
                  
                  # wrt x
                  dxn[:, istart:iend, jstart:jend] += dd * w[f] 
                  
      # remove paddding
      dx[n] = dxn[:, pad:-pad, pad:-pad]
                  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
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
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  pH= pool_param['pool_height']
  pW = pool_param['pool_width']
  stride = pool_param['stride']
  
  # how many output windows i.e. output spatial size 
  N, C, H, W = x.shape
  HH = (H - pH) / stride + 1
  WW = (W - pW) / stride + 1
  
  # allocate
  out = np.zeros((N, C, HH, WW))
  
  # loop
  for n in xrange(N):
    for c in xrange(C):
      for i in xrange(HH):
        # window in input in i dimension
        istart = i * stride
        iend = istart + pH
        
        for j in xrange(WW):
          # window in input in j dimension
          jstart = j * stride
          jend = jstart + pW
          
          # get max
          out[n, c, i, j] = np.max(x[n,c,istart:iend,jstart:jend])
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
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
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x, pool_param = cache
  
  pH= pool_param['pool_height']
  pW = pool_param['pool_width']
  stride = pool_param['stride']
  
  # how many output windows i.e. output spatial size 
  N, C, HH, WW = dout.shape
  
  # allocate
  dx = np.zeros_like(x)
  
  # loop
  for n in xrange(N):
    for c in xrange(C):
      for i in xrange(HH):
        # window in input in i dimension
        istart = i * stride
        iend = istart + pH
        
        for j in xrange(WW):
          # window in input in j dimension
          jstart = j * stride
          jend = jstart + pW
          
          # distribute dout according to this
          # dout[n, c, i, j] = np.max(x[n,c,istart:iend,jstart:jend])
          # i.e. find the location where the maximum was and put dout there
          ii, jj = np.unravel_index(np.argmax(x[n,c,istart:iend,jstart:jend]), 
                                    dims=(pH, pW))
          dx[n, c, istart+ii, jstart+jj] = dout[n,c,i,j]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

