import numpy as np
from random import shuffle

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[1]
  C = W.shape[0]
  
  for i in xrange(N):
      # score
      score = W.dot(X[:,i])
      # probability = softmax of score
      prob = softmax(score)
      # add loss
      loss += -np.log(prob[y[i]])
      
      # get y_hat - y
      prob[y[i]] -= 1
      # add gradient
      dW += np.outer(prob, X[:,i])
      
  # normalize
  loss /= N
  dW /= N
  
  # regularize
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[1]

  # scores: one column per image
  scores = W.dot(X)
  
  # prob
  # subtract max
  prob = scores - np.max(scores, axis=0)
  # compute exp
  prob = np.exp(prob)
  # divide by sum
  prob = prob / np.sum(prob, axis=0)
  
  # compute loss
  loss = -np.sum(np.log(prob[y, range(N)]))
  
  # get y_hat - y
  prob[y,range(N)] -= 1
  
  # gradient: outer product of prob and X (same as covariance)
  # i.e. the sum of outer products outer(prob, X[:,i]) is the same as the 
  # matrix product prob.dot(X.T)
    #x = np.random.randn(2,3)
    #p = np.random.randn(5,2)
    #z1 = np.outer(p[:,0], x[0]) + np.outer(p[:,1], x[1])
    #z2 = p.dot(x)  
  dW = prob.dot(X.T)
  
  # normalize
  loss /= N
  dW /= N
  
  # regularize
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
