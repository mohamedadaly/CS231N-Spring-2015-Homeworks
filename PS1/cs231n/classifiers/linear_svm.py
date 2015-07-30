import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]: continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
          # loss
          loss += margin      
          # gradient for dW[y[i]]
          dW[y[i]] -= X[:,i]          
          # gradient for dW[j] for j != y[i]
          dW[j] += X[:,i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  # normalize
  dW /= num_train
  # regularization
  dW += reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  
  num_classes = W.shape[0]
  num_train = X.shape[1]  

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # compute scores: one column per image
  scores = W.dot(X)
  
  # correct scores
  correct_scores = scores[y,range(num_train)]
  
  # subtract
  margins = scores - correct_scores + 1
  
  # add loss for +ve margin. Subtract 1 from the entry for the correct_scores
  pos_margins = margins > 0
  loss = np.sum(margins[pos_margins]) / num_train - 1
  # add regularization
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
#  for i in xrange(num_train):
#      idx = np.arange(num_classes) != y[i]
#      dW[y[i]] -= (np.sum(pos_margins[:,i]) - 1) * X[:,i]
#      dW[idx] += np.tile(X[:,i], (num_classes-1, 1))
  for c in range(num_classes):
    # get all examples that are not of this class
    not_class = np.flatnonzero(np.logical_and(y != c, pos_margins[c]))
    # add their X's to W[c] if their margins > 0
    #dW[c] += np.sum(X.T[np.logical_and(not_class, pos_margins[c]),:], axis=0)
    dW[c] += np.sum(X[:, not_class], axis=1)
    #dW[c] += np.sum(X * np.logical_and(not_class, pos_margins[c]), axis=1)
    
    # get all examples that are of this class
    this_class = np.flatnonzero(y == c)
    # add their -X's if their margins > 0
    dW[c] -= np.sum(X[:, this_class] * 
                    (np.sum(pos_margins[:,this_class], axis=0) - 1), 
                    axis=1)
                    
  # normalize
  dW /= num_train
  # reg.
  dW += reg * W
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
