from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange




def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.
  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength
  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)
        softmax = np.exp(scores) / np.sum(np.exp(scores),keepdims = True)
        for j in xrange(num_classes):
            if j == y[i]:
                loss += -(np.log(softmax[j]))
                dW[:,j] += (softmax[j]-1) * X[i].T
            else:
                dW[:,j] += (softmax[j]) * X[i].T
                
  loss = loss / num_train          
  loss += reg*np.sum(W*W)
  dW = dW / num_train
  dW = dW + reg*W
  
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N,D = X.shape
    D,C = W.shape

    f = np.dot(X,W) # N,C
    f = f- np.max(f,axis=1,keepdims=True) # numerical stability

    softmax = np.exp(f)/np.sum(np.exp(f),axis=1,keepdims=True) #(N,C)
    loss = np.sum(-np.log(softmax[np.arange(N),y]))
    #loss = np.sum(-f[np.arange(N),y] + np.log(np.sum(np.exp(f),axis=1)))
    softmax[np.arange(N),y] += -1 # for j==i case
    dW += np.dot(X.T,softmax) # (D,C)  

    loss /=N
    dW /=N

    loss+=reg*np.sum(W*W)
    dW += 2*reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
