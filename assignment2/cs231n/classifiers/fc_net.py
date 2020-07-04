from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params['W1'] = weight_scale * np.random.randn(input_dim,hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        l1,l1catch = affine_forward(X, self.params['W1'],self.params['b1']) 
        l2,l2catch = relu_forward(l1)
        l3,l3catch =  affine_forward(l2, self.params['W2'],self.params['b2'])
        scores = l3
        
      
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss,dx = softmax_loss(scores, y)
        loss += 0.5 *self.reg *(np.sum(self.params['W2']**2) + np.sum(self.params['W1']**2))
        (dx,dw,db)= affine_backward(dx,l3catch)

        grads['W2'] = dw + self.reg*self.params['W2']
        grads['b2'] = db
        
        catch = (l1)
        dx = relu_backward(dx, l2catch)

        (dx,dw,db)= affine_backward(dx, l1catch)

        grads['W1'] = dw + self.reg*self.params['W1']
        grads['b1'] = db

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        scores = []
        j=1
        first_dim = input_dim

        for h_dim in hidden_dims :
          str_j = str(j)
          w,b = 'W'+str_j,'b'+str_j
          #add batch normelazation parameters
          if (normalization == 'batchnorm'):
            gamma,betta = 'gamma'+str_j,'betta'+str_j
            self.params[gamma] = np.ones((h_dim,))
            self.params[betta] = np.zeros((h_dim,))
          self.params[w] = weight_scale * np.random.randn(first_dim,h_dim)
          self.params[b] = np.zeros(h_dim)
         
          #update first dim and couner j
          first_dim = h_dim
          j += 1

        #add output layer
        str_j = str(j)
        w,b  = 'W'+str_j,'b'+str_j
        self.params[w] = weight_scale * np.random.randn(first_dim,num_classes)
        self.params[b] = np.zeros(num_classes)
        #add batch norm param for output layer 
        
            


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
         #define
        scores = []
        catches = []
        last_score = 0

        #add input layer
        w,b = 'W1' ,'b1'
        score,catch = affine_forward(X, self.params[w],self.params[b])
        scores.append(score)
        catches.append(catch)
        if self.normalization == "batchnorm":
          gamma,betta = 'gamma1','betta1'
          score,catch = batchnorm_forward(scores[last_score],self.params[gamma],
                                      self.params[betta], self.bn_params[0])
          scores.append(score)
          catches.append(catch)
          last_score += 1    
        #set activation function
        score,catch = relu_forward(scores[last_score])
        scores.append(score)
        catches.append(catch)

        #add hidden layers
        layer_num = 2
        while( layer_num != self.num_layers):
          w,b = 'W' + str(layer_num),'b' + str(layer_num)
          last_score = len(scores)-1
          score,catch = affine_forward(scores[last_score], self.params[w],self.params[b])
          scores.append(score)
          catches.append(catch)
          last_score += 1
          if self.normalization == "batchnorm":
            gamma,betta = 'gamma'+str(layer_num),'betta'+str(layer_num)
            score,catch = batchnorm_forward(scores[last_score],self.params[gamma],
                           self.params[betta], self.bn_params[layer_num-1])
            scores.append(score)
            catches.append(catch)
            last_score +=1
          #add activation layer      
          score,catch = relu_forward(scores[last_score]) 
          scores.append(score)
          catches.append(catch)
          layer_num += 1
         

        # layer_num is the last layer
        #add last layer without relu activation function
        w,b = 'W' + str(layer_num),'b' + str(layer_num)
        last_score = len(scores)-1
        score,catch = affine_forward(scores[last_score], self.params[w],self.params[b])
        scores.append(score)
        catches.append(catch) 
        last_score += 1
       
        #add softmax score
        final_score = softmax_gate(scores[len(scores)-1])
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == "test":
            return final_score

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss,dx = softmax_loss(scores[len(scores)-1], y)
        L2_reg = 0
        
        for layer_num in range(self.num_layers):
          w = 'W' + str(layer_num+1)
          L2_reg +=np.sum(np.sum(self.params[w]*(self.params[w])))
        
        L2_reg *= 0.5 *self.reg
        loss += L2_reg
        #fill grades
        j = len(catches)-1#index for catches
        for i in range(self.num_layers,0,-1):
          w,b = 'W' + str(i),'b' + str(i)
          if(i == self.num_layers):
            #if were in the last layer there is no activation function
            dx,dw,db = affine_backward(dx,catches[j])
            grads[w] = dw + self.reg*self.params[w]
            grads[b] = db
            j -= 1
          else:
            dx = relu_backward(dx,catches[j])
            j -= 1
            #check if batch normalization is included
            if self.normalization == "batchnorm":
              gamma,betta = ('gamma'+str(i)),('betta'+str(i))
              dx,dgamma,dbetta = batchnorm_backward_alt(dx, catches[j])
              grads[gamma] = dgamma
              grads[betta] = dbetta
              j-=1
            dx,dw,db = affine_backward(dx,catches[j])
            grads[w] = dw + self.reg*self.params[w]
            grads[b] = db
            j -= 1
           


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
