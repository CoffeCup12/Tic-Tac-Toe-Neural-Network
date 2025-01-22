import numpy as np

class layer():

    def __init__(self, numNodes):
        self.layer = np.zeros(numNodes).reshape(-1,1)
    
    def resetLayer(self, layer):
        self.layer = layer

    def getLayer(self):
        return self.layer

    #debug remove later
    def dispLayer(self):
        print(self.layer)

class inputLayer(layer):

    def __init__(self, numNodes):
        super().__init__(numNodes)  
    
class hiddenLayer(layer):

    def __init__(self, numNodes, numNodesLast):

        super().__init__(numNodes)

        #generate random weights and bias
        self.weights = np.random.randn(numNodes, numNodesLast) 
        self.bias = np.random.randn(numNodes, 1)
        self.originalOutput = np.zeros((numNodes, 1))
    
    def forwardPass(self, inputLayer):  
        #self.layer = wx + b
        self.layer = self.weights.dot(inputLayer.getLayer()) + self.bias
        self.originalOutput = self.layer.copy()
        #leaky RELU activation function
        self.layer[self.layer < 0] *= 0.01

    # Other methods remain unchanged...

    def backwardPass(self, lastLayer, nextLayer, nextDelta, learningRate):
        # Get delta
        delta = self.getDelta(nextLayer, nextDelta)
        
        # Calculate gradients
        gradW = np.dot(delta, np.transpose(lastLayer.getLayer()))
        gradB = delta

        # Clip gradients to avoid exploding gradients
        max_grad = 1.0 
        gradW = np.clip(gradW, -max_grad, max_grad) 
        gradB = np.clip(gradB, -max_grad, max_grad)

        # Update weights and bias using gradient descent
        self.weights -= learningRate * gradW
        self.bias -= learningRate * gradB

        # Return the delta for the backward pass
        return delta
    
    def getDelta(self, nextWeight, nextDelta):
        # Calculate delta for the current layer
        return np.dot(np.transpose(nextWeight), nextDelta) * self.derRelu(self.originalOutput)
    
    def derRelu(self, x):
        # Derivative of Leaky ReLU activation function
        return np.where(x > 0, 1, 0.01)
    
    def getWeight(self):
        return self.weights.copy()
    
    def setWeight(self, weight):
        self.weights = weight
    
    def getBias(self):
        return self.bias.copy()
    
    def setBias(self, bias):
        self.bias = bias

class outputLayer(hiddenLayer):
    
    def __init__(self, numNodes, numNodesLast):
        super().__init__(numNodes, numNodesLast)
    
    # def softmax(self, inputLayer):
    #     return (np.e ** inputLayer) / np.sum(inputLayer)
    
    def forwardPass(self, inputLayer):
        #self.layer = wx + b
        self.layer = self.weights.dot(inputLayer.getLayer()) + self.bias

    def backwardPass(self, lastLayer, learningRate, loss):
        # Get delta from the loss (TD error)
        delta = self.getDelta(loss)

        # Calculate gradients
        gradW = np.dot(delta, np.transpose(lastLayer.getLayer()))
        gradB = delta

        # Clip gradients to avoid exploding gradients
        max_grad = 1.0 
        gradW = np.clip(gradW, -max_grad, max_grad) 
        gradB = np.clip(gradB, -max_grad, max_grad)
        
        # Update weights and bias using gradient descent
        self.weights -= learningRate * gradW
        self.bias -= learningRate * gradB
    
    def getDelta(self, loss):
        # Delta for the output layer in DQN is the TD error
        return loss


        
