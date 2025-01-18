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
        #self.weights = np.random.randn(numNodes, numNodesLast)
        # He initialization for weights 
        self.weights = np.random.randn(numNodes, numNodesLast) * np.sqrt(2 / numNodesLast)
        self.bias = np.random.randn(numNodes, 1)
    
    def forwardPass(self, inputLayer):  
        #self.layer = wx + b
        self.layer = self.weights.dot(inputLayer.getLayer()) + self.bias
        #leaky RELU activation function
        self.layer[self.layer < 0] *= 0.01

    def backwardPass(self, lastLayer, nextLayer, nextDelta, learningRate):

        #get delta
        delta = self.getDelta(nextLayer, nextDelta, lastLayer)
        #dW = delta dot x^T
        sigmaWeight = np.dot(delta, np.transpose(lastLayer.getLayer()))
        #dB = delta
        sigmaBias = delta

        #clip gradient
        max_grad = 1.0 
        sigmaWeight = np.clip(sigmaWeight, -max_grad, max_grad) 
        sigmaBias = np.clip(sigmaBias, -max_grad, max_grad)

        #update weights and bias
        self.weights = self.weights - (learningRate * sigmaWeight)
        self.bias = self.bias - (learningRate * sigmaBias)

        #return the delta calculated in this backward pass
        return delta
    
    def getDelta(self, nextlayer, nextDelta, lastLayer):
        #delta = (nextLayer's weight)^T dot (nextLayer's delta) * derivative of RELU 
        return np.dot(np.transpose(nextlayer.getWeight()), nextDelta) * self.derRelu(self.layer)
    
    def derRelu(self, x):
        return np.where(x > 0, 1, 0.01)
    
    
    def getWeight(self):
        return self.weights
    
    def setWeight(self, weight):
        self.weights = weight
    
    def getBias(self):
        return self.bias
    
    def setBias(self, bias):
        self.bias = bias

class outputLayer(hiddenLayer):
    
    def __init__(self, numNodes, numNodesLast):
        super().__init__(numNodes, numNodesLast)
    
    def softmax(self, inputLayer):
        return (np.e ** inputLayer)/np.sum(inputLayer)
    
    def forwardPass(self, inputLayer):
        #self.layer = wx + b
        self.layer = self.weights.dot(inputLayer.getLayer()) + self.bias
        #self.layer = 1/(1 + np.exp(-self.layer))
    
    def backwardPass(self, lastLayer, learningRate, loss):

        delta = self.getDelta(loss)

        sigmaWeight = np.dot(delta, np.transpose(lastLayer.getLayer()))
        sigmaBias = delta

        #clip gradient
        max_grad = 1.0 
        sigmaWeight = np.clip(sigmaWeight, -max_grad, max_grad) 
        sigmaBias = np.clip(sigmaBias, -max_grad, max_grad)
        
        #update weights and bias 
        self.weights = self.weights - (learningRate * sigmaWeight)
        self.bias = self.bias - (learningRate * sigmaBias)
    
    def getDelta(self, loss):

        #delta = (finalOutput - actualResult) * derivative of sigmoid 
        return loss * self.layer * (1 - self.layer)
        
