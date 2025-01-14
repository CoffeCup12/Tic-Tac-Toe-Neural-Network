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
    
    def forwadPass(self, inputLayer):  
        #self.layer = wx + b
        self.layer = self.weights.dot(inputLayer.getLayer()) + self.bias
        #RELU activation function
        self.layer[self.layer < 0] = 0

    def backwardPass(self, lastLayer, nextLayer, nextDelta, learningRate):

        #get delta
        delta = self.getDelta(nextLayer, nextDelta, lastLayer)
        #dW = delta dot x^T
        sigmaWeight = np.dot(delta, np.transpose(lastLayer.getLayer()))
        #dB = delta
        sigmaBias = delta

        #update weights and bias
        self.weights = self.weights - (learningRate * sigmaWeight)
        self.bias = self.bias - (learningRate * sigmaBias)

        #return the delta calculated in this backward pass
        return delta
    
    def getDelta(self, nextlayer, nextDelta, lastLayer):
        #delta = (nextLayer's weight)^T dot (nextLayer's delta) * derivative of sigmoid 
        return np.dot(np.transpose(nextlayer.getWeight()), nextDelta) * self.derReLu(lastLayer.getLayer())
    
    def derReLu(self, x):
        newLayer = np.copy(x)
        newLayer[newLayer > 0] = 1
        newLayer[newLayer < 0] = 0
        return newLayer
    
    
    
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
    
    def forwadPass(self, inputLayer):
        #self.layer = wx + b
        self.layer = self.weights.dot(inputLayer.getLayer()) + self.bias
        self.layer = 1/(1 + np.e ** (-1*self.layer))
    
    def backwardPass(self, lastLayer, learningRate, actual):

        delta = self.getDelta(actual)

        sigmaWeight = np.dot(delta, np.transpose(lastLayer.getLayer()))
        sigmaBias = delta
        
        #update weights and bias 
        self.weights = self.weights - (learningRate * sigmaWeight)
        self.bias = self.bias - (learningRate * sigmaBias)
    
    def getDelta(self, actual):

        #delta = (finalOutput - actualResult) * derivative of sigmoid 
        return (self.layer - actual) * self.derSigmoid()
    def derSigmoid(self):
        #z = wx + b
        #derivatiev of sigmoid = sigmoid(z) * (1-sigmoid(z))
        return self.layer * (1 - self.layer)
        
