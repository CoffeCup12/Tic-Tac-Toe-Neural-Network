import numpy as np

class layer():

    def __init__(self, numNodes):
        self.layer = np.zeros(numNodes).reshape(-1,1)
    
    def resetLayer(self, layer):
        self.layer = layer

    def getLayer(self):
        return self.layer.copy()

class inputLayer(layer):

    def __init__(self, numNodes):
        super().__init__(numNodes)
        self.normalizationLayer = batchNormalizationLayer(numNodes)
    def forwardPass(self):
        self.layer = self.normalizationLayer.forwardPass(self.layer)
    
    def backwardPass(self, nextWeight, nextDelta, learningRate):
        delta = self.getDelta(nextWeight, nextDelta)
        self.normalizationLayer.backwardPass(delta, learningRate)
    
    def getDelta(self, nextWeight, nextDelta):
        # Calculate delta for the current layer
        return np.dot(np.transpose(nextWeight), nextDelta)
    
    def getGamma(self):
        return self.normalizationLayer.getGamma()
    def getBeta(self):
        return self.normalizationLayer.getBeta()
    
    def setGamma(self, gamma):
        self.normalizationLayer.setGamma(gamma)
    def setBeta(self, beta):
        self.normalizationLayer.setBeta(beta)
    
class hiddenLayer(layer):

    def __init__(self, numNodes, numNodesLast):

        super().__init__(numNodes)

        #generate random weights and bias
        self.weights = np.random.randn(numNodes, numNodesLast)
        self.bias = np.random.randn(numNodes, 1)
        self.originalOutput = np.zeros((numNodes, 1))
        self.normalizationLayer = batchNormalizationLayer(numNodes)
    
    def forwardPass(self, inputLayer):  
        #self.layer = wx + b
        self.layer = self.weights.dot(inputLayer.getLayer()) + self.bias
        self.originalOutput = self.layer.copy()
        #normalize
        self.layer = self.normalizationLayer.forwardPass(self.layer)
        #leaky RELU activation function
        self.layer[self.layer < 0] *= 0.01

    def clippingGradient(self, gradW, gradB):
        # Clip gradients to avoid exploding gradients
        max_grad = 1 
        gradW = np.clip(gradW, -max_grad, max_grad) 
        gradB = np.clip(gradB, -max_grad, max_grad)

        return gradW, gradB

    def backwardPass(self, lastLayer, nextLayer, nextDelta, learningRate):
        # Get delta
        delta = self.getDelta(nextLayer, nextDelta)
        delta = self.normalizationLayer.backwardPass(delta, learningRate)

        # Calculate gradients
        gradW = np.dot(delta, np.transpose(lastLayer.getLayer()))
        gradB = delta

        # Clip gradients to avoid exploding gradients
        gradW, gradB = self.clippingGradient(gradW, gradB)

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

    def getGamma(self):
        return self.normalizationLayer.getGamma()
    def getBeta(self):
        return self.normalizationLayer.getBeta()
    
    def setGamma(self, gamma):
        self.normalizationLayer.setGamma(gamma)
    def setBeta(self, beta):
        self.normalizationLayer.setBeta(beta)

class batchNormalizationLayer(layer):
    def __init__(self, numNodes):
        super().__init__(numNodes)
        self.scalingFactor = np.full((numNodes, 1),1.0)
        self.shiftingFactor = np.zeros((numNodes, 1))
        self.variance = 0
        self.epsilon = 1e-8
    
    def forwardPass(self, input):
        normalizedVector = self.normalize(input)
        self.layer = normalizedVector * self.scalingFactor + self.shiftingFactor
        return self.layer.copy()

    def normalize(self, input):
        averageVector = np.full((input.shape[0], 1), np.mean(input))
        self.variance = np.mean((input - averageVector)**2)
        return (input - averageVector) / np.sqrt(self.variance + self.epsilon)
    
    def backwardPass(self, loss, learningRate):
        delta = self.scalingFactor * loss / np.sqrt(self.variance + self.epsilon)

        # Gradients for scaling factor (gamma) and shifting factor (beta)
        gradGamma = np.sum(loss * self.layer, axis=0, keepdims=True)
        gradBeta = np.sum(loss, axis=0, keepdims=True)

        #gradGamma, gradBeta = self.clippingGradient(gradGamma, gradBeta)
        
        # Update scaling and shifting factors
        self.scalingFactor -= learningRate * gradGamma
        self.shiftingFactor -= learningRate * gradBeta

        return delta
    
    def clippingGradient(self, gradW, gradB):
        # Clip gradients to avoid exploding gradients
        max_grad = 0.0 
        gradW = np.clip(gradW, -max_grad, max_grad) 
        gradB = np.clip(gradB, -max_grad, max_grad)

        return gradW, gradB
    
    def setGamma(self, gamma):
        self.scalingFactor = gamma

    def setBeta(self, beta):
        self.shiftingFactor = beta

    def getGamma(self):
        return self.scalingFactor
        
    def getBeta(self):
        return self.shiftingFactor


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
        gradW, gradB = self.clippingGradient(gradW, gradB)
        
        # Update weights and bias using gradient descent
        self.weights -= learningRate * gradW
        self.bias -= learningRate * gradB
    
    def getDelta(self, loss):
        # delta = loss * derivative of tanh
        return loss


        
