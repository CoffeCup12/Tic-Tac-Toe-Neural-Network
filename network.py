import numpy as np
import json
import Layer 
import FileReader

class netWork:
    
    def __init__(self, name):
        self.inputLayer = Layer.inputLayer(9)
        self.h1 = Layer.hiddenLayer(45,9)
        self.h2 = Layer.hiddenLayer(45,45)
        self.h3 = Layer.hiddenLayer(27,45)
        self.outputLayer = Layer.outputLayer(9,27)
        self.name = name
    def getName(self):
        return self.name
    
    def forwardCycle(self, input):
        self.inputLayer.resetLayer(input)
        self.h1.forwardPass(self.inputLayer)
        self.h2.forwardPass(self.h1)
        self.h3.forwardPass(self.h2)
        self.outputLayer.forwardPass(self.h3)
        return self.outputLayer.getLayer()
    
    def backpropagation(self, loss, learningRate):
        self.outputLayer.backwardPass(self.h3, learningRate, loss)
        nextDelta = self.h3.backwardPass(self.h2, self.outputLayer, self.outputLayer.getDelta(loss),learningRate)
        nextDelta = self.h2.backwardPass(self.h1, self.h3, nextDelta, learningRate)
        self.h1.backwardPass(self.inputLayer, self.h2, nextDelta, learningRate)
    
    def storeModel(self, fileName):

        model = {
            "h1Weight" : self.h1.getWeight().tolist(),
            "h2Weight" : self.h2.getWeight().tolist(),
            "h3Weight" : self.h3.getWeight().tolist(),
            "outputWeight" : self.outputLayer.getWeight().tolist(),
            "h1Bias" : self.h1.getBias().tolist(),
            "h2Bias" : self.h2.getBias().tolist(),
            "h3Bias" : self.h3.getBias().tolist(),
            "outputBias" : self.outputLayer.getBias().tolist(),
        }

        with open(fileName, 'w') as outfile:
            json.dump(model, outfile)

    def loadModel(self, fileName):

        with open(fileName, 'r') as infile:
            model = json.load(infile)
        
        self.h1.setWeight(np.array(model.get('h1Weight')))
        self.h2.setWeight(np.array(model.get('h2Weight')))
        self.h3.setWeight(np.array(model.get('h3Weight')))
        self.outputLayer.setWeight(np.array(model.get('outputWeight')))

        self.h1.setBias(np.array(model.get('h1Bias')))
        self.h2.setBias(np.array(model.get('h2Bias')))
        self.h3.setBias(np.array(model.get('h3Bias')))
        self.outputLayer.setBias(np.array(model.get('outputBias')))
    
    def transferFrom(self, source):
        self.h1.setWeight(source.h1.getWeight().copy())
        self.h2.setWeight(source.h2.getWeight().copy())
        self.h3.setWeight(source.h3.getWeight().copy())
        self.outputLayer.setWeight(source.outputLayer.getWeight().copy())

        self.h1.setBias(source.h1.getBias().copy())
        self.h2.setBias(source.h2.getBias().copy())
        self.h3.setBias(source.h3.getBias().copy())
        self.outputLayer.setBias(source.outputLayer.getBias().copy())

        


