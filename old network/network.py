import numpy as np
import json
import Layer 

class netWork:
    
    def __init__(self, name):
        self.inputLayer = Layer.inputLayer(27)
        self.h1 = Layer.hiddenLayer(18,27)
        self.h2 = Layer.hiddenLayer(18,18)
        self.outputLayer = Layer.outputLayer(9,18)
        self.name = name
    def getName(self):
        return self.name
    
    def processInput(self, state):
        #create new input layout 
        feed = np.zeros((27,1))

        #for each block, there exist 3 corresponding positons, the top is 1 if X occupied, 
        # mid is 1 is blank, bottom is 1 if O occupied 
        for i in range(9):

            top = i * 3
            mid = top + 1
            bot = top + 2

            if state[i] == 1:
               feed[top] = 1
            elif state[i] == 0.5:
                feed[bot] = 1
            elif state[i] == 0:
                feed[mid] = 1

        return feed

    
    def forwardCycle(self, inputRaw):
        #get processed input
        input = self.processInput(inputRaw)
        #reset input layer
        self.inputLayer.resetLayer(input)
        #forward pass
        self.inputLayer.forwardPass()
        self.h1.forwardPass(self.inputLayer)
        self.h2.forwardPass(self.h1)
        self.outputLayer.forwardPass(self.h2)
        #return output 
        return self.outputLayer.getLayer()
    
    def backpropagation(self, loss, learningRate):
        #store weight before backpropagation
        weightOut = self.outputLayer.getWeight()
        weighth2 = self.h2.getWeight()
        weighth1 = self.h1.getWeight()

        #backward pass
        self.outputLayer.backwardPass(self.h2, learningRate, loss)
        nextDelta = self.h2.backwardPass(self.h1, weightOut, loss, learningRate)
        nextDelta = self.h1.backwardPass(self.inputLayer, weighth2, nextDelta, learningRate)
        self.inputLayer.backwardPass(weighth1, nextDelta, learningRate)
    
    def storeModel(self, fileName):

        model = {
            "h1Weight" : self.h1.getWeight().tolist(),
            "h2Weight" : self.h2.getWeight().tolist(),
            "outputWeight" : self.outputLayer.getWeight().tolist(),
            "h1Bias" : self.h1.getBias().tolist(),
            "h2Bias" : self.h2.getBias().tolist(),
            "outputBias" : self.outputLayer.getBias().tolist(),
            "h1Gamma" : self.h1.getGamma().tolist(),
            "h2Gamma" : self.h2.getGamma().tolist(),
            "inputGamma" : self.inputLayer.getGamma().tolist(),
            "h1Beta" : self.h1.getBeta().tolist(),
            "h2Beta" : self.h2.getBeta().tolist(),
            "inputBeta" : self.inputLayer.getBeta().tolist(),
        }

        with open(fileName, 'w') as outfile:
            json.dump(model, outfile)

    def loadModel(self, fileName):

        with open(fileName, 'r') as infile:
            model = json.load(infile)
        
        self.h1.setWeight(np.array(model.get('h1Weight')))
        self.h2.setWeight(np.array(model.get('h2Weight')))
        self.outputLayer.setWeight(np.array(model.get('outputWeight')))

        self.h1.setBias(np.array(model.get('h1Bias')))
        self.h2.setBias(np.array(model.get('h2Bias')))
        self.outputLayer.setBias(np.array(model.get('outputBias')))

        self.h1.setGamma(np.array(model.get('h1Gamma')))
        self.h2.setGamma(np.array(model.get('h2Gamma')))
        self.inputLayer.setGamma(np.array(model.get('inputGamma')))

        self.h1.setBeta(np.array(model.get('h1Beta')))
        self.h2.setBeta(np.array(model.get('h2Beta')))
        self.inputLayer.setBeta(np.array(model.get('inputBeta')))
    
    def transferFrom(self, source):

        #transfer all weights and bias from source 
        self.h1.setWeight(source.h1.getWeight())
        self.h2.setWeight(source.h2.getWeight())
        self.outputLayer.setWeight(source.outputLayer.getWeight())

        self.h1.setBias(source.h1.getBias())
        self.h2.setBias(source.h2.getBias())
        self.outputLayer.setBias(source.outputLayer.getBias())

        self.inputLayer.setGamma(source.inputLayer.getGamma())
        self.h1.setGamma(source.h1.getGamma())
        self.h2.setGamma(source.h2.getGamma())
        

        


