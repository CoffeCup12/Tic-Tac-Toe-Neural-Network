import numpy as np
import json
import Layer 
import FileReader

class netWork:
    
    def __init__(self):
        self.inputLayer = Layer.inputLayer(9)
        self.h1 = Layer.hiddenLayer(16,9)
        self.h2 = Layer.hiddenLayer(16,16)
        self.h3 = Layer.hiddenLayer(16,16)
        self.outputLayer = Layer.outputLayer(2,16)

        self.reader = FileReader.fileReader()
        self.reader.read()
    
    def forwardCycle(self, input):
        self.inputLayer.resetLayer(input)
        self.h1.forwadPass(self.inputLayer)
        self.h2.forwadPass(self.h1)
        self.h3.forwadPass(self.h2)
        self.outputLayer.forwadPass(self.h3)
        return self.outputLayer.getLayer()
    
    def backprobagation(self, actual):
        learningRate = 0.1

        self.outputLayer.backwardPass(self.h1, learningRate, actual)
        nextDelta = self.h3.backwardPass(self.h2, self.outputLayer, self.outputLayer.getDelta(actual),learningRate)
        nextDelta = self.h2.backwardPass(self.h1, self.h3, nextDelta, learningRate)
        self.h1.backwardPass(self.inputLayer, self.h2, nextDelta, learningRate)

    def lossfunction(self, actual, result):
        return -np.sum(actual * np.log(result) + (1-actual) * np.log(1-result))/np.size(actual)  
    
    def train(self):

        feed = self.reader.getInputTrain()
        actual = self.reader.getOutputTrain()
        
        for i in range(len(feed)):
            for j in range(20):
                self.forwardCycle(feed[i])
                self.backprobagation(actual[i])

    def test(self):
        feed = self.reader.getInputTest()

        actual = self.reader.getOutputTest()
        
        count = 0
        for i in range(len(feed)):
            result = self.forwardCycle(feed[i])
            #if self.lossfunction(actual[i], self.outputLayer.getLayer()) > 1:
            if abs(result[0] - actual[i][0]) > 0.5:
            #     print(self.outputLayer.getLayer())
            #     print(actual[i])
            #     print()
                 count += 1
        return count/len(feed)
    
    def storeModel(self):

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

        with open('model.json', 'w') as outfile:
            json.dump(model, outfile)

    def loadModel(self):

        with open('model.json', 'r') as infile:
            model = json.load(infile)
        
        self.h1.setWeight(np.array(model.get('h1Weight')))
        self.h2.setWeight(np.array(model.get('h2Weight')))
        self.h3.setWeight(np.array(model.get('h3Weight')))
        self.outputLayer.setWeight(np.array(model.get('outputWeight')))

        self.h1.setBias(np.array(model.get('h1Bias')))
        self.h2.setBias(np.array(model.get('h2Bias')))
        self.h3.setBias(np.array(model.get('h3Bias')))
        self.outputLayer.setBias(np.array(model.get('outputBias')))
        
     
    

if __name__ == "__main__":

    net = netWork()

    action = input("1:trainning, 2:testing: ")

    if action == "1":
        rate = 0
        for i in range(6):
            net.train()
            print(net.test())
    
        store = input("store model: ")
        if store == "y":
            net.storeModel()

    elif action == "2":
        net.loadModel()
        net.forwardCycle(np.array([1,1,1,1,-1,-1,-1,1,-1]).reshape(-1,1))
        print(net.outputLayer.getLayer())

