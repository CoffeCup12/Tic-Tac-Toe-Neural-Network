import pandas as pd
import numpy as np
import random

class fileReader():

    def __init__(self):
        self.input = []
        self.output = []

    def read(self):

        file = pd.read_csv('./dataFolder/tic-tac-toe.data', delimiter='\t') 
        dataList = file.values.tolist()
        #dataList = dataList[400: len(dataList)]
        random.shuffle(dataList)

        for game in dataList:
            game = game[0].split(',')

            board = np.zeros((9,1))
            output = np.zeros((2,1))

            for j in range(len(game)):
                if game[j] == 'x':
                    board[j] = 1
                elif game[j] == 'o':
                    board[j] = -1
                elif game[j] == 'positive':
                    output[0] = 1
                elif game[j] == 'negative':
                    output[1] = 1
            
            self.input.append(board)
            self.output.append(output)
            

    def getOutputTrain(self):
        outputTrain = self.output[0:int(len(self.output)/2)]
        return outputTrain
    
    def getInputTrain(self):
        inputTrain = self.input[0:int(len(self.input)/2)]
        return inputTrain
    
    def getOutputTest(self):
        outputTest = self.output[int(len(self.output)/2): len(self.output)]
        return outputTest
    
    def getInputTest(self):
        inputTest = self.input[int(len(self.input)/2): len(self.input)]
        return inputTest
    
    

                    
                
