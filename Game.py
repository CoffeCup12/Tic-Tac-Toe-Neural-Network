import network
import numpy as np
import time

class game():
    def __init__(self):
        self.net = network.netWork()
        self.net.loadModel()

        self.board = np.zeros((9,1))

        self.playerHand = []
        self.computerhand = []
        self.count = 0

    def playerMove(self, input):
        if self.board[input] == 0:
            self.board[input] = 1
            self.playerHand.append(input)
            
    def computerMove(self):
        pass
        posInput = []
        for i in range(len(self.board)):
            if self.board[i] == 0:
                newBoard = self.board.copy()
                newBoard[i] = -1
                posInput.append(newBoard)
        
        winRates = []
        for move in posInput:
            winRates.append(self.net.forwardCycle(move)[1])
            print(self.net.forwardCycle(move))
        
        finalChoice = posInput[winRates.index(max(winRates))]

        self.computerhand.append(np.where((finalChoice - self.board) == -1)[0][0])
        self.board = finalChoice
    
    def checkWin(self, side):
        winState = [{0,1,2}, {3,4,5}, {6,7,8}, {0,4,8}, {2,4,6}, {0,3,6}, {1,4,7}, {2,5,8}]
        handSet = set(side)
        i = 0
        while(i < len(winState) and not winState[i].issubset(handSet) and winState[i] != handSet ):
              i += 1
        return i < len(winState)
    
            
    def oneRound(self, move):

        #check if either player or computer wins 
        if(self.count < 9 and not self.checkWin(self.playerHand) and not self.checkWin(self.computerhand)): 
            #implements players movement
            self.playerMove(move)
            #check if the player wins after this movement 
            if(self.checkWin(self.playerHand)):
                return "You Win"
            else:
                time.sleep(0.3)
                #if the player didn't win, let computer move 
                self.computerMove()
                self.count += 2
                return int(self.computerhand[len(self.computerhand)-1]) 
        else:

            #display all possible results if either of player or computer wins 
            if self.checkWin(self.playerHand):
                return "You Win"
            elif self.checkWin(self.computerhand):
                return "Computer Win"
            else:
                return "Draw"

    def reset(self):
        self.board = np.zeros((9,1))
        self.playerHand = []
        self.computerhand = []
        self.count = 0
    