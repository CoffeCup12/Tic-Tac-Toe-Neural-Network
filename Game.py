import network
import numpy as np
import random
import time

class game():
    def __init__(self):
        self.playerO = []
        self.playerX = []
        self.board = np.zeros((9,1))
        self.count = 0
        self.winState = [{0,1,2}, {3,4,5}, {6,7,8}, {0,4,8}, {2,4,6}, {0,3,6}, {1,4,7}, {2,5,8}]

    def playerOMove(self, input):
        if len(self.playerO) == 0 or input != self.playerO[-1]:
            self.playerO.append(input)
            self.board[input] = -1
            self.count += 1

    def playerXMove(self, input):
        if len(self.playerX) == 0 or input != self.playerX[-1]:
            self.playerX.append(input)
            self.board[input] = 1
            self.count += 1

    def getXActionList(self):
        return self.playerX

    def getOActionList(self):
        return self.playerO
    
    def checkWin(self, side):
        handSet = set(side)
        return any(win.issubset(handSet) for win in self.winState)
    
    def isDone(self):
        return self.checkWin(self.playerO) or self.checkWin(self.playerX) or self.count >= 9

    def getReward(self, action, player):

        if player == "playerO":
            receiver = self.playerO
            opponent = self.playerX
        else:
            receiver = self.playerX
            opponent = self.playerO

        reward = 0

        if self.checkWin(receiver):
            reward = 1  # Maximum reward for winning
        elif self.checkWin(opponent):
            reward = -1  # Minimum reward for losing
        elif self.count >= 9:
            reward = 0.8  # Reward for a draw
        else:

            # Intermediate Rewards
            # reward += sum(0.05 for win in self.winState if win & set(receiver) and not win & set(opponent))  # Reward for potential winning moves
            # reward -= sum(0.05 for win in self.winState if win & set(opponent) and not win & set(receiver))  # Penalty for opponent's potential winning moves
           # Encourage center and corner moves (optional)
            if action in [4]:  # Center position
                reward += 0.2
            # Check for blocking opponent's potential winning move
            action = int(action)
            opponentHand = set(opponent)
            receiverHand = set(receiver)

            for win in self.winState:
                
                if len(win & opponentHand) == 2  and action in win:
                    reward += 0.5
                    
                elif len(win & opponentHand) == 2 and len(win & receiverHand) == 0:
                    reward -= 0.5
                
        #reward = np.clip(reward, -1, 1)

        return reward
    
    def getActionSpace(self, state):
        if not self.isDone():
            spaceList = np.where(state == 0)[0].tolist()
        else:
            spaceList = []
        return spaceList
    
    def getPredictAction(self, space, prediction):
        validPredictions = prediction[space]  # Get predictions for valid actions
        maxElement = np.max(validPredictions)  # Find the maximum value among valid predictions
        max_indices = [i for i in space if prediction[i] == maxElement]  # Get all indices with max value

        # Randomly select among the best actions if there's a tie
        return random.choice(max_indices)

    
    def getBoard(self):
        return self.board

    def reset(self):
        self.board = np.zeros((9,1))
        self.playerO = []
        self.playerX = []
        self.count = 0

class backend(game):
    def __init__(self):
        super().__init__()
        self.model = network.netWork('modelO')
        self.model.loadModel('modelO.json')
                                                                                      
    def oneRound(self, move):
        if self.count < 9 and not self.checkWin(self.playerO) and not self.checkWin(self.playerX):
            self.playerXMove(move)
            if(self.checkWin(self.playerX)):
                return "You Win"
            else:
                #time.sleep(0.3)
                #if the player didn't win, let computer move
                print(self.board)
                #print(self.model.forwardCycle(self.board))
                space = self.getActionSpace(self.board)
                print(space)
                print(self.model.forwardCycle(self.board))
                computerMove = self.getPredictAction(space, self.model.forwardCycle(self.board))
                #computerMove = np.argmax(self.model.forwardCycle(self.board))
                self.playerOMove(computerMove)
                print(int(computerMove))
                return int(computerMove)
        
    