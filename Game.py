import network1
import numpy as np
import random
import time
import torch

class game():
    def __init__(self):
        self.playerO = []
        self.playerX = []
        self.board = np.zeros(9)
        self.count = 0
        self.winState = [{0,1,2}, {3,4,5}, {6,7,8}, {0,4,8}, {2,4,6}, {0,3,6}, {1,4,7}, {2,5,8}]

    def playerOMove(self, input):
        self.playerO.append(input)
        self.board[input] = 0.5
        self.count += 1

    def playerXMove(self, input):
        self.playerX.append(input)
        self.board[input] = 1
        self.count += 1
    
    def checkWin(self, side):
        handSet = set(side)
        return any(win.issubset(handSet) for win in self.winState)

    def getReward(self, action, actionSpace, player):

        done = True

        if player == "playerX":
            receiver = self.playerX
            opponent = self.playerO
        else:
            receiver = self.playerO
            opponent = self.playerX
        
        if action not in actionSpace:
            reward = -30  
        else:
            reward = 3
            # winning
            if self.checkWin(receiver):
                reward = 10
            #losing 
            elif self.checkWin(opponent):
                reward = -10
            # Draw
            elif self.count >= 9:
                reward = 6 
            else:
                done = False

                # Check for blocking opponent's potential winning move
                action = int(action)
                opponentHand = set(opponent)
                receiverHand = set(receiver)

                for win in self.winState:
                    if len(win & opponentHand) == 2 and action in win:
                        reward += 7
                    elif len(win & opponentHand) == 2 and len(win & receiverHand) == 0:
                        reward -= 5  
                    # elif len(win & receiverHand) == 2 and len(win & opponentHand) == 0:
                    #     reward += 0.5  # Increase reward for going for the win

        return reward, done


    
    def getActionSpace(self, state):
        spaceList = np.where(state == 0)[0].tolist()
        return spaceList
    
    def getPredictAction(self, space, prediction):
        validPredictions = prediction[space]  # Get predictions for valid actions
        maxElement = np.max(validPredictions)  # Find the maximum value among valid predictions
        max_indices = [i for i in space if prediction[i] == maxElement]  # Get all indices with max value

        # Randomly select among the best actions if there's a tie
        return random.choice(max_indices)

    
    def getBoard(self):
        return self.board.copy()

    def reset(self):
        self.board = np.zeros((9,1))
        self.playerO = []
        self.playerX = []
        self.count = 0

class backend(game):
    def __init__(self):
        super().__init__()
        self.model = network1.netWork()
        self.model.load_state_dict(torch.load('modelO.pth'))
        self.model.eval()
                                                                                      
    def oneRound(self, move):
        if self.count < 9 and not self.checkWin(self.playerO) and not self.checkWin(self.playerX):
            self.playerXMove(move)
            if(self.checkWin(self.playerX)):
                return "You Win"
            else:
                #time.sleep(0.3)
                #if the player didn't win, let computer move
                print(self.board)
                print(self.model.forward(self.board))

                qValues = self.model.forward(self.board)
                _, computerMove = torch.max(qValues, 1)
                computerMove = computerMove.item()
                self.playerOMove(computerMove)

                print(int(computerMove))
                return int(computerMove)
        
    