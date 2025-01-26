import network1
import Game
import numpy as np    
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Hyperparameters
batchSize = 64
gamma = 0.8
epsMax = 1
eps = epsMax
epsMin = 0.1
# epsDecay = 0.001
epsDecay = 0.00001
targetUpdate = 1000
memorySize = 10000
totalEpisodes = 10000 
learningRate = 0.0001
learningRateDecay = 0.001
step = 0

# Initialize networks and memory
qNetPlayerO = network1.netWork()#.to("xpu")
targetNetO = network1.netWork()#.to("xpu")

qNetPlayerX = network1.netWork()#.to("xpu")
targetNetX = network1.netWork()#.to("xpu")

criteration = nn.MSELoss()#.to("xpu")
optimizerO = optim.SGD(qNetPlayerO.parameters(), lr = learningRate, momentum= 0.9)
optimizerX = optim.SGD(qNetPlayerX.parameters(), lr = learningRate, momentum= 0.9)

memoryO = deque(maxlen=memorySize)
memoryX = deque(maxlen=memorySize)

myGame = Game.game()

def train(qNet, targetNet, memory, optimizer):

    if len(memory) > batchSize:

        # Take sample batch based on reward
        #prioritizedMemory = sorted(memory, key=lambda x: x[2], reverse=True) # Sort by reward
        sampleBatch = random.sample(memory, batchSize)
        
        for state, action, reward, nextState, done in sampleBatch:

            predictedQs = qNet.forward(state)[0]
            targetQs = predictedQs.clone().detach()
            
            targetQs[action] = reward
            if not done:
                nextQs = targetNet.forward(nextState)
                maxElement, _ = torch.max(nextQs, 1)
                targetQs[action] += gamma * maxElement.item()

            optimizer.zero_grad()
            loss = criteration(predictedQs, targetQs)
            
            loss.backward()
            optimizer.step()

def getAction(qNet,actionSpace, currentState):
    if np.random.rand() <= eps:
        action = np.random.choice(actionSpace)  # Explore
    else:
        qValues = qNet.forward(currentState)
        _, action = torch.max(qValues,1)  # Exploit
    return action.item()

#debug function to show the board
def displayBoard(board):
    board = board.reshape(-1,1).tolist()
    for i in range(3):
        print(board[i*3], board[i*3+1], board[i*3+2])

for episode in range(totalEpisodes):
    
    #for debug purposes
    if(episode % 100 == 0):
        displayBoard(myGame.getBoard())
        print("Episode: ", episode)
        print("epsilon ", eps)
        print("learning Rate", learningRate)
        print()

    #reset game
    myGame.reset()

    #let player X do the first Move
    currentStateX = myGame.getBoard()
    actionSpace = myGame.getActionSpace(currentStateX)
    actionX = getAction(qNetPlayerX, actionSpace, currentStateX)
    myGame.playerXMove(actionX)

    #get reward of that move
    rewardX, doneX = myGame.getReward(actionX, actionSpace, "playerX")
    
    step += 1

    for i in range(8):

        #move player O
        currentStateO = myGame.getBoard()
        actionSpace = myGame.getActionSpace(currentStateO)
        actionO = getAction(qNetPlayerO, actionSpace, currentStateO)
        myGame.playerOMove(actionO)
        step += 1

        #get reward for this action
        rewardO, doneO = myGame.getReward(actionO, actionSpace, "playerO")

        #store experience X
        nextStateX = myGame.getBoard()
        memoryX.append((currentStateX, actionX, rewardX, nextStateX, doneX))

        if doneO:
            break

        #move player X
        currentStateX = myGame.getBoard()
        actionSpace = myGame.getActionSpace(currentStateX)
        actionX = getAction(qNetPlayerX, actionSpace, currentStateX)
        myGame.playerXMove(actionX)
        step += 1

        #get reward for this action
        rewardX, doneX = myGame.getReward(actionX, actionSpace, "playerX")

        #set nextState for O
        nextStateO = myGame.getBoard()

        #store experience for player O
        memoryO.append((currentStateO, actionO, rewardO, nextStateO, doneO))

        if doneX:
            break

    #if player x wins or draw
    if doneX:

        #store memory for x
        memoryX.append((currentStateX, actionX, rewardX, None, True))

        #store memory for O
        reward, done = myGame.getReward(actionO, actionSpace, "playerO")

        lastMemo = memoryO.pop()
        modifedMemo = lastMemo[0], lastMemo[1], reward, None,True
        
        memoryO.append(modifedMemo)
        
    else:
        #store memory for O
        memoryO.append((currentStateO, actionO, rewardO, None, True))

        #store lose memory for X
        reward, done = myGame.getReward(actionO, actionSpace, "playerX")

        lastMemo = memoryX.pop()
        modifedMemo = lastMemo[0], lastMemo[1], reward, None,True
        
        memoryX.append(modifedMemo)

    # print(memoryX)
    # print(memoryO)

    # train both models
    train(qNetPlayerO, targetNetO, memoryO, optimizerO)
    train(qNetPlayerX, targetNetX, memoryX, optimizerX) 
    
    #update targetNet every 1000 step 
    if step > targetUpdate:
        #update target network
        targetNetO.load_state_dict(qNetPlayerO.state_dict())
        targetNetX.load_state_dict(qNetPlayerX.state_dict())
        step = 0

    # epsilon decay
    if eps > epsMin:
        eps = epsMin + (eps - epsMin) * np.exp(-epsDecay * episode)
    # #learning rate decay
    # if learningRate > 0.00001:
    #     learningRate *= 1 / (1 + learningRateDecay * episode)

torch.save(qNetPlayerO.state_dict(), 'modelO.pth')
torch.save(qNetPlayerX.state_dict(), 'modelX.pth')
print("Training Complete")
