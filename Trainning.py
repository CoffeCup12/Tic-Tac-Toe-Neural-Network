import network
import Game
import numpy as np    
import random
from collections import deque

# Hyperparameters
batchSize = 128
gamma = 0.8
epsMax = 1
eps = epsMax
epsMin = 0.1
# epsDecay = 0.001
epsDecay = 0.000001
targetUpdate = 1000
memorySize = 10000
totalEpisodes = 10000 
learningRate = 0.000025
learningRateDecay = 0.001
step = 0

# Initialize networks and memory
qNetPlayerO = network.netWork("playerO")
targetNetO = network.netWork("targetO")

qNetPlayerX = network.netWork("playerX")
targetNetX = network.netWork("targetX")

memoryO = deque(maxlen=memorySize)
memoryX = deque(maxlen=memorySize)

myGame = Game.game()

def train(qNet, targetNet, memory, learningRate):

    if len(memory) > batchSize:

        # Take sample batch based on reward
        prioritizedMemory = sorted(memory, key=lambda x: x[2], reverse=True) # Sort by reward
        sampleBatch = random.sample(prioritizedMemory, batchSize)
        
        for state, action, reward, nextState, done in sampleBatch:

            predictedQs = qNet.forwardCycle(state)
            targetQs = predictedQs.copy()
            
            targetQs[action] = reward
            if not done:
                nextQs = targetNet.forwardCycle(nextState)
                targetQs[action] += gamma * np.max(nextQs)

            loss = targetQs - predictedQs
            qNet.backpropagation(loss, learningRate)
         

def getAction(qNet,actionSpace, currentState):
    if np.random.rand() <= eps:
        action = np.random.choice(actionSpace)  # Explore
    else:
        qValues = qNet.forwardCycle(currentState)
        action = myGame.getPredictAction(actionSpace, qValues)  # Exploit
        #action = np.argmax(qValues)
        #print(qValues)
    return action

#debug function to show the board
def displayBoard(board):
    board = board.reshape(-1,1).tolist()
    for i in range(3):
        print(board[i*3], board[i*3+1], board[i*3+2])

for episode in range(totalEpisodes):
    
    #for debug purposes
    if(episode % 10 == 0):
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

    #get reward for this action
    rewardX, doneX = myGame.getReward(actionX, "playerX")
    doneO = False

    for i in range(8):

        #move player O
        currentStateO = myGame.getBoard().copy()
        actionSpace = myGame.getActionSpace(currentStateO)
        actionO = getAction(qNetPlayerO, actionSpace, currentStateO)
        myGame.playerOMove(actionO)

        # if episode % 100 == 0:
        #     displayBoard(myGame.getBoard())
        #     print()

        #get reward for this action
        rewardO, doneO = myGame.getReward(actionO, "playerO")

        #set nextState for X 
        nextStateX = myGame.getBoard()

        #store experience for player X 
        experience = currentStateX, actionX, rewardX, nextStateX, doneX
        memoryX.append(experience)

        if doneO:
            break

        #move player X
        currentStateX = myGame.getBoard()
        actionSpace = myGame.getActionSpace(currentStateX)
        actionX = getAction(qNetPlayerX, actionSpace, currentStateX)
        myGame.playerXMove(actionX)
        # if episode % 100 == 0:
        #     displayBoard(myGame.getBoard())
        #     print()

        #get reward for this action
        rewardX, doneX = myGame.getReward(actionX, "playerX")

        #set nextState for O
        nextStateO = myGame.getBoard()

        #store experience for player O
        experience = currentStateO, actionO, rewardO, nextStateO, doneO
        memoryO.append(experience)

        if doneX:
            break

        # #train both models
        # train(qNetPlayerX, targetNetX, memoryX, learningRate)
        # train(qNetPlayerO, targetNetO, memoryO, learningRate) 

        step += 2

    #if player x wins or draw
    if doneX:
        #store win memeory for X 
        reward, done = myGame.getReward(actionX, "playerX")
        memoryX.append((currentStateX, actionX, reward, None, True))

        #store lose memory for O
        reward, done = myGame.getReward(actionO, "playerO")

        lastMemo = memoryO.pop()
        modifedMemo = currentStateO, actionO, reward, None,True
        
        memoryO.append(modifedMemo)
        
    else:
        #store win memeory for O 
        reward, done = myGame.getReward(actionX, "playerO")
        memoryO.append((currentStateO, actionO, reward, None, True))

        #store lose memory for X
        reward, done = myGame.getReward(actionX, "playerX")

        lastMemo = memoryX.pop()
        modifedMemo = currentStateX, actionX, reward, None, True

        memoryX.append(modifedMemo)
    # print(memoryX)
    # print(memoryO)

    # for state, action, reward, nextState, done in memoryO:
    #     displayBoard(state)
    #     print(action)
    #     print(reward)
    #     if nextState is not None:
    #         displayBoard(nextState)
    #     else:
    #         print("None")
    #     print(done)
    #     print()

    # train both models
    train(qNetPlayerX, targetNetX, memoryX, learningRate)
    train(qNetPlayerO, targetNetO, memoryO, learningRate) 
    
    #update targetNet every 1000 step 
    if step > targetUpdate:
        #update target network
        targetNetO.transferFrom(qNetPlayerO)
        targetNetX.transferFrom(qNetPlayerX)
        
        step = 0

    # epsilon decay
    if eps > epsMin:
        eps = epsMin + (eps - epsMin) * np.exp(-epsDecay * episode)
    # #learning rate decay
    # if learningRate > 0.00001:
    #     learningRate *= 1 / (1 + learningRateDecay * episode)

qNetPlayerO.storeModel("ModelO.json")
qNetPlayerX.storeModel("ModelX.json")
print("Training Complete")
