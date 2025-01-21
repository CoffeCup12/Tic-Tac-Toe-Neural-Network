import network
import Game
import numpy as np    
import random
from collections import deque

# Hyperparameters
batchSize = 128
gamma = 0.99
epsMax = 1
eps = epsMax
epsMin = 0.1
# epsDecay = 0.001
epsDecay = 0.0005
targetUpdate = 100
memorySize = 10000
totalEpisodes = 10000 
learningRate = 0.0005
learningRateDecay = 0.001

# Initialize networks and memory
qNetPlayerO = network.netWork("playerO")
targetNetO = network.netWork("targetO")

qNetPlayerX = network.netWork("playerX")
targetNetX = network.netWork("targetX")

memoryO = deque(maxlen=memorySize)
memoryX = deque(maxlen=memorySize)

myGame = Game.game()

# def prioritizeExperience(memory, batchSize):
#     prioritizedMemory = sorted(memory, key=lambda x: x[2], reverse=True) # Sort by reward 
#     sample = random.sample(prioritizedMemory, batchSize)
#     return sample

def getRidOfInvalidActions(qValues, actionSpace):
    mask = np.zeros_like(qValues, dtype=bool)
    mask[actionSpace] = True
    qValues[~mask] = 0


def train(qNet, targetNet, memory, learningRate):
    if len(memory) > batchSize:

        # Take sample batch based on reward
        prioritizedMemory = sorted(memory, key=lambda x: x[2], reverse=True) # Sort by reward
        sampleBatch = random.sample(prioritizedMemory, batchSize)
        
        for state, action, reward, nextState, done in sampleBatch:

            actualQs = qNet.forwardCycle(state)
            actionSpace = myGame.getActionSpace(state)
            
            # Get rid of invalid actions in the current state's Q-values
            getRidOfInvalidActions(actualQs, actionSpace)
            vs = np.max(actualQs)

            if done:
                vsPrime = 0
            else:
                nextQs = targetNet.forwardCycle(nextState)
                nextActionSpace = myGame.getActionSpace(nextState)
                
                # Get rid of invalid actions in the next state's Q-values
                getRidOfInvalidActions(nextQs, nextActionSpace)
                vsPrime = np.max(nextQs)

            target = reward + gamma * vsPrime

            # Compute the loss for the specific action
            lossVector = np.zeros_like(actualQs)
            lossVector[action] = target - actualQs[action]

            qNet.backpropagation(lossVector, learningRate)

            # #calcuate actual Q values
            # actualQs = qNet.forwardCycle(state)
            # targetQs = actualQs.copy()

            # #get target Q values
            # if done:
            #     targetQs[action] = reward
            # else:
            #     nextQ = targetNet.forwardCycle(nextState)
            #     #getRidOfInvalidActions(nextQ, myGame.getActionSpace(nextState))
            #     targetQs[action] = reward + gamma * np.max(nextQ)
            
            # #make all q values zero except for the valid actions
            # getRidOfInvalidActions(targetQs, myGame.getActionSpace(state))

            # #backpropagate network 
            # gradients = actualQs - targetQs
            # qNet.backpropagation(gradients, learningRate)  

def getAction(qNet,actionSpace, currentState):
    if actionSpace != []:
        if np.random.rand() <= eps:
            action = np.random.choice(actionSpace)  # Explore
        else:
            qValues = qNet.forwardCycle(currentState)
            action = myGame.getPredictAction(actionSpace, qValues)  # Exploit
            #action = np.argmax(qValues)
    else:
        #return the last action if the game has already ended 
        if qNet.getName() == "playerX":
            actionList = myGame.getXActionList()
            action = actionList[-1]
        else:
            actionList = myGame.getOActionList()
            action = actionList[-1]
    return action

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
        print()

    #reset game
    myGame.reset()

    #let player X do the first Move
    currentStateX = myGame.getBoard().copy()
    actionSpace = myGame.getActionSpace(currentStateX)
    actionX = getAction(qNetPlayerX, actionSpace, currentStateX)
    myGame.playerXMove(actionX)

    #get reward for this action
    rewardX = myGame.getReward(actionX, "playerX")

    while not myGame.isDone():

        done = myGame.isDone()

        #move player O
        currentStateO = myGame.getBoard().copy()
        actionSpace = myGame.getActionSpace(currentStateO)
        actionO = getAction(qNetPlayerO, actionSpace, currentStateO)
        myGame.playerOMove(actionO)

        #get reward for this action
        rewardO = myGame.getReward(actionO, "playerO")

        #set nextState for X 
        nextStateX = myGame.getBoard().copy()

        #store experience for player X 
        experience = currentStateX, actionX, rewardX, nextStateX, done
        memoryX.append(experience)


        done = myGame.isDone()
        #move player X
        currentStateX = myGame.getBoard().copy()
        actionSpace = myGame.getActionSpace(currentStateX)
        actionX = getAction(qNetPlayerX, actionSpace, currentStateX)
        myGame.playerXMove(actionX)

        #get reward for this action
        rewardX = myGame.getReward(actionX, "playerX")

        #set nextState for O
        nextStateO = myGame.getBoard().copy()

        #store experience for player O
        experience = currentStateO, actionO, rewardO, nextStateO, done
        memoryO.append(experience)

        # #train both models
        # train(qNetPlayerX, targetNetX, memoryX, learningRate)
        # train(qNetPlayerO, targetNetO, memoryO, learningRate) 

    #if player x wins the game store the losing state and lastest action of O in O memory
    #additionally stoer the missing win state in x memory 
    if not myGame.checkWin(myGame.getOActionList()):

        experience = currentStateX, actionX, rewardX, None, True
        memoryX.append(experience)

        currentStateO = myGame.getBoard().copy()
        actionO = myGame.getOActionList()[-1] 

        experience = currentStateO, actionO, myGame.getReward(actionO, "playerO"), None, True
        memoryO.append(experience)

    else:
        #if player O wins, store the losing experience in x
        currentStateX = myGame.getBoard().copy()
        actionX = myGame.getXActionList()[-1] 

        experience = currentStateX, actionX, -1, None, True
        memoryX.append(experience)



    #train both models
    train(qNetPlayerX, targetNetX, memoryX, learningRate)
    train(qNetPlayerO, targetNetO, memoryO, learningRate) 
    
    #update targetNet every 1000 step 
    if episode % targetUpdate == 0:
        #update target network
        targetNetO.transferFrom(qNetPlayerO)
        targetNetX.transferFrom(qNetPlayerX)
        step = 0

    # epsilon decay
    if eps > epsMin:
        eps = epsMin + (eps - epsMin) * np.exp(-epsDecay * episode)
        #eps -= epsDecay
    #learning rate decay
    learningRate *= 1/(1 + learningRateDecay * episode)

qNetPlayerO.storeModel("ModelO.json")
qNetPlayerX.storeModel("ModelX.json")
print("Training Complete")
