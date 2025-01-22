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
targetUpdate = 1000
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

        #get reward for this action
        rewardX, doneO = myGame.getReward(actionX, "playerX")

        #set nextState for O
        nextStateO = myGame.getBoard()

        #store experience for player O
        experience = currentStateO, actionO, rewardO, nextStateO, doneO
        memoryO.append(experience)

        if doneX:
            break

        #train both models
        # train(qNetPlayerX, targetNetX, memoryX, learningRate)
        # train(qNetPlayerO, targetNetO, memoryO, learningRate) 

    #if player x wins or draw
    if doneX:
        #store win memeory for X 
        currentState = myGame.getBoard()
        action = myGame.getXActionList()[-1]
        reward = myGame.getReward(action, "playerX")
        memoryX.append((currentState,action,reward, None, True))

        #store lose memory for O
        currentState = myGame.getBoard()
        action = myGame.getOActionList()[-1]
        reward = myGame.getReward(action, "playerX")
        memoryO.append((currentState,action,reward, None, True))
        
    elif doneO:
        #store win memeory for O 
        currentState = myGame.getBoard()
        action = myGame.getOActionList()[-1]
        memoryO.append((currentState,action,1, None, True))

        #store lose memory for X
        currentState = myGame.getBoard()
        action = myGame.getXActionList()[-1]
        memoryX.append((currentState,action,-1, None, True))


    # #train both models
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
