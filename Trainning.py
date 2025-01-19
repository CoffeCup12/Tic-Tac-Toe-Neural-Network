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
totalEpisodes = 100000 
learningRate = 0.0005
#learningRateDecay = 0.001
step = 0

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
    qValues[~mask] = -1


def train(qNet, targetNet, memory, learningRate):

    # Train the network
    if len(memory) > batchSize:

        #take sample batch based on reward
        prioritizedMemory = sorted(memory, key=lambda x: x[2], reverse=True) # Sort by reward
        sampleBatch = random.sample(prioritizedMemory, batchSize)
        
        #for each experience in the batch
        for state, action, reward, nextState, done in sampleBatch:

            #calcuate actual Q values
            actualQs = qNet.forwardCycle(state)
            targetQs = actualQs.copy()

            #get target Q values
            if done:
                targetQs[action] = reward
            else:
                nextQ = targetNet.forwardCycle(nextState)
                #getRidOfInvalidActions(nextQ, myGame.getActionSpace(nextState))
                targetQs[action] = reward + gamma * np.max(nextQ)
            
            #make all q values zero except for the valid actions
            getRidOfInvalidActions(targetQs, myGame.getActionSpace(state))

            #backpropagate network 
            gradients = learningRate * (targetQs - actualQs)
            qNet.backpropagation(gradients, learningRate)  

def getAction(qNet,actionSpace, currentState):
    if actionSpace != []:
        if np.random.rand() <= eps:
            action = np.random.choice(actionSpace)  # Explore
        else:
            qValues = qNet.forwardCycle(currentState)
            action = myGame.getPredictAction(actionSpace, qValues)  # Exploit
            #action = np.argmax(qValues)
    else:
        action = -1
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

    while not myGame.isDone():
        #move player x
        currentState = myGame.getBoard().copy()
        actionSpace = myGame.getActionSpace(currentState)
        actionX = getAction(qNetPlayerX, actionSpace, currentState)
        myGame.playerXMove(actionX)

        #nextState and calculate reward
        nextState = myGame.getBoard().copy()
        reward = myGame.getReward(actionX, "playerX")

        #store experience
        experience = currentState, actionX, reward, nextState, myGame.isDone() 
        memoryX.append(experience)

        #print(experience)
        
        #move player o
        currentState = myGame.getBoard().copy()
        actionSpace = myGame.getActionSpace(currentState)
        actionO = getAction(qNetPlayerX, actionSpace, currentState)
        myGame.playerOMove(actionO)

        #next state and calculate reward
        nextState = myGame.getBoard().copy() 
        reward = myGame.getReward(actionO, "playerO")    

        #store experince in memory
        experience = currentState, actionO, reward, nextState, myGame.isDone()
        memoryO.append(experience)

        #print(experience)

        step += 2     

    #train both models
    train(qNetPlayerX, targetNetX, memoryX, learningRate)
    train(qNetPlayerO, targetNetO, memoryO, learningRate) 
    
    #update hyper parameters
    if step % targetUpdate == 0:
        #update target network
        targetNetO.transferFrom(qNetPlayerO)
        targetNetX.transferFrom(qNetPlayerX)
        step = 0

    # epsilon decay
    if eps > epsMin:
        #eps = epsMin + (eps - epsMin) * np.exp(-epsDecay * episode)
        eps -= epsDecay
    #learning rate decay
    #learningRate *= 1/(1 + learningRateDecay * episode)

qNetPlayerO.storeModel("ModelO.json")
qNetPlayerX.storeModel("ModelX.json")
print("Training Complete")
