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
epsDecay = 0.00001
targetUpdate = 10
memorySize = 100000
totalEpisodes = 10000
learningRate = 0.1
learningRateDecay = 0.001 
percentOWinning = 0
percentDraw = 0

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
                getRidOfInvalidActions(nextQ, myGame.getActionSpace(nextState))
                targetQs[action] = reward + gamma * np.max(nextQ)
            
            #make all q values zero except for the valid actions
            getRidOfInvalidActions(targetQs, myGame.getActionSpace(state))

            #backpropagate network 
            gradients = learningRate * (targetQs - actualQs)
            qNet.backpropagation(gradients, learningRate)  

def getAction(qNet,actionSpace, currentState):
    if np.random.rand() <= eps:
        action = np.random.choice(actionSpace)  # Explore
    else:
        qValues = qNet.forwardCycle(currentState)
        action = myGame.getPredictAction(actionSpace, qValues)  # Exploit
        #action = np.argmax(qValues)
    return action

#debug function to show the board
def displayBoard(board):
    board = board.reshape(-1,1).tolist()
    for i in range(3):
        print(board[i*3], board[i*3+1], board[i*3+2])

for episode in range(totalEpisodes):
    
    #for debug purposes
    if(episode % 1 == 0):
        displayBoard(myGame.getBoard())
        print("Episode: ", episode)
        print()

    #reset game
    myGame.reset()

    #let x player the first move 
    currentStateX = np.zeros((9,1))
    actionSpace = [0,1,2,3,4,5,6,7,8]
    actionX = getAction(qNetPlayerX, actionSpace, currentStateX)
    myGame.playerXMove(actionX)

    #get the reward for the first Move 
    rewardX = myGame.getReward(actionX, "playerX")
    
    #while game is not done
    while not myGame.isDone():

        #player O move
        currentStateO = myGame.getBoard().copy()
        actionSpace = myGame.getActionSpace(currentStateO)
        actionO = getAction(qNetPlayerO, actionSpace, currentStateO)
        myGame.playerOMove(actionO)

        #get reward for player O
        rewardO = myGame.getReward(actionO, "playerO")

        #record the state after O moves to be the next state for X
        nextStateX = myGame.getBoard().copy()

        #get and store experience for player X
        experience = (currentStateX, actionX, rewardX, nextStateX, myGame.isDone())
        memoryX.append(experience)
        
        #if player O didn't win
        if not myGame.isDone():
            #player X move
            currentStateX = myGame.getBoard().copy()
            actionSpace = myGame.getActionSpace(currentStateX)
            actionX = getAction(qNetPlayerX, actionSpace, currentStateX)
            myGame.playerXMove(actionX)

            #get reward for player X
            rewardX = myGame.getReward(actionX, "playerX")

            #get next state for player O
            nextStateO = myGame.getBoard().copy()

            #get and store experience for player O
            experience = (currentStateO, actionO, rewardO, nextStateO, myGame.isDone())
            memoryO.append(experience)    
        else:
            #if action O is the terminating action
            nextStateO = np.zeros((9,1))
            experience = (currentStateO, actionO, rewardO, nextStateO, True)
            memoryO.append(experience)         

    #if actionX is the terminating action
    nextStateX = np.zeros((9,1))
    experience = (currentStateX, actionX, rewardX, nextStateX, True)
    memoryX.append(experience)

    #train both models
    train(qNetPlayerX, targetNetX, memoryX, learningRate)
    train(qNetPlayerO, targetNetO, memoryO, learningRate) 
    
    #update hyper parameters
    if episode % targetUpdate == 0:
        #update target network
        targetNetO.transferFrom(qNetPlayerO)
        targetNetX.transferFrom(qNetPlayerX)

    # epsilon decay
    if eps > epsMin:
        eps *= np.exp(-epsDecay * episode)
    #learning rate decay
    learningRate *= 1/(1 + learningRateDecay * episode)

qNetPlayerO.storeModel("ModelO.json")
qNetPlayerX.storeModel("ModelX.json")
print("Training Complete")
