import network
import Game
import numpy as np    
import random
from tqdm import tqdm
from collections import deque

# Hyperparameters
batchSize = 400
gamma = 0.99
epsMax = 1
eps = epsMax
epsMin = 0.1
epsDecay = 0.00001
targetUpdate = 10
memorySize = 600
totalEpisodes = 10000
learningRate = 0.3
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

def getExperience(state, action, name, opponentNet):

    #initialize variables
    reward = 0
    done = False

    # Play and get reward
    if name == "playerX":
        myGame.playerXMove(action)
    else:
        myGame.playerOMove(action)

    space = myGame.getActionSpace(myGame.getBoard())
    done = myGame.checkWin(myGame.playerX) or myGame.checkWin(myGame.playerO) or len(space) == 0

    #if the game is not done, let opponent make an arbitrary move and get the next state
    if not done:

        nextQ = opponentNet.forwardCycle(myGame.getBoard())
        opponentAction = myGame.getPredictAction(space, nextQ)

        #TODO this is redudent, might need to refactor
        nextState = myGame.getBoard().copy()
        if name == "playerX":
            nextState[opponentAction] = -1
        else:  
            nextState[opponentAction] = 1

    else:
        done = True
        nextState = myGame.getBoard().copy()
    
    reward = myGame.getReward(nextState, action, name)

    return state, action, reward, nextState, done

# def prioritizeExperience(memory, batchSize):
#     prioritizedMemory = sorted(memory, key=lambda x: x[2], reverse=True) # Sort by reward 
#     sample = random.sample(prioritizedMemory, batchSize)
#     return sample

def getRidOfInvalidActions(qValues, actionSpace):
    mask = np.zeros_like(qValues, dtype=bool)
    mask[actionSpace] = True
    qValues[~mask] = 0


def train(qNet, opponentNet, targetNet, memory, learningRate):
    #get current state and action space
    currentState = myGame.getBoard().copy()
    actionSpace = myGame.getActionSpace(currentState)

    # choose action
    if np.random.rand() <= eps:
        action = np.random.choice(actionSpace)  # Explore
    else:
        qValues = qNet.forwardCycle(currentState)
        action = myGame.getPredictAction(actionSpace, qValues)  # Exploit
        #action = np.argmax(qValues)
    
    # Get experience and store in memory
    experience = getExperience(currentState, action, qNet.getName(), opponentNet)
    memory.append(experience)

    # Train the network
    if len(memory) > batchSize:
        prioritizedMemory = sorted(memory, key=lambda x: x[2], reverse=True) # Sort by reward
        sampleBatch = random.sample(prioritizedMemory, batchSize)
        
        for state, action, reward, nextState, done in sampleBatch:

            actualQs = qNet.forwardCycle(state)
            targetQs = actualQs.copy()

            if done:
                targetQs[action] = reward
            else:
                nextQ = targetNet.forwardCycle(nextState)
                getRidOfInvalidActions(nextQ, myGame.getActionSpace(nextState))
                targetQs[action] = reward + gamma * np.max(nextQ)
            
            getRidOfInvalidActions(targetQs, myGame.getActionSpace(state))

            #train the network  
            gradients = learningRate * (targetQs - actualQs)
            qNet.backpropagation(gradients, learningRate) 

    return experience[4]  

def displayBoard(board):
    board = board.reshape(-1,1).tolist()
    for i in range(3):
        print(board[i*3], board[i*3+1], board[i*3+2])

for episode in range(totalEpisodes):
    
    if(myGame.checkWin(myGame.playerO)):
        percentOWinning += 1
    elif(myGame.checkWin(myGame.playerX)):
        pass
    else:
        percentDraw += 1
    if(episode % 100 == 0):
        displayBoard(myGame.getBoard())
        print("percent draw:", percentDraw/(episode+1))
        print("Episode: ", episode)
        print()

    myGame.reset()
    
    for step in range(9):

        done = False
        if step % 2 == 0:
            done = train(qNetPlayerX, qNetPlayerO, targetNetX, memoryX, learningRate)
        else:
            done = train(qNetPlayerO, qNetPlayerX, targetNetO, memoryO, learningRate)

        if done:
            break
    
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
print("Player O winning percentage: ", percentOWinning/totalEpisodes)
print("Draw percentage: ", percentDraw/totalEpisodes)
print("Training Complete")

# import network
# import Game
# import numpy as np    
# import random
# from tqdm import tqdm
# from collections import deque

# # Hyperparameters
# batchSize = 100
# gamma = 0.99
# epsMax = 1
# eps = epsMax
# epsMin = 0.01
# epsDecay = 0.001
# targetUpdate = 10
# memorySize = 600
# totalEpisodes = 1000
# learningRate = 0.1
# tau = 0.01  # Soft update parameter

# qNet = network.netWork("playerO")
# targetNet = network.netWork("target")

# memory = deque(maxlen=memorySize)
# myGame = Game.game()

# def getExperience(state, action, name):
#     # Play and store result
#     if name == "playerX":
#         myGame.playerXMove(action)
#         reward, done = myGame.getReward(state, action, name)
#     else:
#         myGame.playerOMove(action)
#         reward, done = myGame.getReward(state, action, name)

#     nextState = myGame.getBoard()
#     return state, action, reward, nextState, done

# for episode in range(totalEpisodes):
#     myGame.reset()
#     for step in range(9):
#         currentState = myGame.getBoard().copy()
#         actionSpace = myGame.getActionSpace(currentState)

#         if np.random.rand() <= eps:
#             action = np.random.choice(actionSpace)  # Explore
#         else:
#             q_values = qNet.forwardCycle(currentState)
#             action = myGame.getPredictAction(actionSpace, q_values)  # Exploit
                
#         experience = getExperience(currentState, action, qNet.getName())
#         memory.append(experience)

#         if len(memory) > batchSize:
#             sampleBatch = random.sample(memory, batchSize)
#             for state, action, reward, nextState, done in sampleBatch:
#                 actualQs = qNet.forwardCycle(state)
#                 targetQs = actualQs.copy()
#                 if done:
#                     targetQs[action] = reward
#                 else:
#                     nextQ = targetNet.forwardCycle(nextState)
#                     targetQs[action] = reward + gamma * np.max(nextQ)
                        
#                 gradients = learningRate * (targetQs - actualQs)
#                 qNet.backpropagation(gradients)     

#         if experience[4]:
#             break
        
#     if episode % targetUpdate == 0:
#         # Soft update target network
#         qNet.storeModel('modelO.json')
#         targetNet.update(tau, 'modelO.json')
        
#     if eps > epsMin:
#         eps *= np.exp(-epsDecay * episode)
