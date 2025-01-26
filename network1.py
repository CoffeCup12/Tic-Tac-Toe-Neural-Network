import torch
import torch.nn as nn
import torch.optim as optim

class netWork(nn.Module):

    def __init__(self):
        super(netWork, self).__init__()
        self.fc1 = nn.Linear(27, 18)
        self.fc2 = nn.Linear(18, 18)
        self.fc3 = nn.Linear(18, 9)
    
    def forward(self, x):
        x = self.processInput(x)
        x = x.view(-1, 27)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def processInput(self, state):
        #create new input layout 
        #feed = torch.zeros(27, device="xpu")
        feed = torch.zeros(27)
        #for each block, there exist 3 corresponding positons, the top is 1 if X occupied, 
        # mid is 1 is blank, bottom is 1 if O occupied 
        for i in range(9):

            top = i * 3
            mid = top + 1
            bot = top + 2

            if state[i] == 1:
               feed[top] = 1
            elif state[i] == 0.5:
                feed[bot] = 1
            elif state[i] == 0:
                feed[mid] = 1

        return feed  
