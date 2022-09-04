import torch.nn as nn
import torch.nn.functional as F
import torch 
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(2048,512)
        self.lin2=nn.Linear(512,64)
        self.lin3=nn.Linear(64,1)
        self.dense_bn1 = nn.BatchNorm1d(512)
        self.dense_bn2 = nn.BatchNorm1d(64) 
        self.drop = nn.Dropout(0.1) 
        
    def forward(self, inp):
        # x = F.relu(self.dense_bn1(self.lin1(inp)))
        x = F.relu(self.lin1(inp))
        x = self.drop(x)
        # x = F.relu(self.dense_bn2(self.lin2(x)))
        x = F.relu(self.lin2(x))
        x = self.drop(x)

        x = torch.sigmoid(self.lin3(x)).reshape(-1)
        return x
if(__name__=="__main__"):
    print("in model file")