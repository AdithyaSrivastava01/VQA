import torch
import torch.nn as nn
import numpy as np
from model import Model
from scipy import stats

def train(dataloader, model, loss_fn, optimizer):
    
    loss_list=[]
    model.train()
    

    for index,(x,y) in enumerate(dataloader):
        optimizer.zero_grad()


        #Compute prediction error
        pred = model(x)
        
        loss = loss_fn(pred,torch.squeeze(y.float()))
        loss_list+=[loss]
        loss.backward()
       

        # Backpropagation
        optimizer.step()
        
        
        
    return((sum(loss_list))/len(loss_list))





def training(it,train_dl,model,loss_fn,opt,test_data,testY):
    correlation={}
    loss={}
    final_list_loss=np.array([])
    for i in range(it):
        preds=[]
        final_list_loss=np.append(final_list_loss,np.array(train(train_dl, model, loss_fn, opt).cpu().detach().numpy()))
        # print(i," iterations are done")
        # print(final_list_loss[-1])
        if(i%10==0):
            for img, label in test_data:
               
                with torch.no_grad():
                    model.eval()
                    pred = model(img.unsqueeze(0))[0].item()
                preds.append(pred)
            
            # print(stats.spearmanr(preds, testY))
            loss[i]=final_list_loss[-1]
            correlation[i]=stats.spearmanr(preds, testY)[0]

    return (correlation,loss)
    

        #torch.save(model.state_dict(), 'VQAweights.pth')


if(__name__=="__main__"):
    print("in train file")