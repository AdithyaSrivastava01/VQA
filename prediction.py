from scipy import stats
import torch


def prediction(model,test_data,testY):
	
	preds = []
	for img, label in test_data:
	   
	    with torch.no_grad():
	        model.eval()
	        pred = model(img.unsqueeze(0))[0].item()
	    preds.append(pred)
	
	x,y=stats.spearmanr(preds, testY)
	
	

	return x
if(__name__=="__main__"):
    print("in prediction file")