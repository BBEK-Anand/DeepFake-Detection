#import your nessessary libreries here


#define your Optimizer functions here
###DEMO
import torch.optim as optim

def OptAdam(model,**kwargs):
    return optim.Adam(model.parameters(),**kwargs)
