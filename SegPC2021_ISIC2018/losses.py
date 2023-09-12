import torch
from torch.nn import functional as F



EPSILON = 1e-6

class DiceLoss_old(torch.nn.Module):
    def __init__(self,):
        super().__init__()
    
    def forward(self, pred, mask):
        pred = pred.flatten()
        mask = mask.flatten()
        
        intersect = (mask * pred).sum()
        dice_score = 2*intersect / (pred.sum() + mask.sum() + EPSILON)
        dice_loss = 1 - dice_score
        return dice_loss

class DiceLoss(torch.nn.Module):
    def __init__(self,):
        super().__init__()
    
    def forward(self, pred, mask):
        pred = pred.flatten()
        mask = mask.flatten()
        p = pred
        t = mask
        numerator = 2 * (p * t).sum(1)
        denominator = p.sum(-1) + t.sum(-1)
        dice_loss = 1 - (numerator + 1) / (denominator + 1)
        
        # print "------",dice.data

        
        return dice_loss
    
class DiceLossWithLogtis(torch.nn.Module):
    def __init__(self,):
        super().__init__()
    
    def forward(self, pred, mask):
        prob = F.softmax(pred, dim=1)
        true_1_hot = mask.type(prob.type())
        
        dims = (0,) + tuple(range(2, true_1_hot.ndimension()))
        intersection = torch.sum(prob * true_1_hot, dims)
        cardinality = torch.sum(prob + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + EPSILON)).mean()
        return (1 - dice_loss)

class DiceLossWithLogtis_new(torch.nn.Module):
    def __init__(self,):
        super().__init__()
    
    def forward(self, pred, mask):
        logit = F.softmax(pred, dim=1)
        truth = mask
        batch_size = len(logit)
        logit = logit.view(batch_size,-1)
        truth = truth.view(batch_size,-1)
        assert(logit.shape==truth.shape)
        p = logit.view(batch_size,-1)
        t = truth.view(batch_size,-1)
        #w = truth.detach()
        #w = w*(self.weights[1]-self.weights[0])+self.weights[0]
        ## p = w*(p*2-1)  #convert to [0,1] --> [-1, 1]
        ## t = w*(t*2-1)
        #p = w*(p)
        #t = w*(t)
        numerator = 2 * (p * t).sum(1)
        denominator = p.sum(-1) + t.sum(-1)
        dice = 1 - (numerator + 1) / (denominator + 1)
        loss = dice.mean()
        return loss
    
    