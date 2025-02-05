import torch 
from sklearn.metrics import r2_score

def get_results(gt, pred):
    mae=torch.mean(torch.abs(gt-pred))
    error=torch.sum((gt-pred)**2,axis=1)
    rmse=torch.sqrt(torch.mean(error)) ## RMSE
    r2 = r2_score(gt.cpu().detach().numpy(), pred.cpu().detach().numpy())

    return rmse, mae, r2