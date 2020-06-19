import numpy as np
import torch

def predictor(model, targets, model_type='fc', org_size=28, cuda=True):
    preds = []
    for target in targets:
        if cuda: target = target.cuda()
        if model_type == 'fc': target = torch.flatten(target)
        pred = model(target).cpu().detach().numpy()
        if model_type == 'fc': pred = np.reshape(pred, (org_size,org_size))
        elif model_type == 'cnn': pred = pred[0][0]
        preds.append(pred)
    return np.concatenate(preds, axis=1)