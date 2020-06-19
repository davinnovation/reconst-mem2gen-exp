import numpy as np

import torch
import torch.nn as nn

def reconst_trainer(model, target_tensor, obj_threshold=1e-4, max_iter=1000, loss=nn.MSELoss(), opt=torch.optim.Adam)->None:
    opt = opt(model.parameters(), lr=0.001)
    diff = np.inf
    cnt = 0
    while diff > obj_threshold or max_iter > cnt:
        pred = model(target_tensor)
        
        diff = torch.mean(target_tensor - pred)

        opt.zero_grad()
        loss(pred, target_tensor).backward()
        opt.step()
        
        cnt+=1
