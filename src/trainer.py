from model import get_model
import torch

def train_model(x,y,lr,epochs):
        losses = []
        model = get_model()
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(params=model.parameters(),lr=lr)
        for epoch in range(epochs):
            model.train()
            y_pred = model(x)
            loss = loss_fn(y_pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        return model,losses