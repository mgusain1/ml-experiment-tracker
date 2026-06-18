from model import get_model
import torch

def train_model(X_train,X_val,y_train,y_val,lr,epochs,run_folder):
        train_losses = []
        validation_losses = []
        count =0
        best_validation_loss = float("inf")
        patience = 5
        model = get_model()
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(params=model.parameters(),lr=lr)
        for epoch in range(epochs):
            model.train()
            y_pred = model(X_train)
            loss = loss_fn(y_pred,y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            with torch.inference_mode():
                model.eval()
                y_val_pred = model(X_val)
                val_loss = loss_fn(y_val_pred,y_val)
                validation_losses.append(val_loss.item())
                if val_loss.item() < best_validation_loss:
                    best_validation_loss = val_loss.item()
                    best_epoch = epoch+1
                    count=0
                    torch.save(model.state_dict(),f"{run_folder}/best_model.pt")
                else:
                    count+=1
                if count >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        return model,train_losses, validation_losses, best_validation_loss, best_epoch