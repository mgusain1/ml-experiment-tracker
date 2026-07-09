from model import get_model
import torch
import time
from torch.optim.lr_scheduler import StepLR

def train_model(X_train,X_val,y_train,y_val,lr,epochs,run_folder,logger):
        train_losses = []
        validation_losses = []
        count =0
        best_validation_loss = float("inf")
        patience = 5
        model = get_model()
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(params=model.parameters(),lr=lr)
        scheduler = StepLR(optimizer,step_size=5,gamma=0.5)
        logger.info("Training Started")
        start_time = time.perf_counter()
        best_epoch = 0
        completed_epochs = 0
        learning_rate_history = []
        for epoch in range(epochs):
            model.train()
            y_pred = model(X_train)
            loss = loss_fn(y_pred,y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())
            completed_epochs+=1
            with torch.inference_mode():
                model.eval()
                y_val_pred = model(X_val)
                val_loss = loss_fn(y_val_pred,y_val)
                validation_losses.append(val_loss.item())
                current_lr = scheduler.get_last_lr()[0]
                learning_rate_history.append(current_lr)
                logger.info(f"Epoch {epoch + 1}: train_loss={loss.item():.6f}, val_loss={val_loss.item():.6f}, lr= {current_lr}")
                if val_loss.item() < best_validation_loss:
                    best_validation_loss = val_loss.item()
                    best_epoch = epoch+1
                    count=0
                    torch.save(model.state_dict(),f"{run_folder}/best_model.pt")
                    logger.info(f"Best model updated at epoch {epoch + 1}: val_loss={val_loss.item():.6f}")
                else:
                    count+=1
                if count >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": loss.item(),
                "val_loss": val_loss.item(),
                "best_validation_loss": best_validation_loss,
                "best_epoch": best_epoch,
                "patience_count": count,
                "patience": patience,
            }
            checkpoint_pth = f"{run_folder}/checkpoint.pt"
            torch.save(checkpoint, checkpoint_pth)
        end_time = time.perf_counter()
        total_time = end_time-start_time
        avg_epoch_time = total_time/completed_epochs
        logger.info(f"Training finished. Best epoch={best_epoch}, best_val_loss={best_validation_loss:.6f}")
        return model,train_losses, validation_losses, best_validation_loss, best_epoch, total_time, avg_epoch_time, completed_epochs, learning_rate_history