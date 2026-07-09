import matplotlib.pyplot as plt

def plot_loss(train_losses,val_losses, run_folder):
    epochs = list(range(len(train_losses)))
    plt.plot(epochs,train_losses,label="Training losses over epochs")
    plt.plot(epochs,val_losses,label="Validation losses over epochs")
    plt.title("Training vs Validation Loss")
    plt.xlabel('Epochs') #
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{run_folder}/loss_curve.png")
    plt.close()