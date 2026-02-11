# weight sand biases experiment guide
import wandb

# init W&B
wandb.init(
    project="da6401-mlp-assignment"
    config = {
        "architecture": "MLP",
        "dataset": "mnist",
        "epochs": 10,
        #...other hyperparameters
    }
)

# log during training
wandb.log({
    "train_loss": train_loss,
    "train_acc": train_acc,
    "val_loss": val_loss,
    "val_acc": val_acc,
    "epoch" : epoch
})

# Finsh run
wandb.finish()