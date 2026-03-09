
import yaml
import torch
import argparse
import sys
import json
import subprocess
from datetime import datetime
import random
import string
import os
import shlex
from data import make_dataset
from model import get_model
from trainer import train_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",required=False,default="configs/exp1.yaml")
    parser.add_argument("--lr",type=float,default=None)
    parser.add_argument("--epochs",type=int,default=None)
    parser.add_argument("--noise",type=float,default=None)
    parser.add_argument("--n_samples",type=int,default=None)
    parser.add_argument("--seed",type=int,default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    rand_has = ''.join(random.choices(string.ascii_lowercase+string.digits,k=6))
    run_id = f"{timestamp}_{rand_has}"
    run_folder = f"runs/{run_id}"
    os.makedirs(run_folder,exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)
        if args.seed is not None:
            seed = args.seed
        else:
            seed = config["seed"]
        if args.n_samples is not None:
            n_samples = args.n_samples
        else:
            n_samples = config["data"]["n_samples"]
        if args.noise is not None:
            noise = args.noise
        else:
            noise = config["data"]["noise"]
        if args.lr is not None:
            lr = args.lr
        else:
            lr = config["training"]["lr"]
        if args.epochs is not None:
            epochs = args.epochs
        else:
            epochs = config["training"]["epochs"]
        torch.manual_seed(seed)
        x, y = make_dataset(n_samples,noise)
        model,losses = train_model(x,y,lr,epochs)

    git_hash = subprocess.getoutput("git rev-parse HEAD")

    metadata = {
        "run_id":run_id,
        "timestamp":datetime.now().isoformat(),
        "git_commit":git_hash,
        "python_version":sys.version,
        "command":f"python "+ shlex.join(sys.argv) 
    }

    config_data = {
        "seed":seed,
        "data":{
            "n_samples":n_samples,
            "noise":noise
        },
        "training":{
            "lr":lr,
            "epochs":epochs
        }
    }

    with open(f"{run_folder}/metadata.json", "w") as f:
        json.dump(metadata,f,indent=4)
        
    with open(f"{run_folder}/config.yaml","w") as f:
        yaml.dump(config_data,f,indent=1)
        
        
    metrics = {
        "loss": losses,
        "final_loss":losses[-1]
    }

    with open(f"{run_folder}/metrics.json","w") as f:
        json.dump(metrics,f,indent=4)
    torch.save(model.state_dict(), f"{run_folder}/model.pt")

    print(f"Run folder created: {run_folder}")

if __name__ == "__main__":
    main()
