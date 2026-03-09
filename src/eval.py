import argparse
import json
from pathlib import Path
import yaml
import torch

def make_dataset(n_sample:int, noise: int):
    x = torch.randn(n_sample,1)
    y = 3*x+2+noise*torch.randn(n_sample,1)
    return x,y

def main()->None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",required=True, help="Path to a run folder inside runs/")
    args = parser.parse_args()
    run_dir = Path(args.run)
    metrics_path = run_dir/"metrics.json"
    if not run_dir.exists():
        raise FileNotFoundError(f"Run folder not found: {run_dir}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found in: {run_dir}")
    with open(metrics_path,"r",encoding="utf-8") as f:
        metrics = json.load(f)
    config_path = run_dir/"config.yaml"
    checkpoint_path = run_dir/"model.pt"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in: {run_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"model.pt not found in: {run_dir}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    seed = config["seed"]
    n_samples = config["data"]["n_samples"]
    noise = config["data"]["noise"]
    torch.manual_seed(seed)
    X,y = make_dataset(n_samples,noise)
    model = torch.nn.Linear(1,1)
    loss_fn = torch.nn.MSELoss()
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.inference_mode():
        preds = model(X)
        loss = loss_fn(preds,y).item()
    saved_loss = metrics["final_loss"]
    print(config)
    print("\n=== EVAL REPORT ===")
    print(f"Run: {run_dir.name}")
    print(json.dumps(metrics, indent=2))
    print(f"Saved loss:{saved_loss:.3f}")
    print(f"Eval loss:{loss:.3f}")
    print(f"Differnec :{abs(saved_loss-loss)}")
    if abs(saved_loss - loss) < 1e-3 :
        print("Passed")
    else:
        print("Failed")
    print("===================\n")
    
if __name__ == "__main__":
    main()

