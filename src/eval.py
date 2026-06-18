import argparse
import json
from pathlib import Path
import yaml
import torch

def make_dataset(n_sample:int, noise: int):
    x = torch.randn(n_sample,1)
    y = 3*x+2+noise*torch.randn(n_sample,1)
    return x,y

def evaluate_checkpoint(checkpoint_path, X, y):
    model = torch.nn.Linear(1,1)
    loss_fn = torch.nn.MSELoss()
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.inference_mode():
        preds = model(X)
        loss = loss_fn(preds,y).item()
    return loss
    

def main()->None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",required=True, help="Path to a run folder inside runs/")
    parser.add_argument("--checkpoint",choices=["best","final"],help="Select the checkpoint type (default: best)",default="best")
    parser.add_argument("--compare",action="store_true") 
    args = parser.parse_args()
    run_dir = Path(args.run)
    checkpoint_type = args.checkpoint
    metrics_path = run_dir/"metrics.json"
    if not run_dir.exists():
        raise FileNotFoundError(f"Run folder not found: {run_dir}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found in: {run_dir}")
    with open(metrics_path,"r",encoding="utf-8") as f:
        metrics = json.load(f)
    config_path = run_dir/"config.yaml"
    if checkpoint_type == "best":
        checkpoint_path = run_dir/"best_model.pt"
    elif checkpoint_type == "final":
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
    if args.compare:
        final_checkpoint = run_dir / "model.pt"
        best_checkpoint = run_dir / "best_model.pt"

        if not final_checkpoint.exists():
            raise FileNotFoundError(f"Final checkpoint not found: {final_checkpoint}")

        if not best_checkpoint.exists():
            raise FileNotFoundError(f"Best checkpoint not found: {best_checkpoint}")

        final_model_loss = evaluate_checkpoint(final_checkpoint, X, y)
        best_model_loss = evaluate_checkpoint(best_checkpoint, X, y)

        print("\n=== CHECKPOINT COMPARISON ===")
        print(f"Run: {run_dir.name}")
        print(f"Best checkpoint loss : {best_model_loss:.6f}")
        print(f"Final checkpoint loss: {final_model_loss:.6f}")

        if best_model_loss < final_model_loss:
            print("Winner: best checkpoint")
        elif final_model_loss < best_model_loss:
            print("Winner: final checkpoint")
        else:
            print("Winner: tie")

        print("=============================\n")
        return
    loss = evaluate_checkpoint(checkpoint_path,X,y)
    saved_loss = metrics["final_validation_loss"]
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

