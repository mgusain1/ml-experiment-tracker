import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top",type=int,default=3,required=False)
    args = parser.parse_args()
    top = args.top
    run_folder = Path("runs")
    records = []
    for run in run_folder.iterdir():
        if not run.is_dir():
            continue
        metrics_path = run/"metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        loss = metrics["loss"]
        final_loss = metrics["final_loss"]
        records.append({
            "run_id":run.name,
            "losses":loss,
            "final_loss":final_loss
        })
    records = sorted(records,key=lambda x:x["final_loss"])
    if not records:
        print("No runs found")
        return
    for r in records[:top]:
        print(f"{r['run_id']}  loss={r['final_loss']}")
        lossess = r['losses']
        epochs = list(range(len(lossess)))
        plt.plot(epochs,lossess,label=r["run_id"])
    plt.title("Epochs vs Loss Each run")
    plt.xlabel('Epochs') #
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{run_folder}/comparison.png")
    plt.close()
        