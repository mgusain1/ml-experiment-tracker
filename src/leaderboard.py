from pathlib import Path
import os
import yaml
import json
import argparse

run_folder = Path("runs")
records = []
subdirectories = [x.name for x in run_folder.iterdir() if x.is_dir()]
for dir_name in subdirectories:
    directory = run_folder / dir_name
    metrics_path = directory/"metrics.json"
    config_path = directory/"config.yaml"
    if not metrics_path.exists():
        print(f"No matricss in {directory}")
        continue
    if not config_path.exists():
        print(f"No Config in {directory}")
        continue
    with open(config_path) as f:
        config = yaml.safe_load(f)
    with open(metrics_path) as f:
        metrics = json.load(f)
    seed = config["seed"]
    lr = config["training"]["lr"]
    epochs = config["training"]["epochs"]
    final_loss = metrics["final_loss"]
    records.append({
        "run_id":directory,
        "seed":seed,
        "epochs": epochs,
        "Learning_rate":lr,
        "final_loss":final_loss
    })
        
records = sorted(records,key=lambda x:x["final_loss"])
parser = argparse.ArgumentParser()
parser.add_argument("--top",type=int,default=5,required=False, help="“Show top N runs")
args = parser.parse_args()
top = args.top
rank = 1
if not records:
    print("Records is Empty")
else:
    for r in records[:top]:
        print(f"Rank {rank}: Run_Id={r['run_id']}, seed={r['seed']}, Learning rate={r['Learning_rate']}, Epochs={r['epochs']}, final loss={r['final_loss']}")
        rank+=1
    print(f"Best one is Run_id: {records[0]['run_id']} and Loss: {records[0]['final_loss']}")
    

