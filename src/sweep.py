import subprocess
import sys

lrs = [0.001, 0.01, 0.1]
noises = [0.05, 0.1, 0.2]
comb= int(0)

for i in range(len(lrs)):
    for j in range(len(noises)):
        comb+=1
        print(f"Running {comb}/9 lr={lrs[i]} noise = {noises[j]}")
        command = [sys.executable,"src/train.py","--lr", f"{lrs[i]}", "--noise", f"{noises[j]}"]
        result = subprocess.run(command,capture_output=True,text=True)
        if result.returncode !=0:
            print(result.stderr)