import subprocess
import sys
from datetime import datetime
import random
import string

lrs = [0.001, 0.01, 0.1]
noises = [0.05, 0.1, 0.2]
comb= int(0)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
rand_has = ''.join(random.choices(string.ascii_lowercase+string.digits,k=6))
sweep_id = f"sweep_{timestamp}_{rand_has}"
print(f"Starting sweep: {sweep_id}")


for i in range(len(lrs)):
    for j in range(len(noises)):
        comb+=1
        print(f"Running {comb}/9 lr={lrs[i]} noise = {noises[j]}")
        command = [sys.executable,"src/train.py","--lr", f"{lrs[i]}", "--noise", f"{noises[j]}","--sweep_id",f"{sweep_id}"]
        result = subprocess.run(command,capture_output=True,text=True)
        if result.returncode !=0:
            print(result.stderr)
        else:
            print(result.stdout)