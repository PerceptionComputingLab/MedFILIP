import multiprocessing
import subprocess
from typing import List
import time
import os

def task_func(args):
    while True:
        try:
            subprocess.check_call(['python3', "prompt.py",args])
        except Exception as e:
            time.sleep(10)
            print(f"Error occurred: {e}. Restarting...")


if __name__ == '__main__':
    processes = []
    for i in range(10,20):
        path = "./results/"+"p"+str(i)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    for i in range(10,20):
        p = multiprocessing.Process(target=task_func, args=("p"+str(i),))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()