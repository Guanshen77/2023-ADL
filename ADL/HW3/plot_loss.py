import json
import datetime
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--record_file', type=str, default = None)
args = parser.parse_args()
record_path = args.record_file



loss_array = np.array([])
step_array = np.array([])

with open(args.record_file, 'r') as file_record:
    file_content = file_record.read()  # Read the content of the file as a string
    data_record = json.loads(file_content)['log_history']
    

for i in range(len(data_record)):
    try:            
        if "eval_loss" in data_record[i]:
            loss_array =  np.append(loss_array, float(data_record[i]["eval_loss"]))
            step_array =  np.append(step_array, data_record[i]["step"])
    except json.JSONDecodeError:
        # Handle lines that are not valid JSON, if necessary
        print(f"Skipping invalid JSON: {line}")


plt.figure(0)
plt.title('validation loss',fontsize=24)        
plt.xlabel('steps',fontsize=14)
plt.ylabel('loss',fontsize=14)
x_major_locator=MultipleLocator(500) 
ax=plt.gca() 
ax.plot(step_array, loss_array)
ax.xaxis.set_major_locator(x_major_locator)
plt.savefig("./loss.png")