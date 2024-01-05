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


train_loss_array = np.array([])
eval_loss_array = np.array([])
eval_metric_array = np.array([])
epoch_array = np.array([])

file_record = open(record_path,"r")
data_record = json.load(file_record)


for i in range(len(data_record)):
    try:            
        
        train_loss_array =  np.append(train_loss_array, data_record[i]["train_loss"])
        eval_loss_array =  np.append(eval_loss_array, data_record[i]["eval_loss"])
        eval_metric_array =  np.append(eval_metric_array, data_record[i]["eval_metric"]["exact_match"])
        epoch_array =  np.append(epoch_array, i)
    except json.JSONDecodeError:
        # Handle lines that are not valid JSON, if necessary
        print(f"Skipping invalid JSON: {line}")


plt.figure(0)
plt.title('training loss',fontsize=24)        
plt.xlabel('epochs',fontsize=14)
plt.ylabel('loss',fontsize=14)
x_major_locator=MultipleLocator(1) 
ax=plt.gca() 
ax.plot(epoch_array, train_loss_array)
ax.xaxis.set_major_locator(x_major_locator)
plt.savefig("./train_loss.png")

plt.figure(1)
plt.title('validation loss',fontsize=24)        
plt.xlabel('epochs',fontsize=14)
plt.ylabel('loss',fontsize=14)
x_major_locator=MultipleLocator(1) 
ax=plt.gca() 
ax.plot(epoch_array, eval_loss_array)
ax.xaxis.set_major_locator(x_major_locator)
plt.savefig("./eval_loss.png")

plt.figure(2)
plt.title('accuracy',fontsize=24)        
plt.xlabel('epochs',fontsize=14)
plt.ylabel('accuracy',fontsize=14)
x_major_locator=MultipleLocator(1) 
ax=plt.gca() 
ax.plot(epoch_array, eval_metric_array)
ax.xaxis.set_major_locator(x_major_locator)
plt.savefig("./accuracy.png")