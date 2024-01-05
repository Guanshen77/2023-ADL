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
lr_array = np.array([])
rouge_1_array = np.array([])
rouge_2_array = np.array([])
rouge_L_array = np.array([])
rouge_Lsum_array = np.array([])
epoch_array = np.array([])

file_record = open(record_path,"r")
data_record = json.load(file_record)


for i in range(len(data_record)):
    try:            
        train_loss_array =  np.append(train_loss_array, data_record[i]["train_loss"])
        lr_array =  np.append(lr_array, data_record[i]["lr"])
        rouge_1_array =  np.append(rouge_1_array, data_record[i]["rouge"]["rouge1"])
        rouge_2_array =  np.append(rouge_2_array, data_record[i]["rouge"]["rouge2"])
        rouge_L_array =  np.append(rouge_L_array, data_record[i]["rouge"]["rougeL"])
        rouge_Lsum_array =  np.append(rouge_Lsum_array, data_record[i]["rouge"]["rougeLsum"])
        epoch_array =  np.append(epoch_array, i)
    except json.JSONDecodeError:
        # Handle lines that are not valid JSON, if necessary
        print(f"Skipping invalid JSON: {line}")


plt.figure(0)
plt.title('training loss',fontsize=24)        
plt.xlabel('epochs',fontsize=14)
plt.ylabel('loss',fontsize=14)
x_major_locator=MultipleLocator(10) 
ax=plt.gca() 
ax.plot(epoch_array, train_loss_array)
ax.xaxis.set_major_locator(x_major_locator)
plt.savefig("./train_loss.png")

plt.figure(1)
plt.title('learning rate',fontsize=24)        
plt.xlabel('epochs',fontsize=14)
plt.ylabel('rate',fontsize=14)
x_major_locator=MultipleLocator(10) 
ax=plt.gca() 
ax.plot(epoch_array, lr_array)
ax.xaxis.set_major_locator(x_major_locator)
plt.savefig("./learning_rate.png")

plt.figure(2)
plt.title('rouge_1',fontsize=24)        
plt.xlabel('epochs',fontsize=14)
plt.ylabel('accuracy',fontsize=14)
x_major_locator=MultipleLocator(10) 
ax=plt.gca() 
ax.plot(epoch_array, rouge_1_array)
ax.xaxis.set_major_locator(x_major_locator)
plt.savefig("./rouge_1.png")

plt.figure(3)
plt.title('rouge_2',fontsize=24)        
plt.xlabel('epochs',fontsize=14)
plt.ylabel('accuracy',fontsize=14)
x_major_locator=MultipleLocator(10) 
ax=plt.gca() 
ax.plot(epoch_array, rouge_2_array)
ax.xaxis.set_major_locator(x_major_locator)
plt.savefig("./rouge_2.png")

plt.figure(4)
plt.title('rouge_L',fontsize=24)        
plt.xlabel('epochs',fontsize=14)
plt.ylabel('accuracy',fontsize=14)
x_major_locator=MultipleLocator(10) 
ax=plt.gca() 
ax.plot(epoch_array, rouge_L_array)
ax.xaxis.set_major_locator(x_major_locator)
plt.savefig("./rouge_L.png")

plt.figure(4)
plt.title('rouge_Lsum',fontsize=24)        
plt.xlabel('epochs',fontsize=14)
plt.ylabel('accuracy',fontsize=14)
x_major_locator=MultipleLocator(10) 
ax=plt.gca() 
ax.plot(epoch_array, rouge_Lsum_array)
ax.xaxis.set_major_locator(x_major_locator)
plt.savefig("./rouge_Lsum.png")