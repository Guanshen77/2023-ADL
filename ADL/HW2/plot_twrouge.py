import os
import json
import datetime
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--record_file', type=str, default = None)
parser.add_argument('--output_dir', type=str, default = None)
args = parser.parse_args()
record_path = args.record_file
os.makedirs(args.output_dir, exist_ok=True)


rouge_1_f_array = np.array([])
rouge_2_f_array = np.array([])
rouge_L_f_array = np.array([])
rouge_1_p_array = np.array([])
rouge_2_p_array = np.array([])
rouge_L_p_array = np.array([])
rouge_1_r_array = np.array([])
rouge_2_r_array = np.array([])
rouge_L_r_array = np.array([])
epoch_array = np.array([])


file_record = open(record_path,"r")
data_record = json.load(file_record)


for i in range(len(data_record)):
    try:            
        rouge_1_f_array =  np.append(rouge_1_f_array, data_record[i]["rouge_tw"]["rouge-1"]['f']*100)
        rouge_2_f_array =  np.append(rouge_2_f_array, data_record[i]["rouge_tw"]["rouge-2"]['f']*100)
        rouge_L_f_array =  np.append(rouge_L_f_array, data_record[i]["rouge_tw"]["rouge-l"]['f']*100)
        rouge_1_p_array =  np.append(rouge_1_p_array, data_record[i]["rouge_tw"]["rouge-1"]['p']*100)
        rouge_2_p_array =  np.append(rouge_2_p_array, data_record[i]["rouge_tw"]["rouge-2"]['p']*100)
        rouge_L_p_array =  np.append(rouge_L_p_array, data_record[i]["rouge_tw"]["rouge-l"]['p']*100)
        rouge_1_r_array =  np.append(rouge_1_r_array, data_record[i]["rouge_tw"]["rouge-1"]['r']*100)
        rouge_2_r_array =  np.append(rouge_2_r_array, data_record[i]["rouge_tw"]["rouge-2"]['r']*100)
        rouge_L_r_array =  np.append(rouge_L_r_array, data_record[i]["rouge_tw"]["rouge-l"]['r']*100)
        epoch_array =  np.append(epoch_array, i)
    except json.JSONDecodeError:
        # Handle lines that are not valid JSON, if necessary
        print(f"Skipping invalid JSON: {line}")



plt.figure(0)
plt.title('rouge_1_f',fontsize=24)        
plt.xlabel('epochs',fontsize=14)
plt.ylabel('accuracy',fontsize=14)
x_major_locator=MultipleLocator(10) 
ax=plt.gca() 
ax.plot(epoch_array, rouge_1_f_array)
ax.xaxis.set_major_locator(x_major_locator)
png_path = args.output_dir + '/rouge_1_f.png'
plt.savefig(png_path)

plt.figure(1)
plt.title('rouge_2_f',fontsize=24)        
plt.xlabel('epochs',fontsize=14)
plt.ylabel('accuracy',fontsize=14)
x_major_locator=MultipleLocator(10) 
ax=plt.gca() 
ax.plot(epoch_array, rouge_2_f_array)
ax.xaxis.set_major_locator(x_major_locator)

png_path = args.output_dir + '/rouge_2_f.png'
plt.savefig(png_path)

plt.figure(2)
plt.title('rouge_L_f',fontsize=24)        
plt.xlabel('epochs',fontsize=14)
plt.ylabel('accuracy',fontsize=14)
x_major_locator=MultipleLocator(10) 
ax=plt.gca() 
ax.plot(epoch_array, rouge_L_f_array)
ax.xaxis.set_major_locator(x_major_locator)
png_path = args.output_dir + '/rouge_L_f.png'
plt.savefig(png_path)

plt.figure(3)
plt.title('rouge_1_p',fontsize=24)        
plt.xlabel('epochs',fontsize=14)
plt.ylabel('accuracy',fontsize=14)
x_major_locator=MultipleLocator(10) 
ax=plt.gca() 
ax.plot(epoch_array, rouge_1_p_array)
ax.xaxis.set_major_locator(x_major_locator)
png_path = args.output_dir + '/rouge_1_p.png'
plt.savefig(png_path)

plt.figure(4)
plt.title('rouge_2_p',fontsize=24)        
plt.xlabel('epochs',fontsize=14)
plt.ylabel('accuracy',fontsize=14)
x_major_locator=MultipleLocator(10) 
ax=plt.gca() 
ax.plot(epoch_array, rouge_2_p_array)
ax.xaxis.set_major_locator(x_major_locator)
png_path = args.output_dir + '/rouge_2_p.png'
plt.savefig(png_path)

plt.figure(5)
plt.title('rouge_L_p',fontsize=24)        
plt.xlabel('epochs',fontsize=14)
plt.ylabel('accuracy',fontsize=14)
x_major_locator=MultipleLocator(10) 
ax=plt.gca() 
ax.plot(epoch_array, rouge_L_p_array)
ax.xaxis.set_major_locator(x_major_locator)
png_path = args.output_dir + '/rouge_L_p.png'
plt.savefig(png_path)

plt.figure(6)
plt.title('rouge_1_r',fontsize=24)        
plt.xlabel('epochs',fontsize=14)
plt.ylabel('accuracy',fontsize=14)
x_major_locator=MultipleLocator(10) 
ax=plt.gca() 
ax.plot(epoch_array, rouge_1_r_array)
ax.xaxis.set_major_locator(x_major_locator)
png_path = args.output_dir + '/rouge_1_r.png'
plt.savefig(png_path)

plt.figure(7)
plt.title('rouge_2_r',fontsize=24)        
plt.xlabel('epochs',fontsize=14)
plt.ylabel('accuracy',fontsize=14)
x_major_locator=MultipleLocator(10) 
ax=plt.gca() 
ax.plot(epoch_array, rouge_2_r_array)
ax.xaxis.set_major_locator(x_major_locator)
png_path = args.output_dir + '/rouge_2_r.png'
plt.savefig(png_path)

plt.figure(8)
plt.title('rouge_L_r',fontsize=24)        
plt.xlabel('epochs',fontsize=14)
plt.ylabel('accuracy',fontsize=14)
x_major_locator=MultipleLocator(10) 
ax=plt.gca() 
ax.plot(epoch_array, rouge_L_r_array)
ax.xaxis.set_major_locator(x_major_locator)
png_path = args.output_dir + '/rouge_L_r.png'
plt.savefig(png_path)


# Create a 3x3 grid of subplots
fig, axs = plt.subplots(3, 3, figsize=(15, 15))

# List of data arrays
data_arrays = [rouge_1_p_array, rouge_1_r_array, rouge_1_f_array,
               rouge_2_p_array, rouge_2_r_array, rouge_2_f_array,
               rouge_L_p_array, rouge_L_r_array, rouge_L_f_array]

# List of titles
titles = ['rouge_1_p', 'rouge_1_r', 'rouge_1_f',
          'rouge_2_p', 'rouge_2_r', 'rouge_2_f',
          'rouge_L_p', 'rouge_L_r', 'rouge_L_f']

# Loop through subplots and data arrays
for i, ax in enumerate(axs.flat):
    ax.set_title(titles[i], fontsize=24)
    ax.set_xlabel('epochs', fontsize=14)
    ax.set_ylabel('accuracy', fontsize=14)
    x_major_locator = MultipleLocator(10)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.plot(epoch_array, data_arrays[i])

# Adjust spacing between subplots
plt.tight_layout()

# Save the combined figure
plt.savefig(args.output_dir + '/combined_rouge_plots.png')

# Show the combined figure (optional)
plt.show()