import matplotlib.pyplot as plt
import numpy as np

import sys

def parse_file(f):
    x = []
    y = []
    for line in f:
        if line.startswith("Total Time"):
            y.append(float(line.split()[2]))

    flag = False
    for line in f:
        if line.startswith("Total Workers"):
            st = "(" + line.split()[2]
            flag = True

        if flag and line.startswith("      CPU Cores"):
            st += "," + line.split()[2] + ")"
            x.append(st)
            flag = False
    return x,y

def parse_metric(f):
    y = []
    for line in f:
        if sys.argv[1] == 'Classification':
            if line.startswith("Accuracy"):
                y.append(float(line.split()[5])*100)
        elif sys.argv[1] == 'Regression':
            if line.startswith("Root"):
                y.append(float(line.split()[9]))
    return y

# function to add value labels
def addlabels_1(x,y):
    for i in range(len(x)):
        plt.text(i, y[i]+5, round(y[i],3), ha = 'center')

def addlabels_2(x,y):
    for i in range(len(x)):
        plt.text(i, y[i]+0.1, round(y[i],3), ha = 'center')

f1 = open(sys.argv[1]+'_Ray.txt','r').readlines()
f2 = open(sys.argv[1]+'_Spark.txt','r').readlines()

x1,y1 = parse_file(f1)
x2,y2 = parse_file(f2)

if (len(x1) < len(x2)):
    y1 += [0]*(len(x2)-len(x1))

x = x2

if sys.argv[1] == 'PageRank':
    x[2] = '(2,6/4)' # slight modification for PageRank

y = {
    'Ray': y1,
    'Spark': y2
}

x_t = np.arange(len(x))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

colors = ['salmon','royalblue']
col_idx = 0
for attribute, measurement in y.items():
    offset = width * multiplier
    rects = ax.bar(x_t + offset, measurement, width, label=attribute, color=colors[col_idx])
    ax.bar_label(rects, padding=3)
    multiplier += 1
    col_idx += 1

plt.title(f"Execution Time for {sys.argv[1]}")
plt.xlabel("Worker Info (num_workers, num_cores)")
plt.ylabel("Execution Time (s)")
plt.ylim(0,max(max(y1),max(y2))+100)
ax.set_xticks(x_t + width/2, x)
ax.legend(loc='upper right', ncols=2)

plt.show()

if sys.argv[1] == 'Classification':
    width = 0.25 
    multiplier = 0

    y1 = parse_metric(f1)

    if (len(x1) < len(x2)):
        y1 += [0]*(len(x2)-len(x1))
    
    y2 = parse_metric(f2)
    y = {
        'Ray': y1,
        'Spark': y2
    }

    fig, ax = plt.subplots(layout='constrained')

    colors = ['bisque','teal']
    col_idx = 0
    for attribute, measurement in y.items():
        offset = width * multiplier
        rects = ax.bar(x_t + offset, measurement, width, label=attribute, color=colors[col_idx])
        ax.bar_label(rects, padding=3)
        multiplier += 1
        col_idx += 1

    plt.title(f"Accuracy Score for {sys.argv[1]}")
    plt.xlabel("Worker Info (num_workers, num_cores)")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0,max(max(y1),max(y2))+30)
    ax.set_xticks(x_t + width/2, x)
    ax.legend(loc='upper left', ncols=2)

    plt.show()

elif sys.argv[1] == 'Regression':

    width = 0.25
    multiplier = 0

    y1 = parse_metric(f1)

    if (len(x1) < len(x2)):
        y1 += [0]*(len(x2)-len(x1))
    
    y2 = parse_metric(f2)

    y = {
        'Ray': y1,
        'Spark': y2
    }

    fig, ax = plt.subplots(layout='constrained')

    colors = ['bisque','teal']
    col_idx = 0
    for attribute, measurement in y.items():
        offset = width * multiplier
        rects = ax.bar(x_t + offset, measurement, width, label=attribute, color=colors[col_idx])
        ax.bar_label(rects, padding=3)
        multiplier += 1
        col_idx += 1

    plt.title(f"RMSE Score for {sys.argv[1]}")
    plt.xlabel("Worker Info (num_workers, num_cores)")
    plt.ylabel("RMSE")
    plt.ylim(0,max(max(y1),max(y2))+50)
    ax.set_xticks(x_t + width/2, x)
    ax.legend(loc='upper right', ncols=2)

    plt.show()

x = ['3','2']

if sys.argv[1] == 'Classification':
    y1 = [7.8, 12.3]
    y2 = [7.6,8.2]

if sys.argv[1] == 'Regression':
    y1 = [10.9, 0]
    y2 = [8.1,8.8]

if sys.argv[1] == 'Clustering':
    y1 = [10.6, 12.2]
    y2 = [5.1, 7]

if sys.argv[1] == 'PageRank':
    y1 = [7.9, 7.6]
    y2 = [9.8,9.8]

y = {
    'Ray': y1,
    'Spark': y2
}

x_t = np.arange(len(x))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

colors = ['skyblue','forestgreen']
col_idx = 0
for attribute, measurement in y.items():
    offset = width * multiplier
    rects = ax.bar(x_t + offset, measurement, width, label=attribute, color=colors[col_idx])
    ax.bar_label(rects, padding=3)
    multiplier += 1
    col_idx += 1

plt.title(f"Peak Memory Consumption for {sys.argv[1]}")
plt.xlabel("Number Of Workers")
plt.ylabel("Peak Memory Consumption (GB) per Worker")
plt.ylim(0,max(max(y1),max(y2))+5)
ax.set_xticks(x_t + width/2, x)
ax.legend(loc='upper right', ncols=2)
                       
plt.show()