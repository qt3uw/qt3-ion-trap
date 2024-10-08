# This file contains all the graphing code for figure 5

import matplotlib.pyplot as plt

# Initialize the style of the graph
plt.style.use('seaborn-v0_8-bright')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.grid'] = True  # Turn on gridlines
plt.rcParams['grid.color'] = 'gray'  # Set the color of the gridlines
plt.rcParams['grid.linestyle'] = '--'  # Set the style of the gridlines (e.g., dashed)
plt.rcParams['grid.linewidth'] = 0.5  # Set the width of the gridlines

# Putting data from a .txt file of COMSOL simulation points into lists
# This function returns a list of positions and potentials as well as
# the minimum voltage
def build_data_comsol(file):
    graph_data = open(file, 'r').read()
    lines = graph_data.split('\n')
    v_min = float(lines[0].split()[1])
    arc_length = []
    potential = []
    for line in lines:
        if len(line) > 1:
            x, y = line.split()
            if float(y) < v_min:
                v_min = float(y)
            arc_length.append(1000*((float(x) - 0.06)))
            potential.append(float(y))

    return arc_length, potential, v_min


# Puts data from a txt file into lists after converting from
# pixels per frame to mm per second
# Returns: a position list, a time list, a max velocity position, a max
#          velocity time, and a max velocity
def build_data(file, frameOffset, firstPosType):
    graph_data = open(file, 'r').read()
    lines = graph_data.split('\n')
    maxVel = 0
    if firstPosType == 'zero':
        firstPos = 0
    elif firstPosType == 'average':
        firstPos = 0.07545 * float(lines[0].split(',')[1])
    maxVelPos = 0
    maxVelTime = 0
    position = []
    time = []
    for line in lines:
        if len(line) > 1:
            x, y = line.split(',')
            currentPos = 0.07545 * float(y)
            currentTime = 0.05 * (float(x) - frameOffset)
            secondPos = currentPos
            vel = abs(secondPos - firstPos) / 0.05
            if vel > maxVel:
                maxVel = vel
                maxVelPos = secondPos
                maxVelTime = currentTime
            firstPos = secondPos

            position.append(currentPos)
            time.append(currentTime)

    return position, time, maxVelPos, maxVelTime, maxVel

# COMSOL Shuttle ---------------------------------------------------------------------------------------------------------------------------

# Setting up the figure object
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# Getting data from our COMSOL text files
arc_length1, potential1, v_min1 = build_data_comsol('Data/Shuttle_Experiment_Data/COMSOLShuttle1.txt')
arc_length2, potential2, v_min2 = build_data_comsol('Data/Shuttle_Experiment_Data/COMSOLShuttle2.txt')

# Establishing the bottom of the graph
v_min_final = min(v_min1, v_min2)

# Creating a list of potentials for the initial and final settings
potential1 = [point - v_min_final for point in potential1]
potential2 = [point - v_min_final for point in potential2]

# Building the plot
ax.set_title('COMSOL Shuttling')
ax.set_xlabel('Position (mm)', fontsize='x-large')
ax.set_ylabel('Electric Potential (V)', fontsize='x-large')
ax.plot(arc_length1, potential1, marker='o', label='Initial', linewidth=1.5, markersize=0.25, c=(45/255, 178/255, 125/255))
ax.plot(arc_length2, potential2, marker='o', label='Final', linewidth=1.5, markersize=0.25, c=(68/255, 1/255, 84/255))

# Setting the y-axis ticks and bounding the graph
ax.set_yticks([0, 10, 20, 30, 40, 50])
ax.set_aspect(0.168)
plt.xlim([0, 25])
plt.ylim([-5, 50])

# Move the x-axis to the top
ax.xaxis.set_label_position('bottom')

# Set ticks on both the top and bottom of the plot
plt.tick_params(axis='x', which='both', top=True, bottom=True, labeltop=True, labelbottom=True, labelsize='large')
plt.tick_params(axis='y', labelsize='large')

# Place the legend at the bottom right
plt.legend(loc='upper right', fontsize='small')

plt.savefig('COMSOLShuttleGraph.pdf', format='pdf', bbox_inches='tight')

# Get the bounding box of the axes in display (pixel) coordinates
bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

# Get the width and height of the plot (axes) in inches
width, height = bbox.width, bbox.height
print(width)
print(height)
plt.show()

# COMSOL Split ---------------------------------------------------------------------------------------------------------------------------

# Setting up the figure object
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# Getting data from our COMSOL text files
arc_length1, potential1, v_min1 = build_data_comsol('Data/Shuttle_Experiment_Data/COMSOLSplit1.txt')
arc_length2, potential2, v_min2 = build_data_comsol('Data/Shuttle_Experiment_Data/COMSOLSplit2.txt')

# Establishing the bottom of the graph
v_min_final = min(v_min1, v_min2)

# Creating a list of potentials for the initial and final settings
potential1 = [point - v_min_final for point in potential1]
potential2 = [point - v_min_final for point in potential2]

# Building the plot
ax.clear()
ax.set_title('COMSOL Splitting')
ax.set_xlabel('Position (mm)', fontsize='x-large')
ax.set_ylabel('Electric Potential (V)', fontsize='x-large')
ax.plot(arc_length1, potential1, marker='o', label='Initial', linewidth=1.5, markersize=0.25, c=(45/255, 178/255, 125/255))
ax.plot(arc_length2, potential2, marker='o', label='Final', linewidth=1.5, markersize=0.25, c=(68/255, 1/255, 84/255))

# Setting the y-axis ticks and bounding the graph
ax.set_yticks([0, 10, 20, 30, 40, 50])
ax.set_aspect(0.4)
plt.xlim([-30, 30])
plt.ylim([-5, 50])

# Move the x-axis to the top
ax.xaxis.set_label_position('bottom')

# Set ticks on both the top and bottom of the plot
plt.tick_params(axis='x', which='both', top=True, bottom=True, labeltop=True, labelbottom=True, labelsize='large')
plt.tick_params(axis='y', labelsize='large')

# Place the legend at the bottom right
plt.legend(loc='upper right', fontsize='small')

plt.savefig('COMSOLSplitGraph.pdf', format='pdf', bbox_inches='tight')

# Get the bounding box of the axes in display (pixel) coordinates
bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

# Get the width and height of the plot (axes) in inches
width, height = bbox.width, bbox.height
print(width)
print(height)
plt.show()


# Shuttle ---------------------------------------------------------------------------------------------------------------------------

# Setting up the graph object
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)


# Getting the motion data from the text file
frameOffset = 100
position1, time1, maxVelPos1, maxVelTime1, maxVel = build_data('Data/Shuttle_Experiment_Data/TruncatedShuttleData.txt', frameOffset, 'zero')
print(maxVel)

# Building the graph
ax1.clear()
ax1.set_title('Shuttling')
ax1.set_xlabel('Position (mm)', fontsize='x-large')
ax1.set_ylabel('Time (s)', fontsize='x-large')
ax1.scatter(position1, time1, s=3, marker='o', label='Ion position data', c=[(31/255, 161/255, 135/255)])
ax1.set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
ax1.set_aspect(0.7)

# Bounding the graph
plt.xlim([0, 25])
plt.ylim([-0.5, 7])

# Building the graph
plt.scatter([maxVelPos1], [maxVelTime1], color='red', s=60, marker='D', edgecolor='black', label='Max velocity location')

# Move the x-axis to the top
ax1.xaxis.set_label_position('bottom')

# Set ticks on both the top and bottom of the plot
plt.tick_params(axis='x', which='both', top=True, bottom=True, labeltop=True, labelbottom=True, labelsize='large')
plt.tick_params(axis='y', labelsize='large')

# Place the legend at the bottom right
plt.legend(loc='lower right', fontsize='small')

plt.savefig('ShuttlePlot.pdf', format='pdf', bbox_inches='tight')
plt.show()


# Split ---------------------------------------------------------------------------------------------------------------------------

# Setting up the graph object
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

# Getting the motion data from the text file
frameOffset = 100
position1, time1, maxVelPos1, maxVelTime1, maxVel1 = build_data('Data/Shuttle_Experiment_Data/NewSplitData1.txt', frameOffset, 'average')
position2, time2, maxVelPos2, maxVelTime2, maxVel2 = build_data('Data/Shuttle_Experiment_Data/NewSplitData2.txt', frameOffset, 'average')
print(maxVel1)
print(maxVel2)

# Building the graph
ax1.clear()
ax1.set_title('Splitting')
ax1.set_xlabel('Position (mm)', fontsize='x-large')
ax1.set_ylabel('Time (s)', fontsize='x-large')
ax1.scatter(position1, time1, s=3, marker='o', label='Ion 1 position data', c=[(33/255, 145/255, 140/255)])
ax1.set_yticks([0, 1, 2, 3, 4, 5, 6])
ax1.set_aspect(1.9)

# Bounding the graph
plt.xlim([-30, 30])
plt.ylim([-0.5, 6])

# Building the graph
ax1.scatter(position2, time2, s=3, marker='o', label='Ion 2 position data', c=[(70/255, 50/255, 126/255)])
plt.scatter([maxVelPos1, maxVelPos2], [maxVelTime1, maxVelTime2], color='red', s=60, marker='D', edgecolor='black', label='Max velocity location')

# Move the x-axis to the top
ax1.xaxis.set_label_position('bottom')

# Set ticks on both the top and bottom of the plot
plt.tick_params(axis='x', which='both', top=True, bottom=True, labeltop=True, labelbottom=True, labelsize='large')
plt.tick_params(axis='y', labelsize='large')

# Place the legend at the bottom right
plt.legend(loc='lower right', fontsize='small')

plt.savefig('SplitPlot.pdf', format='pdf', bbox_inches='tight')
plt.show()
