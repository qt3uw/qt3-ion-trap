import matplotlib.pyplot as plt

# Initialize the style of the graph
plt.style.use('seaborn-v0_8-bright')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.grid'] = True  # Turn on gridlines
plt.rcParams['grid.color'] = 'gray'  # Set the color of the gridlines
plt.rcParams['grid.linestyle'] = '--'  # Set the style of the gridlines (e.g., dashed)
plt.rcParams['grid.linewidth'] = 0.5  # Set the width of the gridlines

# Setting up the graph object
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

# Puts data from our splitting txt file into lists after converting from
# pixels per frame to mm per second
# Returns: a position list, a time list, a max velocity position, a max
#          velocity time, and a max velocity
def build_data(file, frameOffset):
    graph_data = open(file, 'r').read()
    lines = graph_data.split('\n')
    maxVel = 0
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

# Getting the motion data from the text file
frameOffset = 100
position1, time1, maxVelPos1, maxVelTime1, maxVel1 = build_data('GraphingData/NewSplitData1.txt', frameOffset)
position2, time2, maxVelPos2, maxVelTime2, maxVel2 = build_data('GraphingData/NewSplitData2.txt', frameOffset)
print(maxVel1)
print(maxVel2)

# Building the graph
ax1.clear()
ax1.set_xlabel('Position (mm)', fontsize='x-large')
ax1.set_ylabel('Time (s)', fontsize='x-large')
ax1.scatter(position1, time1, s=3, marker='o', label='Ion 1 position data', c=[(33/255, 145/255, 140/255)])
ax1.set_yticks([0, 1, 2, 3, 4, 5, 6])
ax1.set_aspect(1.9)

# Bounding the grapph
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
