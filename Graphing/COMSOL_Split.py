import matplotlib.pyplot as plt

# Initialize the style of the graph
plt.style.use('seaborn-v0_8-bright')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.grid'] = True  # Turn on gridlines
plt.rcParams['grid.color'] = 'gray'  # Set the color of the gridlines
plt.rcParams['grid.linestyle'] = '--'  # Set the style of the gridlines (e.g., dashed)
plt.rcParams['grid.linewidth'] = 0.5  # Set the width of the gridlines

# Setting up the figure object
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# Putting data from a .txt file of COMSOL simulation points into lists
# This function returns a list of positions and potentials as well as
# the minimum voltage
def build_data(file):
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

# Getting data from our COMSOL text files
arc_length1, potential1, v_min1 = build_data('GraphingData/COMSOLSplit1.txt')
arc_length2, potential2, v_min2 = build_data('GraphingData/COMSOLSplit2.txt')

# Establishing the bottom of the graph
v_min_final = min(v_min1, v_min2)

# Creating a list of potentials for the initial and final settings
potential1 = [point - v_min_final for point in potential1]
potential2 = [point - v_min_final for point in potential2]

# Building the plot
ax.clear()
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
