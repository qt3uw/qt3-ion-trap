# Planar pollen trap instructional lab

This repository contains tracking programs, graphing scripts, and raw data for our QT3 Planar Trap demonstration lab. 
This project can perform three experiments:
* Experiment 1: Measure the micromotion amplitude, height, charge-to-mass ratio, and escape voltage of a single particle
  * Code use: Use "Tracking.py" to generate the Tuple.txt file containing the data points. The "push" function in "calibrate_pixel_distances.py" will then convert the data points into units of mm and store the new data file in a separate folder. Running the "pull" function of "calibrate_pixel_distances.py" over this new data folder will print group statistics.
* Experiment 2: Observe shuttling motion of a single particle and identify its maximum velocity
* Experiment 3: Observe the separation (splitting) motion of two particles and identify their maximum velocities

Example video data can be found in [this google shared drive](https://drive.google.com/drive/folders/1BFZFoC0MBmop-VNujvh46sj-NNSO6O-G?usp=drive_link).


## How to Plot
In our top level directory we have included the python scripts and text data files which were used to produce our plots for figures 2, 4, and 5. To plot with our "figures234.py" file, set the fname parameter to your desired file name such as "data/raw_micromotion/8-18_Trial18_data.txt". To plot with our "figure5.py" script, put your desired file name (e.g. "data/shuttle_experiment_data/COMSOLShuttle1.txt") in the build_data_comsol() function argument. This applies to each of the four plots produced by that script.



## Python Code Installation
These instructions assume you've installed [anaconda](https://www.anaconda.com/).  I also recommend the [pycharm community](https://www.jetbrains.com/pycharm/download) IDE for editing and debugging python code.  The instructions also assume that you know how to use the [command line "cd" command to change directory](https://www.digitalcitizen.life/command-prompt-how-use-basic-commands/).

Open a terminal (git bash if using windows, terminal on mac or linux). Navigate to the parent folder where you store your git repositories using the 'cd' command in the terminal.
Once there clone the repository and cd into it.
```
git clone https://github.com/qt3uw/qt3-ion-trap.git
cd qt3-ion-trap
```
You can use the .yml file contained in the repository to set up an anaconda environment with the required packages using the following command (if you are in windows, you will need to switch from the git bash terminal to the "anaconda prompt" terminal that can be found in start menu if you've installed anaconda, on a mac or linux you can use the good old terminal for everything):
```
conda env create -f environment.yml
```
This creates an anaconda environment called "qt3_ion_trap", which contains all of the python dependencies of this repository.  You can activate that environment with the following command:
```
conda activate qt3_ion_trap
```
Once activated your terminal will usually show (qt3_ion_trap) to the left of the command prompt.

Now that your terminal has activated the anaconda environment you can use pip to install this package in that environment.  cd into the parent folder that contains the qt3-ion-trap repository you cloned earlier.  Then use the following command to install the repository into the qt3_ion_trap conda environment.
```
pip install -e qt3_ion_trap
```

### Configure Interpreter in IDE
At this point is complete.  If using an IDE, like pycharm, you will need to make sure that the python interpreter for your project is set to the python.exe file for the anaconda environment that you just created.  An easy way to find the path to that python executable within the environment is to use the following command in a terminal where the qt3_ion_trap enviornment is activated:
```angular2html
where python
```
The path that we want is the middle line.  If using pycharm, follow [these instructions](https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html#view_list) to set your interpereter to that path.

## Micromotion Experiment Instructions ##
  Relevant code files for micromotion experiments are acquisition/micromotion_tracking.py, micromotion_data_analysis.py, and figures234.py.<br/><br/> micromotion_tracking.py evaluates raw .avi video files to detect a particle's location over time. These results are evaluated using the micromotion_data_analysis.py file to find charge-to-mass ratio, RF null height, and RF null voltage. Lastly, figures234.py can visualize these data and save the resulting figures. Follow the steps below to navigate full video analysis.

1. Collect a calibration image or video of a measuring tool in the viewframe. Use this to determine the pixel-to-millimeter conversion, using the following steps if necessary. Do not move the optical setup beyond this point.
2. Place your video file into the acquisition directory, then open tracking.py and change the class variable "self.VIDEO_FILE" to the desired .avi file.
3. Ensure all parameters within the "TrackingConfig" class reflect those of your experiment.
4. Run the code. Use any letter or arrow key to progress through the frames one at a time.
5. Identify the desired object's location and change the "self.X_RANGE" and "self.Y_RANGE" class variables to fit the object in frame. Make sure the bottom of the frame is set to the central surface of the trap, as gathered from the calibration video. This ensures proper height data collection.
6. Once the viewframe is set, press the spacebar and the data will begin to collect automatically. Values will appear in the terminal of the form: <br/>
"Saved: {voltage}, {height}, {micromotion}; Completion: {percent completion}, {pixel number}". <br/>
The results are output to a .txt file of the same name as the video as lists for each data point in the form "[voltage, height, micromotion]".
7. "calibrate_pixel_distances.py" can be used to analyze the data. It will generate a file containing the list [charge to mass ratio (C/kg), RF null voltage (-V), RF null height (mm)].
8. "figures234py" can then be used to visualize the data via the "plot_height_and_micro" and "plot_c2m_hist" functions.


# LICENSE

[LICENSE](LICENSE)
