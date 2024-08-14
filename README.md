# qt3-ion-trap
Code for the pollen trap instructional labs developed by QT3.


## Installation
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
On my windows machine in an anaconda prompt this command returns the following (the command itself is the top line):
```
(qt3_ion_trap) C:\Users\mfpars\repos>where python
C:\Users\mfpars\anaconda3\envs\qt3_ion_trap\python.exe
C:\Users\mfpars\AppData\Local\Microsoft\WindowsApps\python.exe
```
The path that we want is the middle line.  If using pycharm, follow [these instructions](https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html#view_list) to set your interpereter to that path.

###


# LICENSE

[LICENSE](LICENSE)
