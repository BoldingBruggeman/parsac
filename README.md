# AutoCalibration Python

Python-based auto-calibration tool. Expect configuration of model parameters through namelist or yaml files and model output in the form of NetCDF files.

Pronounced - A-C-Py


## Howto wheel

### To make a wheel

pip wheel .


### To install a wheel

pip install <--upgrade> acpy-0.1-py2-none-any.whl

Add options -v -v -v to see where files are installed

Add --user to install in default *user* directory


### To work with a wheel in edit mode

pip2 install -e . --user --upgrade
