## Tactip Time-Series Data
This repository contains YAML configuration files for collecting and processing tactip time-series data, Python scripts for simple analysis, and some sample data. This repository is intended to be used in conjunction with the `time-series` branch of `tactile-bench`.

## About the Structure

The overall folder structure follows the same convention as the tactile-data repository, with configuration files stored in the `cfg` folder and tactile data stored in the `tactile_data` folder. The main difference is that time-series data are organized into separate subfolders, with each subfolder corresponding to a single experimental run.

```text
experiment/
├── parameters.yaml
├── targets.csv
├── run_0/
│   ├── time_series_images/
│   ├── control_data.csv
│   ├── ft_sensor_data.csv
│   ├── image_data.csv
│   └── robot_data.csv
├── run_1/
├── run_2/
└── ...
```

### File Descriptions

* **`parameters.yaml`**
  A copy of the data collection settings used for the experiment.

* **`targets.csv`**
  A randomly generated sequence of target motions sampled from the specified `pose_lims`, `shear_lims`, and `speed_lims`. Each row corresponds to the commanded movement parameters for a single run.

* **`ft_sensor_data.csv`**
  Contains 3D force and 3D torque measurements recorded from the force-torque sensor. Timestamps are generated using Python's `time.perf_counter()`.

* **`robot_data.csv`**
  Contains robot state information recorded using the robot controller's internal timestamp.

* **`image_data.csv`**
  Maps image filenames to their corresponding acquisition timestamps, recorded using Python's `time.perf_counter()`.

<<<<<<< HEAD
* **`control_data.csv`**
  Timestamps for control events, e.g. recording start and end ('start', 'end'), shear start and end ('shear_start', 'shear_end').

=======
>>>>>>> 2d1f4cf (update)
* **`time_series_images/`**
  Directory containing the image sequence captured during the run.

## About Calibration

The folder `run_0` contains a short perpendicular press sequence used for timestamp synchronization between the different sensing modalities.

The subsequent folders (`run_1` to `run_i`) contain the actual data collection runs, where `i` is the total number of experimental samples.

In addition, three calibration datasets are provided, with folder names ending in `calibration-1`, `calibration-2`, and `calibration-3`:

### Calibration-1: Robot and Force-Torque Sensor Alignment

Performs controlled presses along the **x**, **y**, and **z** axes to validate the coordinate transformation between the robot frame and the force-torque sensor frame.

### Calibration-2: Image Coordinate Validation

Performs presses with variations in **x**, **y**, **z**, **Rx**, and **Ry** to validate the relationship between the robot coordinate system and the image coordinate system.

### Calibration-3: Sensor Surface Calibration

Performs rotational motions in **Rx** and **Ry** while keeping **x = y = z = 0**. This dataset is used to estimate and calibrate the surface orientation of the tactile sensor skin.
