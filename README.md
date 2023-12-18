# Sampling Planning
This repository provides an implementation of the sampling-based local planning approach proposed in [[1]](#1). This includes: 

- the (offline) generation of gg-diagrams that depend on the total velocity $v$ and the apparent vertical acceleration $\tilde{g}$ according to [[3]](#3) and [[4]](#4)
- the (offline) underapproximation of the true gg-diagrams with diamond shapes according to [[2]](#2)
- the (offline) smoothing of the 3D track according to [[5]](#5)
- the (offline) generation of a global racing line according to [[4]](#4)
- the (online) generation of a local racing line according to [[2]](#2)
- the (online) sampling-based local trajectory generation according to [[1]](#1)

## Notes
- This repository does not intend to provide a directly applicable trajectory planner, installable Python package, or ready to use ROS node. It is rather a collection of methods and algorithms used in the paper [[1]](#1).
- All paths are relative. If you move a file you have to adapt the paths accordingly.
- Please cite our work if you use the provided code or parts of it ([Citing](#citing)). 

## Dependencies
All scripts have only been tested on Ubuntu 22.04.3 LTS with Python 3.10.12 and the package versions listed in [requirements.txt](requirements.txt).

1. Install Acados following https://docs.acados.org/installation/#linux-mac.

2. Install the Python interface following https://docs.acados.org/python_interface/index.html.

3. Install other used Python packages:
    ```
    pip install -r requirements.txt
    ```
## Quick Start
To reproduce the experiments shown in [[1]](#1), run the script:
```
python local_sampling_based/sim_sampling_based_planner.py
```
The desired experiment to be executed can be selected in this script. To customize the scenario, an own experiment file can be created in the folder [local_sampling_based/experiments](local_sampling_based/experiments).

## Comprehensive Workflow
The needed inputs for the quick start are already generated and are stored in the folder [data](data). To reproduce or adapt the inputs, the following steps need to be performed:
- Generation of the gg-diagrams
- Generation of a track file
- Generation of an offline racing line

The code for the generation of the inputs matches the implementation in https://github.com/TUMRT/online_3D_racing_line_planning, whereby necessary adjustments have been made for compatibility with the sampling-based planner. 

### 1. gg-Diagram Generation
The generation of the gg-diagrams for 3D tracks follows [[3]](#3) and [[4]](#4). To generate the gg-diagrams run the script:
```
python gg_diagram_generation/gen_gg_diagrams.py
```
This will create the polar-coordinate representation of the true gg-diagram shapes in the folder [data/gg_diagrams](data/gg_diagrams) for the vehicle and velocity frame.

To generate the diamond-shaped underapproximations of the gg-diagrams as introduced in [[2]](#2) run:
```
python gg_diagram_generation/gen_diamond_representation.py
```
The resulting lookup tables for both frames will be added to [data/gg_diagrams](data/gg_diagrams) as well.

You can visualize the gg-diagrams and its diamond-shaped underapproximations with:
```
python gg_diagram_generation/plot_gg_diagrams.py
```
Vehicle parameters can be changed or added in [data/vehicle_params](data/vehicle_params). If you add a vehicle or change the vehicle name you have to adapt the name in the above scripts accordingly.

### 2. Track Data with Center Line as Reference Line
To create a 3D track according to the representation in [[5]](#5), the track must be available as global $x$, $y$, and $z$ coordinates of track boundary pairs. Examples for the Mount Panorama Circuit in Bathurst and Las Vegas Motor Speedway are given with [data/track_bounds/mpcb_bounds_3d.csv](data/track_bounds/mpcb_bounds_3d.csv) and [data/track_bounds/lvms_bounds_3d.csv](data/track_bounds/lvms_bounds_3d.csv).

To generate the 3D track data from these track bounds run the script:
```
python track_processing/gen_track_data.py
```
This will create a track data file in [data/track_data](data/track_data) with the needed coordinates, euler angles, and angular velocities. As the generated 3D track data can be noisy, also a track smoothing according to [[5]](#5) is performed, which creates an additional track data file in [data/track_data_smoothed](data/track_data_smoothed).

You can visualize the final tracks and its angular information with:
```
python track_processing/plot_track.py
```

### 3. Offline Racing Line Generation
To generate the closed racing line around the track as introduced in [[2]](#2) run:
```
python global_racing_line/gen_global_racing_line.py
```
This will create a racing line file in [data/global_racing_lines](data/global_racing_lines). You can visualize the global racing line with:
```
python global_racing_line/plot_global_racing_line.py
```

### 4. Track Data with Racing Line as Reference Line
The generated track data files in step 2 will use the center line as the reference line of the track. To create a track file in which a desired racing line is used as the reference line as in [[1]](#1), two steps needs to be executed. First, new track bounds needs to be generated in [data/track_bounds](data/track_bounds) using the script:
```
python track_processing/gen_track_bounds_with_rl_as_ref.py
```
Next, run the following script to generate the new desired track data files in [data/track_data](data/track_data) and [data/track_data_smoothed](data/track_data_smoothed):
```
python track_processing/gen_track_data_with_rl_as_ref.py
```
With the new track data files a new offline racing line can be generated as in step 3. All files in which the racing line is used as the reference line have 'rl_as_ref' in their name.

## Citing
If you use the sampling-based local trajectory planning approach, please cite our work [[1]](#1). For the racing line generation, track smoothing procedure, and gg-diagram generation, we suggest to refer to [[2]](#2), [[3]](#3), [[4]](#4), [[5]](#5) respectively. 

## References
<a id="1">[1]</a> 
L. Ögretmen, M. Rowold, and B. Lohmann, “Sampling-Based Trajectory Planning with Online Racing Line Generation for Autonomous Driving on Three-Dimensional Race Tracks” in 2024 IEEE Intelligent Vehicles Symposium (IV). IEEE, 2023, pp. 1-8, submitted.

<a id="2">[2]</a> 
M. Rowold, L. Ögretmen, U. Kasolowsky and B. Lohmann, “Online Time-Optimal Trajectory Planning on
Three-Dimensional Race Tracks” in 2023 IEEE Intelligent Vehicles Symposium (IV). IEEE, 2023, pp. 1-8.

<a id="3">[3]</a> 
M. Veneri and M. Massaro, “A free-trajectory quasi-steady-state
optimal-control method for minimum lap-time of race vehicles”, Vehicle
System Dynamics, vol. 58, no. 6, pp. 933–954, 2020.

<a id="4">[4]</a> 
S. Lovato and M. Massaro, “A three-dimensional free-trajectory quasisteady-
state optimal-control method for minimum-lap-time of race vehicles”,
Vehicle System Dynamics, vol. 60, no. 5, pp. 1512–1530, 2022.

<a id="5">[5]</a> 
G. Perantoni and D. J. N. Limebeer, “Optimal Control of a Formula
One Car on a Three-Dimensional Track—Part 1: Track Modeling and
Identification”, Journal of Dynamic Systems, Measurement, and Control,
vol. 137, no. 5, 2015.
