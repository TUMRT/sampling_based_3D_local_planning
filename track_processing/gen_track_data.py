import os
import sys

# settings
track_bounds = 'mpcb_bounds_3d'
track_data_output_file_name = 'mpcb_3d'

step_size = 2.0  # in meter
visualize = True
ignore_banking = False  # sets phi and mu to zero (rotation around x- and y-axis)

# Dictionary for cost function of track smoothing.
weights = {
    'w_c': 1e0,  # deviation to measurements centerline
    'w_l': 1e0,  # deviation to measurements left bound
    'w_r': 1e0,  # deviation to measurements right bound
    'w_theta': 1e7,  # smoothness theta
    'w_mu': 1e5,  # smoothness mu
    'w_phi': 1e4,  # smoothness phi
    'w_nl': 1e-2,  # smoothness left bound
    'w_nr': 1e-2  # smoothness right bound
}

# paths
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, '..', 'data')
track_bounds_path = os.path.join(data_path, 'track_bounds')
track_data_path = os.path.join(data_path, 'track_data')
track_data_smoothed_path = os.path.join(data_path, 'track_data_smoothed')
os.makedirs(track_data_path, exist_ok=True)
os.makedirs(track_data_smoothed_path, exist_ok=True)
sys.path.append(os.path.join(dir_path, '..', 'src'))

from track3D import Track3D

track_handler = Track3D()

# process data
track_handler.generate_3d_from_3d_track_bounds(
    path=os.path.join(track_bounds_path, track_bounds + '.csv'),
    out_path=os.path.join(track_data_path, track_data_output_file_name + '.csv'),
    reference=None,
    ignore_banking=ignore_banking,
    visualize=visualize
)

# smooth data
track_handler.smooth_track(
    out_path=os.path.join(track_data_smoothed_path, track_data_output_file_name + '_smoothed.csv'),
    weights=weights,
    in_path=os.path.join(track_data_path, track_data_output_file_name + '.csv'),
    visualize=True
)
