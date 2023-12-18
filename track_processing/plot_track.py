import os
import sys

# settings
track_name = 'mpcb_3d_rl_as_ref_smoothed'
plot_3D = False

# paths
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, '..', 'data')
in_data_path = os.path.join(data_path, 'track_data_smoothed')
sys.path.append(os.path.join(dir_path, '..', 'src'))

from track3D import Track3D

track_handler = Track3D(path=os.path.join(in_data_path, track_name + '.csv'))

track_handler.visualize(threeD=plot_3D)