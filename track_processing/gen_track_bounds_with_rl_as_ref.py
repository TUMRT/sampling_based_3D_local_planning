import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from shapely import geometry

# settings
raceline = 'mpcb_3d_dallaraAV21_gg_0.1'  # the racing line which should be used as reference line
track_data_smoothed = 'mpcb_3d_smoothed'  # should be the track data which was used to generate the racing line above
track_bounds_output_file_name = 'mpcb_bounds_3d_rl_as_ref'
visualize = True

# paths
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, '..', 'data')
track_bounds_path = os.path.join(data_path, 'track_bounds')
track_data_smoothed_path = os.path.join(data_path, 'track_data_smoothed')
global_racing_lines_path = os.path.join(data_path, 'global_racing_lines')
os.makedirs(track_data_smoothed_path, exist_ok=True)
sys.path.append(os.path.join(dir_path, '..', 'src'))

# track handler
from track3D import Track3D

track_handler_smoothed = Track3D(
        path=os.path.join(track_data_smoothed_path, track_data_smoothed + '.csv')
    )
x_ref = track_handler_smoothed.x
y_ref = track_handler_smoothed.y
z_ref = track_handler_smoothed.z

# track bound equidistant
left, right = track_handler_smoothed.get_track_bounds()
track_bounds = {
    'left_bound_x': left[0, :],
    'left_bound_y': left[1, :],
    'left_bound_z': left[2, :],
    'right_bound_x': right[0, :],
    'right_bound_y': right[1, :],
    'right_bound_z': right[2, :]
}

# raceline 
rl_data_frame = pd.read_csv(os.path.join(global_racing_lines_path, raceline + '.csv'), comment='#', sep=',')
raceline = {
    's_opt': rl_data_frame['s_opt'].to_numpy(),
    'v_opt': rl_data_frame['v_opt'].to_numpy(),
    'n_opt': rl_data_frame['n_opt'].to_numpy(),
    'chi_opt': rl_data_frame['chi_opt'].to_numpy(),
    'ax_opt': rl_data_frame['ax_opt'].to_numpy(),
    'ay_opt': rl_data_frame['ay_opt'].to_numpy(),
    'jx_opt': rl_data_frame['jx_opt'].to_numpy(),
    'jy_opt': rl_data_frame['jy_opt'].to_numpy(),
}
xyz_rl = track_handler_smoothed.sn2cartesian(s=raceline["s_opt"], n=raceline["n_opt"])
x_rl = raceline['x_opt'] = xyz_rl[:, 0]
y_rl = raceline['y_opt'] = xyz_rl[:, 1]
z_rl = raceline['z_opt'] = xyz_rl[:, 2]

# get normal vectors of reference line
rotation_matrices_ref = track_handler_smoothed.get_rotation_matrix_numpy(track_handler_smoothed.theta, track_handler_smoothed.mu, track_handler_smoothed.phi).T
tangential_vectors_ref = rotation_matrices_ref[:, 0, :]
normal_vectors_ref = rotation_matrices_ref[:, 1, :]
orthogonal_vectors_ref = rotation_matrices_ref[:, 2, :]

# calculate normal vectors of raceline by rotation of normal vectors of reference line
rotation_matrices_chi = track_handler_smoothed.get_rotation_matrix_numpy(raceline['chi_opt'], np.zeros_like(raceline['chi_opt']), np.zeros_like(raceline['chi_opt'])).T
rotation_matrices_rl = np.matmul(rotation_matrices_chi, rotation_matrices_ref)
tangential_vectors_rl = rotation_matrices_rl[:, 0, :]
normal_vectors_rl = rotation_matrices_rl[:, 1, :]
orthogonal_vectors_rl = rotation_matrices_rl[:, 2, :]

# get intersection of normal vectors and track bounds
left_line = geometry.LineString(list(zip(track_bounds["left_bound_x"],track_bounds["left_bound_y"],track_bounds["left_bound_z"])))
right_line = geometry.LineString(list(zip(track_bounds["right_bound_x"],track_bounds["right_bound_y"],track_bounds["right_bound_z"])))

left_intersect_array = np.zeros((normal_vectors_rl.shape[0], 3))
right_intersect_array = np.zeros((normal_vectors_rl.shape[0], 3))
left_intersection_valid_array = np.ones(normal_vectors_rl.shape[0], dtype=bool)
right_intersection_valid_array = np.ones(normal_vectors_rl.shape[0], dtype=bool)
for i in range(0, len(x_rl)):
    normal_vec_line_left = geometry.LineString([[x_rl[i], y_rl[i], z_rl[i]], [x_rl[i] + normal_vectors_rl[i, 0]*20.0, y_rl[i] + normal_vectors_rl[i, 1]*20.0, z_rl[i] + normal_vectors_rl[i, 2]*20.0]])
    normal_vec_line_right = geometry.LineString([[x_rl[i], y_rl[i], z_rl[i]], [x_rl[i] - normal_vectors_rl[i, 0]*20.0, y_rl[i] - normal_vectors_rl[i, 1]*20.0, z_rl[i] - normal_vectors_rl[i, 2]*20.0]])
    left_intersect = normal_vec_line_left.intersection(left_line)
    right_intersect = normal_vec_line_right.intersection(right_line)
    if hasattr(left_intersect, 'x'):
        left_intersect_array[i] =  np.array([left_intersect.x, left_intersect.y, left_intersect.z])
    else:
        left_intersection_valid_array[i] = False
        print(f"Left intersection {i} not found. This intersection will be approximated.")
    if hasattr(right_intersect, 'x'):
        right_intersect_array[i] =  np.array([right_intersect.x, right_intersect.y, right_intersect.z])
    else:
        right_intersection_valid_array[i] = False
        print(f"Right intersection {i} not found. This intersection will be approximated.")

# estimate not found intersections using track width of original track file
orig_width_left = track_handler_smoothed.w_tr_left[~left_intersection_valid_array] - raceline['n_opt'][~left_intersection_valid_array]
orig_width_right = track_handler_smoothed.w_tr_right[~right_intersection_valid_array] - raceline['n_opt'][~right_intersection_valid_array]

left_missing_intersections_x = x_rl[~left_intersection_valid_array] + normal_vectors_rl[~left_intersection_valid_array, 0]*orig_width_left
left_missing_intersections_y = y_rl[~left_intersection_valid_array] + normal_vectors_rl[~left_intersection_valid_array, 1]*orig_width_left
left_missing_intersections_z = z_rl[~left_intersection_valid_array] + normal_vectors_rl[~left_intersection_valid_array, 2]*orig_width_left
left_intersect_array[~left_intersection_valid_array] = np.array([left_missing_intersections_x, left_missing_intersections_y, left_missing_intersections_z]).T

right_missing_intersections_x = x_rl[~right_intersection_valid_array] + normal_vectors_rl[~right_intersection_valid_array, 0]*orig_width_right
right_missing_intersections_y = y_rl[~right_intersection_valid_array] + normal_vectors_rl[~right_intersection_valid_array, 1]*orig_width_right
right_missing_intersections_z = z_rl[~right_intersection_valid_array] + normal_vectors_rl[~right_intersection_valid_array, 2]*orig_width_right
right_intersect_array[~right_intersection_valid_array] = np.array([right_missing_intersections_x, right_missing_intersections_y, right_missing_intersections_z]).T

# output file
track_bound_frame = pd.DataFrame()
track_bound_frame['right_bound_x'] = right_intersect_array[:, 0]
track_bound_frame['right_bound_y'] = right_intersect_array[:, 1]
track_bound_frame['right_bound_z'] = right_intersect_array[:, 2]
track_bound_frame['left_bound_x'] = left_intersect_array[:, 0]
track_bound_frame['left_bound_y'] = left_intersect_array[:, 1]
track_bound_frame['left_bound_z'] = left_intersect_array[:, 2]
track_bound_frame.to_csv(os.path.join(track_bounds_path, track_bounds_output_file_name + '.csv'), sep=',', index=False, float_format='%.6f')

# visualize
if visualize:   
    plt.figure()
    ax = plt.axes(projection='3d')

    # track bounds
    xyz_tr_bound_left = track_handler_smoothed.sn2cartesian(track_handler_smoothed.s, track_handler_smoothed.w_tr_left)
    xyz_tr_bound_right = track_handler_smoothed.sn2cartesian(track_handler_smoothed.s, track_handler_smoothed.w_tr_right)
    ax.plot3D(xyz_tr_bound_left[:, 0], xyz_tr_bound_left[:, 1], xyz_tr_bound_left[:, 2], 'x-', color='k')
    ax.plot3D(xyz_tr_bound_right[:, 0], xyz_tr_bound_right[:, 1], xyz_tr_bound_right[:, 2], 'x-', color='k')

    # new track bounds
    ax.plot3D(left_intersect_array[:, 0], left_intersect_array[:, 1], left_intersect_array[:, 2], 'x', color='g')
    ax.plot3D(right_intersect_array[:, 0], right_intersect_array[:, 1], right_intersect_array[:, 2], 'x', color='g')

    # reference line
    ax.plot3D(x_ref, y_ref, z_ref, 'b', label='Center Line')

    # raceline
    ax.plot3D(raceline['x_opt'], raceline['y_opt'], raceline['z_opt'], 'r', label='Racing Line')

    # vectors
    for i in range(0, len(x_ref), 50):
        ax.plot3D([x_ref[i], x_ref[i] + tangential_vectors_ref[i, 0]],
                  [y_ref[i], y_ref[i] + tangential_vectors_ref[i, 1]],
                  [z_ref[i], z_ref[i] + tangential_vectors_ref[i, 2]], 'k')
        ax.plot3D([x_ref[i], x_ref[i] + normal_vectors_ref[i, 0]],
                  [y_ref[i], y_ref[i] + normal_vectors_ref[i, 1]],
                  [z_ref[i], z_ref[i] + normal_vectors_ref[i, 2]], 'k')
        ax.plot3D([x_ref[i], x_ref[i] + orthogonal_vectors_ref[i, 0]],
                  [y_ref[i], y_ref[i] + orthogonal_vectors_ref[i, 1]],
                  [z_ref[i], z_ref[i] + orthogonal_vectors_ref[i, 2]], 'k')
        
        ax.plot3D([x_rl[i], x_rl[i] + tangential_vectors_rl[i, 0]],
                  [y_rl[i], y_rl[i] + tangential_vectors_rl[i, 1]],
                  [z_rl[i], z_rl[i] + tangential_vectors_rl[i, 2]], 'g')
        ax.plot3D([x_rl[i], x_rl[i] + normal_vectors_rl[i, 0]],
                  [y_rl[i], y_rl[i] + normal_vectors_rl[i, 1]],
                  [z_rl[i], z_rl[i] + normal_vectors_rl[i, 2]], 'g')
        ax.plot3D([x_rl[i], x_rl[i] + orthogonal_vectors_rl[i, 0]],
                  [y_rl[i], y_rl[i] + orthogonal_vectors_rl[i, 1]],
                  [z_rl[i], z_rl[i] + orthogonal_vectors_rl[i, 2]], 'g') 

    ax.set_box_aspect((np.ptp(xyz_tr_bound_right[:, 0]), np.ptp(xyz_tr_bound_right[:, 1]), np.ptp(xyz_tr_bound_right[:, 2])))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()

    plt.figure()
    ax = plt.axes()

    # track bounds
    ax.plot(xyz_tr_bound_left[:, 0], xyz_tr_bound_left[:, 1], 'x-', color='k')
    ax.plot(xyz_tr_bound_right[:, 0], xyz_tr_bound_right[:, 1], 'x-', color='k')

    # new track bounds
    ax.plot(left_intersect_array[:, 0], left_intersect_array[:, 1], 'x', color='g')
    ax.plot(right_intersect_array[:, 0], right_intersect_array[:, 1], 'x', color='g')
    plt.plot(left_intersect_array[0, 0], left_intersect_array[0, 1], 'ro')
    plt.plot(left_intersect_array[1, 0], left_intersect_array[1, 1], 'yo')
    plt.plot(left_intersect_array[2, 0], left_intersect_array[2, 1], 'go')
    plt.plot(right_intersect_array[0, 0], right_intersect_array[0, 1], 'ro')
    plt.plot(right_intersect_array[1, 0], right_intersect_array[1, 1], 'yo')
    plt.plot(right_intersect_array[2, 0], right_intersect_array[2, 1], 'go')

    # connection between new track bound pairs
    x_test = np.array([left_intersect_array[:, 0], right_intersect_array[:, 0]])
    y_test = np.array([left_intersect_array[:, 1], right_intersect_array[:, 1]])
    plt.plot(x_test, y_test)

    # reference line
    ax.plot(x_ref, y_ref, 'b', label='Reference Line')

    # raceline
    ax.plot(raceline['x_opt'], raceline['y_opt'], 'r', label='Racing Line')

    # vectors
    for i in range(0, len(x_ref), 50):
        ax.plot([x_ref[i], x_ref[i] + tangential_vectors_ref[i, 0]],
                [y_ref[i], y_ref[i] + tangential_vectors_ref[i, 1]], 'k')
        ax.plot([x_ref[i], x_ref[i] + normal_vectors_ref[i, 0]],
                [y_ref[i], y_ref[i] + normal_vectors_ref[i, 1]], 'k')
        
        ax.plot([x_rl[i], x_rl[i] + tangential_vectors_rl[i, 0]],
                [y_rl[i], y_rl[i] + tangential_vectors_rl[i, 1]], 'g')
        ax.plot([x_rl[i], x_rl[i] + normal_vectors_rl[i, 0]],
                [y_rl[i], y_rl[i] + normal_vectors_rl[i, 1]], 'g')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis('equal')
    ax.legend()

    plt.show()

print()