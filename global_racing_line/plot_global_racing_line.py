import os
import sys
import numpy as np
import yaml
import pandas as pd
from matplotlib import pyplot as plt

params = {
    'track_name': 'mpcb_3d_rl_as_ref_smoothed.csv',
    'raceline_name': 'mpcb_3d_rl_as_ref_dallaraAV21_gg_0.1.csv',
    'vehicle_name': 'dallaraAV21',
    'plot_3D': False,
    'gg_mode': 'diamond',
    'gg_margin': 0.1,
    'gg_abs_margin': 0.0
}

# paths
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, '..', 'data')
vehicle_params_path = os.path.join(data_path, 'vehicle_params', 'params_' + params['vehicle_name'] + '.yml')
gg_diagram_path = os.path.join(data_path, 'gg_diagrams', params['vehicle_name'], 'velocity_frame')
track_path = os.path.join(data_path, 'track_data_smoothed')
racing_line_path = os.path.join(data_path, 'global_racing_lines')
sys.path.append(os.path.join(dir_path, '..', 'src'))

# load vehicle and tire parameters
with open(vehicle_params_path, 'r') as stream:
    params.update(yaml.safe_load(stream))

from track3D import Track3D
from ggManager import GGManager


def visualize_trajectory(track_path, raceline_path):
    track_handler = Track3D(
        path=os.path.join(track_path)
    )

    normal_vector = track_handler.get_normal_vector_numpy(
        theta=track_handler.theta,
        mu=track_handler.mu,
        phi=track_handler.phi
    )

    gg_handler = GGManager(
        gg_path=gg_diagram_path,
        gg_margin=params['gg_margin']
    )

    trajectory_data_frame = pd.read_csv(raceline_path, sep=',')
    s_opt = trajectory_data_frame['s_opt'].to_numpy()
    v_opt = trajectory_data_frame['v_opt'].to_numpy()
    n_opt = trajectory_data_frame['n_opt'].to_numpy()
    chi_opt = trajectory_data_frame['chi_opt'].to_numpy()
    ax_opt = trajectory_data_frame['ax_opt'].to_numpy()
    ay_opt = trajectory_data_frame['ay_opt'].to_numpy()
    jx_opt = trajectory_data_frame['jx_opt'].to_numpy()
    jy_opt = trajectory_data_frame['jy_opt'].to_numpy()

    # path
    ax = track_handler.visualize(show=False, threeD=params['plot_3D'])
    ax.set_title(f'Racing line {os.path.split(raceline_path)[-1]}')
    if params['plot_3D']:
        ax.plot(track_handler.x + normal_vector[0] * n_opt, track_handler.y + normal_vector[1] * n_opt, track_handler.z + normal_vector[2] * n_opt, color='red')
        ax.plot(track_handler.x, track_handler.y, track_handler.z, '--', color='black')
    else:
        ax.plot(track_handler.x + normal_vector[0] * n_opt, track_handler.y + normal_vector[1] * n_opt, color='red')
        ax.plot(track_handler.x, track_handler.y, '--', color='black')

    # racing line
    fig, ax = plt.subplots(nrows=5, num='Racing line')
    ax[0].grid()
    ax[0].plot(s_opt, v_opt, label=r'$V$')
    ax[0].legend()
    ax[1].grid()
    ax[1].plot(s_opt, n_opt, label=r'$n$')
    ax[1].legend()
    ax[2].grid()
    ax[2].plot(s_opt, chi_opt, label=r'$\hat{\chi}$')
    ax[2].legend()
    ax[3].grid()
    ax[3].plot(s_opt, ax_opt, label=r'$\hat{a}_x$')
    ax[3].plot(s_opt, ay_opt, label=r'$\hat{a}_y$')
    ax[3].legend()
    ax[4].grid()
    ax[4].plot(s_opt, jx_opt, label=r'$\hat{j}_x$')
    ax[4].plot(s_opt, jy_opt, label=r'$\hat{j}_y$')
    ax[4].legend()

    # apparent accelerations
    ax_tilde, ay_tilde, g_tilde = track_handler.calc_apparent_accelerations(
        V=v_opt, n=n_opt, chi=chi_opt, ax=ax_opt, ay=ay_opt, s=s_opt, h=params['vehicle_params']['h']
    )

    fig, ax = plt.subplots(nrows=3, num='Apparent accelerations')
    ax[0].plot(s_opt, g_tilde, label=r'$\tilde{g}$')
    ax[0].grid()
    ax[0].legend()
    ax[1].plot(s_opt, ax_tilde, label=r'$\tilde{a}_\mathrm{x}$')
    ax[1].grid()
    ax[1].legend()
    ax[2].plot(s_opt, ay_tilde, label=r'$\tilde{a}_\mathrm{y}$')
    ax[2].grid()
    ax[2].legend()

    # acceleration usage for original racing line and finely interpolated racing line
    if params['gg_mode'] == 'diamond':
        ax_tilde, ay_tilde, g_tilde = track_handler.calc_apparent_accelerations_numpy(
                s=s_opt,
                V=v_opt,
                n=n_opt,
                chi=chi_opt,
                ax=ax_opt,
                ay=ay_opt
        )
        ax_tilde = np.expand_dims(ax_tilde, axis=1)
        ay_tilde = np.expand_dims(ay_tilde, axis=1)
        g_tilde = np.expand_dims(g_tilde, axis=1)
        gg_exponent, ax_min, ax_max, ay_max = gg_handler.acc_interpolator(
            np.array((v_opt.flatten(), g_tilde.flatten()))
        ).full().squeeze().reshape(4, g_tilde.shape[0], g_tilde.shape[1])
        ax_avail = np.abs(ax_min) * np.power(
            np.maximum(
                (1.0 - np.power(
                    np.minimum(np.abs(ay_tilde) / ay_max, 1.0),
                    gg_exponent
                )),
                1e-3
            ),
            1.0 / gg_exponent
        )

        s_opt_fine = np.linspace(s_opt[0], s_opt[-1], 100*len(s_opt))
        n_opt_fine = np.interp(s_opt_fine, s_opt, n_opt)
        v_opt_fine = np.interp(s_opt_fine, s_opt, v_opt)
        chi_opt_fine = np.interp(s_opt_fine, s_opt, chi_opt)
        ax_opt_fine = np.interp(s_opt_fine, s_opt, ax_opt)
        ay_opt_fine = np.interp(s_opt_fine, s_opt, ay_opt)
        ax_tilde_fine, ay_tilde_fine, g_tilde_fine = track_handler.calc_apparent_accelerations_numpy(
                s=s_opt_fine,
                V=v_opt_fine,
                n=n_opt_fine,
                chi=chi_opt_fine,
                ax=ax_opt_fine,
                ay=ay_opt_fine
        )
        ax_tilde_fine = np.expand_dims(ax_tilde_fine, axis=1)
        ay_tilde_fine = np.expand_dims(ay_tilde_fine, axis=1)
        g_tilde_fine = np.expand_dims(g_tilde_fine, axis=1)
        gg_exponent_fine, ax_min_fine, ax_max_fine, ay_max_fine = gg_handler.acc_interpolator(
            np.array((v_opt_fine.flatten(), g_tilde_fine.flatten()))
        ).full().squeeze().reshape(4, g_tilde_fine.shape[0], g_tilde_fine.shape[1])
        ax_avail_fine = np.abs(ax_min_fine) * np.power(
            np.maximum(
                (1.0 - np.power(
                    np.minimum(np.abs(ay_tilde_fine) / ay_max_fine, 1.0),
                    gg_exponent_fine
                )),
                1e-3
            ),
            1.0 / gg_exponent_fine
        )
        
        fig, ax = plt.subplots(nrows=3, ncols=2, num='Acceleration Usage')
        ax[0, 0].title.set_text('Original Racing Line')
        ax[0, 0].plot(s_opt, ax_tilde, label=r'$\tilde{a}_x$')
        ax[0, 0].plot(s_opt, ax_avail + params['gg_abs_margin'], label=r'$\tilde{a}_{x,available}$')
        ax[0, 0].plot(s_opt, ax_avail + params['gg_abs_margin'] - ax_tilde, label=r'$\tilde{a}_{x,available} - \tilde{a}_x$')
        ax[0, 0].grid()
        ax[0, 0].legend()
        ax[1, 0].plot(s_opt, ax_tilde, label=r'$\tilde{a}_x$')
        ax[1, 0].plot(s_opt, ax_max + params['gg_abs_margin'], label=r'$\tilde{a}_{x,max}$')
        ax[1, 0].plot(s_opt, ax_max + params['gg_abs_margin'] - ax_tilde, label=r'$\tilde{a}_{x,max} - \tilde{a}_x$')
        ax[1, 0].grid()
        ax[1, 0].legend()
        ax[2, 0].plot(s_opt, ay_tilde, label=r'$\tilde{a}_y$')
        ax[2, 0].plot(s_opt, ay_max + params['gg_abs_margin'], label=r'$\tilde{a}_{y,max}$')
        ax[2, 0].plot(s_opt, ay_max + params['gg_abs_margin'] - ay_tilde, label=r'$\tilde{a}_{y,max} - \tilde{a}_y$')
        ax[2, 0].grid()
        ax[2, 0].legend()

        ax[0, 1].title.set_text('Interpolated Racing Line')
        ax[0, 1].plot(s_opt_fine, ax_tilde_fine, label=r'$\tilde{a}_x$')
        ax[0, 1].plot(s_opt_fine, ax_avail_fine + params['gg_abs_margin'], label=r'$\tilde{a}_{x,available}$')
        ax[0, 1].plot(s_opt_fine, ax_avail_fine + params['gg_abs_margin'] - ax_tilde_fine, label=r'$\tilde{a}_{x,available} - \tilde{a}_x$')
        ax[0, 1].grid()
        ax[0, 1].legend()
        ax[1, 1].plot(s_opt_fine, ax_tilde_fine, label=r'$\tilde{a}_x$')
        ax[1, 1].plot(s_opt_fine, ax_max_fine + params['gg_abs_margin'], label=r'$\tilde{a}_{x,max}$')
        ax[1, 1].plot(s_opt_fine, ax_max_fine + params['gg_abs_margin'] - ax_tilde_fine, label=r'$\tilde{a}_{x,max} - \tilde{a}_x$')
        ax[1, 1].grid()
        ax[1, 1].legend()
        ax[2, 1].plot(s_opt_fine, ay_tilde_fine, label=r'$\tilde{a}_y$')
        ax[2, 1].plot(s_opt_fine, ay_max_fine + params['gg_abs_margin'], label=r'$\tilde{a}_{y,max}$')
        ax[2, 1].plot(s_opt_fine, ay_max_fine + params['gg_abs_margin'] - ay_tilde_fine, label=r'$\tilde{a}_{y,max} - \tilde{a}_y$')
        ax[2, 1].grid()
        ax[2, 1].legend()

    # Show plot.
    plt.show()

if __name__ == '__main__':

    track_handler = Track3D(
        path=os.path.join(track_path, params['track_name'])
    )
    visualize_trajectory(
        track_path=os.path.join(track_path, params['track_name']),
        raceline_path=os.path.join(racing_line_path, params['raceline_name'])
    )
