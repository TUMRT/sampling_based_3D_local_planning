import os
import sys
import numpy as np
import yaml
import time

# configuration -------------------------------------------------------------------------------------------------------
# visualizer configuration
visualize = True
zoom_on_ego = True
zoom_margin = 20.0

# choose one of the experiments below or create own experiment under experiments/
experiment = 'ex1_relative_trajectory_generation_mpcb'

# experiment 1
# experiment = 'ex1_relative_trajectory_generation_lvms'
# experiment = 'ex1_relative_trajectory_generation_mpcb'
# experiment = 'ex1_jerk_optimal_trajectory_generation_lvms'
# experiment = 'ex1_jerk_optimal_trajectory_generation_mpcb'

# experiment 2
# experiment = 'ex2_overtake_lvms_3d'
# experiment = 'ex2_overtake_lvms_2d'

# experiment 3
# experiment = 'ex3_online_racing_line_solo_lvms'
# experiment = 'ex3_online_racing_line_solo_mpcb'
# experiment = 'ex3_offline_racing_line_multi_lvms'
# experiment = 'ex3_online_racing_line_multi_lvms'
# experiment = 'ex3_offline_racing_line_multi_mpcb'
# experiment = 'ex3_online_racing_line_multi_mpcb'

# load data -----------------------------------------------------------------------------------------------------------
# experiment parameter
dir_path = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(dir_path, 'experiments/' + experiment + '.yml') , 'r') as stream:
    all_params = yaml.safe_load(stream)

params = all_params['params']
params_sp = all_params['params_sp']
params_rp = all_params['params_rp']
params_ini = all_params['params_ini']
params_opp = all_params['params_opp']

# paths
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
from local_racing_line_planner import LocalRacinglinePlanner
from point_mass_model import export_point_mass_ode_model
from global_racing_line_planner import GlobalRacinglinePlanner
from sampling_based_planner import LocalSamplingPlanner
from visualizer import Visualizer

# create instances ----------------------------------------------------------------------------------------------------
track_handler = Track3D(
        path=os.path.join(track_path, params['track_name'] + '.csv')
)

gg_handler_rl = GGManager(
    gg_path=gg_diagram_path,
    gg_margin=params_rp['gg_margin']
)

gg_handler_sp = GGManager(
    gg_path=gg_diagram_path,
    gg_margin=0.0
)

point_mass_model = export_point_mass_ode_model(
        vehicle_params=params['vehicle_params'],
        track_handler=track_handler,
        gg_handler=gg_handler_rl,
        optimization_horizon=params_rp['optimization_horizon']
)

local_raceline_planner = LocalRacinglinePlanner(
    params=params,
    track_handler=track_handler,
    gg_handler=gg_handler_rl,
    model=point_mass_model,
    optimization_horizon=params_rp['optimization_horizon']
)

global_raceline_planner = GlobalRacinglinePlanner(
    track_handler=track_handler,
    horizon=params_rp['optimization_horizon'],
    racing_line=os.path.join(racing_line_path, params_rp['offline_raceline_name'] + '.csv')
)

local_sampling_planner = LocalSamplingPlanner(
    params=params,
    track_handler=track_handler,
    gg_handler=gg_handler_sp
)

if visualize:
    visualizer = Visualizer(
        track_handler=track_handler,
        gg_handler=gg_handler_sp,
        gg_abs_margin=params_sp['gg_abs_margin'],
        params=params,
        params_opp=params_opp,
        params_sp=params_sp,
        zoom_on_ego=zoom_on_ego,
        zoom_margin=zoom_margin
    )


def create_opp_prediction(
        horizon: float,
        s_opp: float,
        n_opp: float,
        speed_opp: float,
        ego_x: float,
        ego_y: float,
        reference: bool,
        speed_mode: str,
        speed_scale_opp: float,
        pred_id: float,
        track_handler: float,
        global_raceline_planner: GlobalRacinglinePlanner
):
    opp_xyz = track_handler.sn2cartesian(s_opp, n_opp)

    # create prediction if within sensor range
    if (ego_x - opp_xyz[0]) ** 2 + (ego_y - opp_xyz[1]) ** 2 <= params_opp['sensor_range'] ** 2:

        N = 51
        time_array = np.linspace(0.0, horizon, N)

        # get longitudinal movement
        if speed_mode == 'constant':
            s_array = (s_opp + speed_opp * time_array) % track_handler.s[-1]
        elif speed_mode == 'raceline':
            glob_raceline = global_raceline_planner.calc_raceline(s=s_opp)
            if glob_raceline['s'][-1] > glob_raceline['s'][0]:
                s_array = np.interp(time_array, glob_raceline['t'], glob_raceline['s'][0] + speed_scale_opp * (glob_raceline['s'] - glob_raceline['s'][0]))
            else:
                glob_raceline_s_unrwap = np.unwrap(glob_raceline['s'], discont=track_handler.s[-1]/2, period=track_handler.s[-1])
                s_array = np.interp(time_array, glob_raceline['t'], glob_raceline_s_unrwap[0] + speed_scale_opp * (glob_raceline_s_unrwap - glob_raceline_s_unrwap[0])) % track_handler.s[-1]
        
        # get lateral movement
        if reference == 'center':
            centerline_n = (np.array(track_handler.w_tr_left_interpolator(s_array)).squeeze() + np.array(track_handler.w_tr_right_interpolator(s_array)).squeeze())/2
            n_array = n_opp + centerline_n 
        elif reference == 'raceline':
            glob_raceline = global_raceline_planner.calc_raceline(s=s_opp)
            n_array = n_opp + np.interp(s_array, glob_raceline['s'], glob_raceline['n'], period=track_handler.s[-1])
        
        # assume chi to 0
        chi_array = np.zeros(N)

        # get xy-positions
        xyz_array = track_handler.sn2cartesian(s_array, n_array)
        x_array = xyz_array[:, 0]
        y_array = xyz_array[:, 1]

        prediction_data = {}
        prediction_data[pred_id] = {}
        prediction_data[pred_id]['vehicle_id'] = pred_id
        prediction_data[pred_id]['t'] = time_array
        prediction_data[pred_id]['x'] = x_array
        prediction_data[pred_id]['y'] = y_array
        prediction_data[pred_id]['s'] = s_array
        prediction_data[pred_id]['n'] = n_array
        prediction_data[pred_id]['chi'] = chi_array
    else:
        prediction_data = {}
    return prediction_data

# create initial state estimation -------------------------------------------------------------------------------------
state = {}
state["time_ns"] = time.time_ns()
state['s'] = params_ini['start_s']
state['n'] = params_ini['start_n']
state['V'] = params_ini['start_V']
state['chi'] = params_ini['start_chi']
state['ax'] = params_ini['start_ax']
state['ay'] = params_ini['start_ay']

xyz = track_handler.sn2cartesian(params_ini['start_s'], params_ini['start_n'])
state['x'] = xyz[0]
state['y'] = xyz[1]
state['z'] = xyz[2]

Omega_z = np.interp(state['s'], track_handler.s, track_handler.Omega_z, period=track_handler.s[-1])
dOmega_z = np.interp(state['s'], track_handler.s, track_handler.dOmega_z, period=track_handler.s[-1])

state['s_dot'] = state['V'] * np.cos(state['chi']) / (1.0 - state['n'] * Omega_z)
state['n_dot'] = state['V'] * np.sin(state['chi'])
state['chi_dot'] = state['ay'] / state['V'] - Omega_z * state['s_dot']

state['s_ddot'] = (state['ax'] * np.cos(state['chi']) - state['V'] * np.sin(state['chi']) *
                              state['chi_dot']) / (1.0 - state['n'] * Omega_z) - \
                             (state['V'] * np.cos(state['chi']) * (- state['n_dot'] * Omega_z - state[
                                 'n'] * dOmega_z * state['s_dot'])) / (1.0 - state['n'] * Omega_z)**2
state['n_ddot'] = state['V'] * np.cos(state['chi']) * state['chi_dot'] + state['ax'] * \
                        np.sin(state['chi'])

# create initial prediction -------------------------------------------------------------------------------------------
pred_id = 0
prediction = {}
if params_opp['start_s_opponents']:
    s_opponents = np.array(params_opp['start_s_opponents'])
    n_opponents = np.array(params_opp['n_opponents'])
    speed_opponents = np.array(params_opp['speed_opponents'])
    speed_scale_opponents = np.array(params_opp['speed_scale_opponents'])
    if params_opp['speed_mode'] == 'constant':
        speed_scale_opponents = np.zeros(s_opponents.size)
    elif params_opp['speed_mode'] == 'raceline':
        speed_opponents = np.zeros(s_opponents.size)

    for (s_opp, n_opp, speed_opp, speed_scale_opp) in zip(s_opponents, n_opponents, speed_opponents, speed_scale_opponents):
        opp_prediction = create_opp_prediction(
            horizon=params_sp['horizon'],
            speed_opp=speed_opp,
            n_opp=n_opp,
            s_opp=s_opp,
            ego_x=state['x'],
            ego_y=state['y'],
            pred_id=pred_id,
            reference=params_opp['reference'],
            speed_mode=params_opp['speed_mode'],
            speed_scale_opp=speed_scale_opp,
            track_handler=track_handler,
            global_raceline_planner=global_raceline_planner
        )
        pred_id += 1
        prediction.update(opp_prediction)

# simulation ----------------------------------------------------------------------------------------------------------
racing_line = None
trajectory = None
lap = 0
lap_time = 0.0
s_prev = state['s']

while 1:
    # generate racing line
    if params_rp['mode'] == 'local':
        racing_line = local_raceline_planner.calc_raceline(
            s=state['s'],
            V=state['V'],
            n=state['n'],
            chi=state['chi'],
            ax=state['ax'],
            ay=state['ay'],
            safety_distance=params_rp['safety_distance'],
            prev_solution=racing_line
        )
    elif params_rp['mode'] == 'global':
        racing_line = global_raceline_planner.calc_raceline(
            s=state['s']
        )

    # generate locale trajectory
    trajectory = local_sampling_planner.calc_trajectory(
        state=state,
        prediction=prediction,
        relative_generation=params_sp['relative_generation'],
        n_samples=params_sp['n_samples'],
        v_samples=params_sp['v_samples'],
        num_samples=params_sp['num_samples'],
        raceline=racing_line,
        horizon=params_sp['horizon'],
        safety_distance=params_sp['safety_distance'],
        gg_abs_margin=params_sp['gg_abs_margin'],
        friction_check_2d=params_sp['friction_check_2d']
    )

    # plot ------------------------------------------------------------------------------------------------------------
    if visualize:
        visualizer.update(
            state=state,
            trajectory=trajectory,
            racing_line=racing_line,
            prediction=prediction
        )

    # move opponents and predict --------------------------------------------------------------------------------------
    prediction.clear()
    if params_opp['start_s_opponents']:
        for i, (s_opp, n_opp, speed_opp, speed_scale_opp) in enumerate(zip(s_opponents, n_opponents, speed_opponents, speed_scale_opponents)):
            # move opponent
            if params_opp['speed_mode'] == 'constant':
                s_opp = (s_opp + params['assumed_calc_time'] * speed_opp) % track_handler.s[-1]
            elif params_opp['speed_mode'] == 'raceline':
                glob_raceline = global_raceline_planner.calc_raceline(s=s_opp)
                if glob_raceline['s'][-1] > glob_raceline['s'][0]:
                    s_opp = np.interp(params['assumed_calc_time'], glob_raceline['t'], glob_raceline['s'][0] + speed_scale_opp * (glob_raceline['s'] - glob_raceline['s'][0]))
                else:
                    glob_raceline_s_unrwap = np.unwrap(glob_raceline['s'], discont=track_handler.s[-1]/2, period=track_handler.s[-1])
                    s_opp = np.interp(params['assumed_calc_time'], glob_raceline['t'], glob_raceline_s_unrwap[0] + speed_scale_opp * (glob_raceline_s_unrwap - glob_raceline_s_unrwap[0])) % track_handler.s[-1]        
            s_opponents[i] = s_opp

            # predict
            opp_prediction = create_opp_prediction(
                horizon=params_sp['horizon'],
                speed_opp=speed_opp,
                n_opp=n_opp,
                s_opp=s_opp,
                ego_x=state['x'],
                ego_y=state['y'],
                pred_id=pred_id,
                reference=params_opp['reference'],
                speed_mode=params_opp['speed_mode'],
                speed_scale_opp=speed_scale_opp,
                track_handler=track_handler,
                global_raceline_planner=global_raceline_planner           
            )
            pred_id += 1
            prediction.update(opp_prediction)

    # perfect tracking ------------------------------------------------------------------------------------------------
    s_array_unwrap = np.unwrap(
        trajectory['s'],
        discont=track_handler.s[-1] / 2.0,
        period=track_handler.s[-1]
    )
    state['s'] = np.interp(params['assumed_calc_time'], trajectory['t'], s_array_unwrap) % track_handler.s[-1]
    state['s_dot'] = np.interp(params['assumed_calc_time'], trajectory['t'], trajectory['s_dot'])
    state['s_ddot'] = np.interp(params['assumed_calc_time'], trajectory['t'], trajectory['s_ddot'])
    state['n'] = np.interp(params['assumed_calc_time'], trajectory['t'], trajectory['n'])
    state['n_dot'] = np.interp(params['assumed_calc_time'], trajectory['t'], trajectory['n_dot'])
    state['n_ddot'] = np.interp(params['assumed_calc_time'], trajectory['t'], trajectory['n_ddot'])
    state['V'] = np.interp(params['assumed_calc_time'], trajectory['t'], np.unwrap(trajectory['V']))
    state['chi'] = np.interp(params['assumed_calc_time'], trajectory['t'], trajectory['chi'])
    state['ax'] = np.interp(params['assumed_calc_time'], trajectory['t'], trajectory['ax'])
    state['ay'] = np.interp(params['assumed_calc_time'], trajectory['t'], trajectory['ay'])
    xyz = track_handler.sn2cartesian(state['s'], state['n'])
    state['x'] = xyz[0]
    state['y'] = xyz[1]
    state['z'] = xyz[2]
    state["time_ns"] = time.time_ns()

    # calculate lap time ----------------------------------------------------------------------------------------------
    if state['s'] > s_prev:
        lap_time += params['assumed_calc_time']
    else: # start finish line crossed
        time_remain = np.interp(
                track_handler.s[-1],
                [s_prev, state['s']],
                [0, params['assumed_calc_time']],
                period=track_handler.s[-1]
            )
        lap_time += time_remain
        print(f'Lap time in lap {lap}: {lap_time} s')
        lap_time = params['assumed_calc_time'] - time_remain
        lap += 1
    s_prev = state['s']

    print(f's = {state["s"]} | n = {state["n"]} |  V = {state["V"]}')
