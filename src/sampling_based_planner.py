import numpy as np
from track3D import Track3D
import copy

g_earth = 9.81


class LocalSamplingPlanner():

    def __init__(
            self,
            params,
            track_handler,
            gg_handler
    ):
        self.vehicle_params = params['vehicle_params']
        self.track_handler = track_handler
        self.left_track_bounds, self.right_track_bounds = track_handler.get_track_bounds(margin=0.0) 
        self.gggv_handler = gg_handler

        self.trajectory = {}
        self.traj_cnt = 0

    def calc_trajectory(
            self, 
            state: dict,
            prediction: dict,
            raceline: dict,
            relative_generation: bool,
            n_samples: int,
            v_samples: int,
            horizon: float,
            num_samples: int,
            safety_distance: float,
            gg_abs_margin: float,
            friction_check_2d: bool,
            s_dot_min: float = 1.0,
            kappa_thr: float = 0.1,
            raceline_cost_weight: float = 0.1,
            velocity_cost_weight: float = 100.0,
            prediction_cost_weight: float = 5000.0,
            prediction_s_factor: float = 0.015,
            prediction_n_factor: float = 0.5,
    ):
        self.traj_cnt += 1

        # frenet state
        s_start = state['s']
        s_dot_start = max(s_dot_min, state['s_dot'])
        s_ddot_start = state['s_ddot']
        n_start = state['n']
        n_dot_start = state['n_dot']
        n_ddot_start = state['n_ddot']

        # postprocess raceline
        postprocessed_raceline = self.postprocess_raceline(
            raw_raceline=raceline,
            s_start=s_start,
            horizon=horizon,
            track_handler=self.track_handler
        )

        # curves are generated relative to raceline for better tracking of the raceline
        raceline_tendency_s = False
        if abs(raceline['s_dot'][0] - s_dot_start)/raceline['s_dot'][0] < 0.3 and relative_generation:
            raceline_tendency_s = True

        # time arrays
        t_vector = np.linspace(0.0, horizon, num_samples)
        t_array = np.tile(t_vector, (v_samples * n_samples, 1))

        # generate longitudinal curves
        s_array, s_dot_array, s_ddot_array, s_end_values, s_dot_end_values = self.generate_longitudinal_curves(
            track_handler=self.track_handler,
            s_start=s_start,
            s_dot_start=s_dot_start,
            s_ddot_start=s_ddot_start,
            s_dot_min=s_dot_min,
            t_array=t_array,
            v_samples=v_samples,
            n_samples=n_samples,
            postprocessed_raceline=postprocessed_raceline,
            horizon=horizon,
            raceline_tendency=raceline_tendency_s,
        )

        # generate lateral jerk-optimal curves
        raceline_tendency_n = False
        n_array, n_dot_array, n_ddot_array = self.generate_lateral_curves(
            track_handler=self.track_handler,
            s_array=s_array,
            s_dot_array=s_dot_array,
            s_ddot_array=s_ddot_array,
            s_end_values=s_end_values,
            s_dot_end_values=s_dot_end_values,
            n_start=n_start,
            n_dot_start=n_dot_start,
            n_ddot_start=n_ddot_start,
            t_array=t_array,
            n_samples=n_samples,
            postprocessed_raceline=postprocessed_raceline,
            safety_distance=safety_distance,
            raceline_tendency=raceline_tendency_n,
        )

        # generate lateral curves relative to racing line
        if relative_generation:
            raceline_tendency_n = True
            n_array2, n_dot_array2, n_ddot_array2 = self.generate_lateral_curves(
                track_handler=self.track_handler,
                s_array=s_array,
                s_dot_array=s_dot_array,
                s_ddot_array=s_ddot_array,
                s_end_values=s_end_values,
                s_dot_end_values=s_dot_end_values,
                n_start=n_start,
                n_dot_start=n_dot_start,
                n_ddot_start=n_ddot_start,
                t_array=t_array,
                n_samples=n_samples,
                postprocessed_raceline=postprocessed_raceline,
                safety_distance=safety_distance,
                raceline_tendency=raceline_tendency_n,
            )

            t_array = np.vstack((t_array, t_array))

            n_array = np.vstack((n_array, n_array2))
            n_dot_array = np.vstack((n_dot_array, n_dot_array2))
            n_ddot_array = np.vstack((n_ddot_array, n_ddot_array2))

            s_array = np.vstack((s_array, s_array))
            s_dot_array = np.vstack((s_dot_array, s_dot_array))
            s_ddot_array = np.vstack((s_ddot_array, s_ddot_array))
            s_end_values = np.hstack((s_end_values, s_end_values))
            s_dot_end_values = np.hstack((s_dot_end_values, s_dot_end_values))

        # transform frenet curves to velocity frame
        V_array, chi_array, ax_vf_array, ay_vf_array, kappa_array = \
            self.transform_to_velocity_frame(
                track_handler=self.track_handler,
                s_array=s_array,
                s_dot_array=s_dot_array,
                s_ddot_array=s_ddot_array,
                n_array=n_array,
                n_dot_array=n_dot_array,
                n_ddot_array=n_ddot_array,
            )

        # valid_array specifies which trajectories are valid in terms of feasibility. Initially all valid.
        factor = 2 if relative_generation else 1
        valid_array = np.ones(factor * n_samples * v_samples, dtype=bool)

        # checks modify the valid array. The order of the checks can have influence on the calculation time
        self.check_curvature(
            valid_array=valid_array,
            kappa=kappa_array,
            kappa_thr=kappa_thr,
        )

        self.check_path_collision(
            track_handler=self.track_handler,
            valid_array=valid_array,
            s_array=s_array,
            n_array=n_array,
            safety_distance=safety_distance
        )

        self.check_friction_limits(
            valid_array=valid_array,
            track_handler=self.track_handler,
            s_array=s_array,
            V_array=V_array,
            n_array=n_array,
            chi_array=chi_array,
            ax_array=ax_vf_array,
            ay_array=ay_vf_array,
            friction_check_2d=friction_check_2d,
            gg_abs_margin=gg_abs_margin
        )

        # choose best trajectory
        optimal_idx = self.get_optimal_trajectory_idx(
            valid_array=valid_array,
            track_handler=self.track_handler,
            s_array=s_array,
            n_array=n_array,
            t_array=t_array,
            V_array=V_array,
            raceline=raceline,
            prediction=prediction,
            raceline_cost_weight=raceline_cost_weight,
            velocity_cost_weight=velocity_cost_weight,
            prediction_cost_weight=prediction_cost_weight,
            prediction_s_factor=prediction_s_factor,
            prediction_n_factor=prediction_n_factor
        )

        # set planned trajectory
        self.trajectory.clear()
        self.trajectory['traj_cnt'] = self.traj_cnt
        self.trajectory["t"] = t_array[optimal_idx]
        self.trajectory["s"] = s_array[optimal_idx]
        self.trajectory["s_dot"] = s_dot_array[optimal_idx]
        self.trajectory["s_ddot"] = s_ddot_array[optimal_idx]
        self.trajectory["n"] = n_array[optimal_idx]
        self.trajectory["n_dot"] = n_dot_array[optimal_idx]
        self.trajectory["n_ddot"] = n_ddot_array[optimal_idx]
        self.trajectory["V"] = V_array[optimal_idx]
        self.trajectory["chi"] = chi_array[optimal_idx]
        self.trajectory["ax"] = ax_vf_array[optimal_idx]
        self.trajectory["ay"] = ay_vf_array[optimal_idx]
        self.trajectory["kappa"] = kappa_array[optimal_idx]
        xyz_array = self.track_handler.sn2cartesian(s=self.trajectory["s"], n=self.trajectory["n"])
        self.trajectory["x"] = xyz_array[:, 0]
        self.trajectory["y"] = xyz_array[:, 1]
        self.trajectory["z"] = xyz_array[:, 2]

        return self.trajectory


    def get_optimal_trajectory_idx(
            self,
            valid_array: np.ndarray,
            track_handler: Track3D,
            s_array: np.ndarray,
            n_array: np.ndarray,
            t_array: np.ndarray,
            V_array: np.ndarray,
            raceline: dict,
            prediction: dict,
            raceline_cost_weight: float,
            velocity_cost_weight: float,
            prediction_cost_weight: float,
            prediction_s_factor: float,
            prediction_n_factor: float
    ) -> int:
        velocity_cost_array = np.zeros_like(valid_array, dtype=float)
        raceline_cost_array = np.zeros_like(valid_array, dtype=float)
        prediction_cost_array = np.zeros_like(valid_array, dtype=float)

        # time difference array for integration
        diff_time_array = np.diff(t_array[valid_array], axis=1)

        # velocity costs
        V_raceline = np.interp(s_array[valid_array], raceline['s'], raceline['V'], period=track_handler.s[-1])
        velocity_cost_array[valid_array] = velocity_cost_weight * np.add.reduce(
            ((V_array[valid_array, :-1] - V_raceline[:, :-1]) / V_raceline[:, :-1]) ** 2 * diff_time_array,
            axis=1
        )

        # raceline costs
        raceline_deviation = np.interp(s_array[valid_array], raceline['s'], raceline['n'], period=track_handler.s[-1]) - n_array[valid_array]
        raceline_cost_array[valid_array] = raceline_cost_weight * np.add.reduce(
            raceline_deviation[:, :-1] ** 2 * diff_time_array,
            axis=1
        )

        # prediction costs
        for prediction_id in prediction:
            prediction_cur = prediction[prediction_id]

            s_prediction_cur = np.interp(t_array[valid_array], prediction_cur["t"], prediction_cur["s"])
            n_prediction_cur = np.interp(t_array[valid_array], prediction_cur["t"], prediction_cur["n"])
            raw_prediction_costs = np.exp(- prediction_s_factor * (s_array[valid_array] - s_prediction_cur) ** 2 - prediction_n_factor * (n_array[valid_array] - n_prediction_cur) ** 2)
            prediction_cost_array[valid_array] += prediction_cost_weight * np.add.reduce(raw_prediction_costs[:, :-1] * diff_time_array, axis=1)

        # overall costs
        cost_array = velocity_cost_array + raceline_cost_array + prediction_cost_array

        # return index of trajectory with the lowest cost
        opt_subset_idx = np.argmin(cost_array[valid_array])
        opt_idx = np.arange(cost_array.shape[0])[valid_array][opt_subset_idx]
        return opt_idx

    def check_friction_limits(
            self,
            valid_array: np.ndarray,
            track_handler,
            s_array: np.ndarray,
            V_array: np.ndarray,
            n_array: np.ndarray,
            chi_array: np.ndarray,
            ax_array: np.ndarray,
            ay_array: np.ndarray,
            friction_check_2d: bool,
            gg_abs_margin: float = 0.0,
            soft_check: bool = True
    ):
        ax_tilde = np.zeros_like(s_array)
        ay_tilde = np.zeros_like(s_array)
        g_tilde = np.zeros_like(s_array)

        if friction_check_2d:
            ax_tilde[valid_array] = ax_array[valid_array]
            ay_tilde[valid_array] = ay_array[valid_array]
            g_tilde[valid_array] = 9.81 * np.ones_like(s_array[valid_array])
        else:
            ax_tilde[valid_array], ay_tilde[valid_array], g_tilde[valid_array] = track_handler.calc_apparent_accelerations_numpy(
                s=s_array[valid_array],
                V=V_array[valid_array],
                n=n_array[valid_array],
                chi=chi_array[valid_array],
                ax=ax_array[valid_array],
                ay=ay_array[valid_array]
            )

        gg_exponent, ax_min, ax_max, ay_max = self.gggv_handler.acc_interpolator(
            np.array((V_array[valid_array].flatten(), g_tilde[valid_array].flatten()))
        ).full().squeeze().reshape(4, g_tilde[valid_array].shape[0], g_tilde[valid_array].shape[1])
        ax_avail = np.abs(ax_min) * np.power(
            np.maximum(
                (1.0 - np.power(
                    np.minimum(np.abs(ay_tilde[valid_array]) / ay_max, 1.0),
                    gg_exponent
                )),
                1e-3
            ),
            1.0 / gg_exponent
        )
        valid_tmp = np.all(np.abs(ay_tilde[valid_array]) <= ay_max + gg_abs_margin, axis=1) & \
                    np.all(np.abs(ax_tilde[valid_array]) <= ax_avail + gg_abs_margin, axis=1) & \
                    np.all(ax_tilde[valid_array] <= ax_max + gg_abs_margin, axis=1)
        if np.sum(valid_tmp) < 1:
            if soft_check:
                axy_exc = np.max(np.abs(ax_tilde[valid_array]) - ax_avail, axis=1)
                exc_min_idx = np.argmin(axy_exc)
                valid_tmp[exc_min_idx] = True      

        valid_array[valid_array] = valid_tmp

    def check_curvature(
            self,
            valid_array: np.ndarray,
            kappa: np.ndarray,
            kappa_thr: float,
            soft_check: bool = True
    ):
        valid_tmp = np.all(np.abs(kappa[valid_array]) <= kappa_thr, axis=1)
        if np.sum(valid_tmp) < 1:
            if soft_check:
                kappa_max = np.abs(kappa[valid_array]).max(axis=1)
                exc_min_idx = np.argmin(kappa_max)
                valid_tmp[exc_min_idx] = True

        valid_array[valid_array] = valid_tmp

    def check_path_collision(
            self,
            track_handler: Track3D,
            valid_array: np.ndarray,
            s_array: np.ndarray,
            n_array: np.ndarray,
            safety_distance: float,
            soft_check: bool = True
    ):
        left_bound = np.interp(s_array[valid_array], track_handler.s, track_handler.w_tr_left, period=track_handler.s[-1]) - self.vehicle_params['total_width'] / 2.0 - safety_distance
        right_bound = np.interp(s_array[valid_array], track_handler.s, track_handler.w_tr_right, period=track_handler.s[-1]) + self.vehicle_params['total_width'] / 2.0 + safety_distance
        valid_tmp = np.all((n_array[valid_array] < left_bound) & (n_array[valid_array] > right_bound), axis=1)
        if np.sum(valid_tmp) < 1:
            if soft_check:
                d_exc = np.maximum(np.max(n_array[valid_array] - left_bound, axis=1), np.max(right_bound - n_array[valid_array], axis=1))
                exc_min_idx = np.argmin(d_exc)
                valid_tmp[exc_min_idx] = True

        valid_array[valid_array] = valid_tmp

    def transform_to_velocity_frame(
            self,
            track_handler: Track3D,
            s_array: np.ndarray, 
            s_dot_array: np.ndarray, 
            s_ddot_array: np.ndarray,
            n_array: np.ndarray, 
            n_dot_array: np.ndarray, 
            n_ddot_array: np.ndarray,
    ):
        # angular velocity of road frame with respect to s expressed in road frame
        Omega_z_rf_array = np.interp(s_array, track_handler.s, track_handler.Omega_z, period=track_handler.s[-1])
        dOmega_z_rf_array = np.interp(s_array, track_handler.s, track_handler.dOmega_z, period=track_handler.s[-1])

        # absolute velocity
        v_array = np.sqrt((1.0 - Omega_z_rf_array * n_array) ** 2 * s_dot_array ** 2 + n_dot_array ** 2)

        # orientation of the velocity vector relative to reference line
        chi_array = np.arctan(n_dot_array / (s_dot_array * (1.0 - Omega_z_rf_array * n_array)))

        # x-acceleration in velocity frame
        ax_vf_array = 1 / np.sqrt(s_dot_array ** 2 * (1.0 - Omega_z_rf_array * n_array) ** 2 + n_dot_array ** 2) * \
                        (
                            s_dot_array * s_ddot_array * (1.0 - Omega_z_rf_array * n_array) ** 2
                            - s_dot_array ** 2 * (1.0 - Omega_z_rf_array * n_array) * (dOmega_z_rf_array * s_dot_array * n_array + Omega_z_rf_array * n_dot_array)
                            + n_dot_array * n_ddot_array
                        )

        # y-acceleration in velocity frame
        ay_vf_array = 1 / np.sqrt(s_dot_array ** 2 * (1.0 - Omega_z_rf_array * n_array) ** 2 + n_dot_array ** 2) * \
                        (
                            s_dot_array * n_dot_array * (dOmega_z_rf_array * s_dot_array * n_array + 2.0 * Omega_z_rf_array * n_dot_array)
                            - s_ddot_array * n_dot_array * (1.0 - Omega_z_rf_array * n_array)
                            + s_dot_array ** 3 * Omega_z_rf_array * (1 - Omega_z_rf_array * n_array) ** 2
                            + s_dot_array * n_ddot_array * (1.0 - Omega_z_rf_array * n_array)
                        )

        # angular velocity of velocity frame with respect to s
        kappa_array = s_dot_array / np.sqrt(s_dot_array ** 2 * (1.0 - Omega_z_rf_array * n_array) ** 2 + n_dot_array ** 2) * \
                            (
                                1.0 / s_dot_array * (
                                s_dot_array * (1.0 - Omega_z_rf_array * n_array) * n_ddot_array - n_dot_array * (
                                s_ddot_array * (1.0 - Omega_z_rf_array * n_array) - s_dot_array * (
                                dOmega_z_rf_array * s_dot_array * n_array + Omega_z_rf_array * n_dot_array))) /
                                (s_dot_array ** 2 * (1.0 - Omega_z_rf_array * n_array) ** 2 + n_dot_array ** 2) +
                                Omega_z_rf_array
                            )

        return v_array, chi_array, ax_vf_array, ay_vf_array, kappa_array
    
    def postprocess_raceline(
            self,
            raw_raceline: dict,
            s_start: float,
            horizon: float,
            track_handler: Track3D
    ):
        postprocessed_raceline = copy.deepcopy(raw_raceline)

        t_rl_raw = raw_raceline['t']
        n_rl_raw = raw_raceline['n']
        n_rl_dot_raw = raw_raceline['n_dot']
        n_rl_ddot_raw = raw_raceline['n_ddot']
        s_rl_raw = raw_raceline['s']
        s_rl_dot_raw = raw_raceline['s_dot']
        s_rl_ddot_raw = raw_raceline['s_ddot']
        v_rl_raw = raw_raceline['V']
        chi_rl_raw = raw_raceline['chi']
        ax_rl_raw = raw_raceline['ax']
        ay_rl_raw = raw_raceline['ay']

        # interpolate data points at s_start
        t_rl_start = np.interp(s_start, s_rl_raw, t_rl_raw, period=track_handler.s[-1])
        n_rl_start = np.interp(s_start, s_rl_raw, n_rl_raw, period=track_handler.s[-1])
        n_rl_dot_start = np.interp(s_start, s_rl_raw, n_rl_dot_raw, period=track_handler.s[-1])
        n_rl_ddot_start = np.interp(s_start, s_rl_raw, n_rl_ddot_raw, period=track_handler.s[-1])
        s_rl_dot_start = np.interp(s_start, s_rl_raw, s_rl_dot_raw, period=track_handler.s[-1])
        s_rl_ddot_start = np.interp(s_start, s_rl_raw, s_rl_ddot_raw, period=track_handler.s[-1])
        v_rl_start = np.interp(s_start, s_rl_raw, v_rl_raw, period=track_handler.s[-1])
        chi_rl_start = np.interp(s_start, s_rl_raw, chi_rl_raw, period=track_handler.s[-1])
        ax_rl_start = np.interp(s_start, s_rl_raw, ax_rl_raw, period=track_handler.s[-1])
        ay_rl_start = np.interp(s_start, s_rl_raw, ay_rl_raw, period=track_handler.s[-1])

        # insert data points at s_start into raceline
        rl_idx_start = np.searchsorted(t_rl_raw, t_rl_start)
        t_rl = np.insert(t_rl_raw, rl_idx_start, t_rl_start)
        n_rl = np.insert(n_rl_raw, rl_idx_start, n_rl_start)
        n_rl_dot = np.insert(n_rl_dot_raw, rl_idx_start, n_rl_dot_start)
        n_rl_ddot = np.insert(n_rl_ddot_raw, rl_idx_start, n_rl_ddot_start)
        s_rl = np.insert(s_rl_raw, rl_idx_start, s_start)
        s_rl_dot = np.insert(s_rl_dot_raw, rl_idx_start, s_rl_dot_start)
        s_rl_ddot = np.insert(s_rl_ddot_raw, rl_idx_start, s_rl_ddot_start)
        v_rl = np.insert(v_rl_raw, rl_idx_start, v_rl_start)
        chi_rl = np.insert(chi_rl_raw, rl_idx_start, chi_rl_start)
        ax_rl = np.insert(ax_rl_raw, rl_idx_start, ax_rl_start)
        ay_rl = np.insert(ay_rl_raw, rl_idx_start, ay_rl_start)

        # remove all points before data point at s_start
        t_rl = t_rl[rl_idx_start:]
        n_rl = n_rl[rl_idx_start:]
        n_rl_dot = n_rl_dot[rl_idx_start:]
        n_rl_ddot = n_rl_ddot[rl_idx_start:]
        s_rl = s_rl[rl_idx_start:]
        s_rl_dot = s_rl_dot[rl_idx_start:]
        s_rl_ddot = s_rl_ddot[rl_idx_start:]
        v_rl = v_rl[rl_idx_start:]
        chi_rl = chi_rl[rl_idx_start:]
        ax_rl = ax_rl[rl_idx_start:]
        ay_rl = ay_rl[rl_idx_start:]

        # shift data point at s_start to t=0
        t_rl = t_rl - t_rl[0]

        # remove all points greater planning horizon
        idxs = np.where(t_rl > horizon*1.5)
        t_rl = np.delete(t_rl, idxs[0][1:])
        n_rl = np.delete(n_rl, idxs[0][1:])
        n_rl_dot = np.delete(n_rl_dot, idxs[0][1:])
        n_rl_ddot = np.delete(n_rl_ddot, idxs[0][1:])
        s_rl = np.delete(s_rl, idxs[0][1:])
        s_rl_dot = np.delete(s_rl_dot, idxs[0][1:])
        s_rl_ddot = np.delete(s_rl_ddot, idxs[0][1:])
        v_rl = np.delete(v_rl, idxs[0][1:])
        chi_rl = np.delete(chi_rl, idxs[0][1:])
        ax_rl = np.delete(ax_rl, idxs[0][1:])
        ay_rl = np.delete(ay_rl, idxs[0][1:])

        # save postprocessed data into dictionary
        postprocessed_raceline['t_post'] = t_rl
        postprocessed_raceline['n_post'] = n_rl
        postprocessed_raceline['n_dot_post'] = n_rl_dot
        postprocessed_raceline['n_ddot_post'] = n_rl_ddot
        postprocessed_raceline['s_post'] = s_rl
        postprocessed_raceline['s_dot_post'] = s_rl_dot
        postprocessed_raceline['s_ddot_post'] = s_rl_ddot
        postprocessed_raceline['V_post'] = v_rl
        postprocessed_raceline['chi_post'] = chi_rl
        postprocessed_raceline['ax_post'] = ax_rl
        postprocessed_raceline['ay_post'] = ay_rl

        return postprocessed_raceline

    def generate_longitudinal_curves(
            self,
            track_handler: Track3D,
            s_start: float,
            s_dot_start: float,
            s_ddot_start: float,
            s_dot_min: float,
            t_array: np.ndarray,
            v_samples: int,
            n_samples: int,
            postprocessed_raceline: dict,
            horizon: float,
            raceline_tendency: bool,
    ):
        s_array = np.zeros_like(t_array)
        s_dot_array = np.zeros_like(t_array)
        s_ddot_array = np.zeros_like(t_array)

        # raceline end conditions
        s_dot_end_rl = np.interp(horizon, postprocessed_raceline['t_post'], postprocessed_raceline['s_dot_post'])
        s_ddot_end_rl = np.interp(horizon, postprocessed_raceline['t_post'], postprocessed_raceline['s_ddot_post'])

        # sampled s_dot end conditions
        s_dot_max = min(max(s_dot_start, s_dot_end_rl) * 1.2, self.gggv_handler.V_max)
        s_dot_end_values = np.concatenate((np.linspace(s_dot_min, s_dot_max, v_samples - 1), [s_dot_end_rl]))  # always sample raceline

        # end values of s and s_dot (needed for lateral curves)
        s_end_values = np.zeros_like(s_dot_end_values)

        for i, (s_dot_end, t_end) in enumerate(zip(s_dot_end_values, t_array[:, -1])):
            
            # set end acceleration between 0 and raceline acceleration dependent on sampled velocity
            s_ddot_end_tmp = np.interp(s_dot_end, [0.0, s_dot_end_rl], [0.0, s_ddot_end_rl])
            # only adhere to end acceleration of raceline when start velocity is also near to raceline velocity
            s_ddot_end = np.interp(s_dot_start, [0.0, postprocessed_raceline['s_dot_post'][0]], [0.0, s_ddot_end_tmp])

            # formulate linear system of equations
            a = np.array([[1, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0],
                          [0, 0, 2, 0, 0],
                          [0, 1, 2 * t_end, 3 * t_end ** 2, 4 * t_end ** 3],
                          [0, 0, 2, 6 * t_end, 12 * t_end ** 2]])
            if raceline_tendency:  # sample curves relative to raceline
                b = np.array([s_start-postprocessed_raceline['s_post'][0], s_dot_start-postprocessed_raceline['s_dot_post'][0], s_ddot_start-postprocessed_raceline['s_ddot_post'][0], s_dot_end-s_dot_end_rl, s_ddot_end-s_ddot_end_rl])
            else:  # sample curves absolute
                b = np.array([s_start, s_dot_start, s_ddot_start, s_dot_end, s_ddot_end])

            # calculate coefficients of quartic polynomial
            c = np.linalg.solve(a=a, b=b)
            
            # sampled s curve
            s_sample = c[0] + c[1] * t_array[i] + c[2] * t_array[i] ** 2 + c[3] * t_array[i] ** 3 + c[4] * t_array[i] ** 4
            s_dot_sample = c[1] + 2 * c[2] * t_array[i] + 3 * c[3] * t_array[i] ** 2 + 4 * c[4] * t_array[i] ** 3
            s_ddot_sample = 2 * c[2] + 6 * c[3] * t_array[i] + 12 * c[4] * t_array[i] ** 2

            if raceline_tendency:
                # evaluate raceline s data at t_array points
                s_continuous = np.unwrap(postprocessed_raceline['s_post'], discont=track_handler.s[-1]/2, period=track_handler.s[-1])
                s_rl_eval = np.mod(np.interp(t_array[i], postprocessed_raceline['t_post'], s_continuous), track_handler.s[-1])
                s_dot_rl_eval = np.interp(t_array[i], postprocessed_raceline['t_post'], postprocessed_raceline['s_dot_post'])
                s_ddot_rl_eval = np.interp(t_array[i], postprocessed_raceline['t_post'], postprocessed_raceline['s_ddot_post'])

                # add raceline s data to sampled relative s curve
                s = s_sample + s_rl_eval
                s_dot = s_dot_sample + s_dot_rl_eval
                s_ddot = s_ddot_sample + s_ddot_rl_eval
            else:
                s = s_sample
                s_dot = s_dot_sample
                s_ddot = s_ddot_sample

            # consider track length
            s = np.mod(s, track_handler.s[-1])

            # save last values
            s_end_values[i] = s[-1]

            s_array[i * n_samples:(i + 1) * n_samples, :] = np.tile(s, (n_samples, 1))
            s_dot_array[i * n_samples:(i + 1) * n_samples, :] = np.tile(s_dot, (n_samples, 1))
            s_ddot_array[i * n_samples:(i + 1) * n_samples, :] = np.tile(s_ddot, (n_samples, 1))

        return s_array, s_dot_array, s_ddot_array, s_end_values, s_dot_end_values

    def generate_lateral_curves(
            self,
            track_handler: Track3D,
            s_array: np.array,
            s_dot_array: np.array,
            s_ddot_array: np.array,
            s_end_values: np.array,
            s_dot_end_values: np.array,
            n_start: float,
            n_dot_start: float,
            n_ddot_start: float,
            t_array: np.ndarray,
            n_samples: int,
            postprocessed_raceline: dict,
            safety_distance: float,
            raceline_tendency: bool,
    ):

        n_array = np.zeros_like(t_array)
        n_dot_array = np.zeros_like(t_array)
        n_ddot_array = np.zeros_like(t_array)

        i = 0
        for s_end, s_dot_end in zip(s_end_values, s_dot_end_values):
            
            # evaluate raceline at specific s points
            s_dot_rl = np.interp(s_array[i], postprocessed_raceline['s_post'], postprocessed_raceline['s_dot_post'], period=track_handler.s[-1])
            s_ddot_rl = np.interp(s_array[i], postprocessed_raceline['s_post'], postprocessed_raceline['s_ddot_post'], period=track_handler.s[-1])
            n_rl = np.interp(s_array[i], postprocessed_raceline['s_post'], postprocessed_raceline['n_post'], period=track_handler.s[-1])
            n_dot_rl = np.interp(s_array[i], postprocessed_raceline['s_post'], postprocessed_raceline['n_dot_post'], period=track_handler.s[-1])
            n_ddot_rl = np.interp(s_array[i], postprocessed_raceline['s_post'], postprocessed_raceline['n_ddot_post'], period=track_handler.s[-1])

            n_rl_eval = n_rl
            n_dot_rl_eval = n_dot_rl / s_dot_rl * s_dot_array[i]
            n_ddot_rl_eval = n_ddot_rl / (s_dot_rl ** 2) * (s_dot_array[i] ** 2) \
                             - n_dot_rl / (s_dot_rl ** 3) * s_ddot_rl * (s_dot_array[i] ** 2) \
                             + n_dot_rl / s_dot_rl * s_ddot_array[i]

            # sampled n end conditions (relative to raceline)
            n_min_track = track_handler.w_tr_right_interpolator(s_end).full().squeeze()
            n_max_track = track_handler.w_tr_left_interpolator(s_end).full().squeeze()
            n_min = n_min_track + self.vehicle_params['total_width'] / 2.0 + safety_distance
            n_max = n_max_track - self.vehicle_params['total_width'] / 2.0 - safety_distance
            n_end_values = np.concatenate((np.linspace(n_min, n_max, n_samples - 1), [n_rl_eval[-1]]))  # always sample raceline

            # chi of track bounds at end position
            nearest_idx = (np.abs(track_handler.s - s_end)).argmin()
            next_idx = nearest_idx+1 if nearest_idx+1 < self.track_handler.s.size else 1  # not since start point and end point are the same

            left_bound_change_end = self.left_track_bounds[:, next_idx] - self.left_track_bounds[:, nearest_idx]
            right_bound_change_end = self.right_track_bounds[:, next_idx] - self.right_track_bounds[:, nearest_idx]
            referenceline_change_end = np.array([self.track_handler.x[next_idx] - self.track_handler.x[nearest_idx],
                                                 self.track_handler.y[next_idx] - self.track_handler.y[nearest_idx],
                                                 self.track_handler.z[next_idx] - self.track_handler.z[nearest_idx]])
            
            left_bound_change_end_normed = left_bound_change_end / np.linalg.norm(left_bound_change_end)
            right_bound_change_end_normed = right_bound_change_end / np.linalg.norm(right_bound_change_end)
            referenceline_change_end_normed = referenceline_change_end / np.linalg.norm(referenceline_change_end)

            chi_end_left_bound = np.arccos(np.dot(referenceline_change_end_normed, left_bound_change_end_normed))
            chi_end_right_bound = np.arccos(np.dot(referenceline_change_end_normed, right_bound_change_end_normed))
            
            cross_left = np.cross(referenceline_change_end_normed, left_bound_change_end_normed)
            cross_right = np.cross(referenceline_change_end_normed, right_bound_change_end_normed)
            normal_vector = track_handler.get_rotation_matrix_numpy(track_handler.theta[nearest_idx], track_handler.mu[nearest_idx], track_handler.phi[nearest_idx])[2]
            if np.dot(normal_vector, cross_left) < 0:
                chi_end_left_bound = -chi_end_left_bound
            if np.dot(normal_vector, cross_right) < 0:
                chi_end_right_bound = -chi_end_right_bound

            # chi of raceline at end position
            chi_end_rl = np.interp(s_end, postprocessed_raceline['s'], postprocessed_raceline['chi'], period=track_handler.s[-1])

            for n_end, t_end in zip(n_end_values, t_array[:, -1]):
                
                chi_end = np.interp(n_end, [n_min_track, n_rl_eval[-1], n_max_track], [chi_end_right_bound, chi_end_rl, chi_end_left_bound])
                n_dot_end = s_dot_end * np.tan(chi_end) * (1 - float(track_handler.Omega_z_interpolator(s_end))*n_end)

                # n_ddot at track boundaries should be zero and on the raceline it should correspond to the racing line
                n_ddot_end = np.interp(n_end, [n_min_track, n_rl_eval[-1], n_max_track], [0.0, n_ddot_rl_eval[-1], 0.0])
                
                # formulate linear system of equations
                a = np.array([[1, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0],
                              [0, 0, 2, 0, 0, 0],
                              [1, t_end, t_end ** 2, t_end ** 3, t_end ** 4, t_end ** 5],
                              [0, 1, 2 * t_end, 3 * t_end ** 2, 4 * t_end ** 3, 5 * t_end ** 4],
                              [0, 0, 2, 6 * t_end, 12 * t_end ** 2, 20 * t_end ** 3]])
                if raceline_tendency:  # sample curves relative to raceline
                    b = np.array([n_start-n_rl_eval[0], n_dot_start-n_dot_rl_eval[0], n_ddot_start-n_ddot_rl_eval[0], n_end-n_rl_eval[-1], n_dot_end-n_dot_rl_eval[-1], n_ddot_end-n_ddot_rl_eval[-1]])
                else:  # sample curves absolute
                    b = np.array([n_start, n_dot_start, n_ddot_start, n_end, n_dot_end, n_ddot_end])

                # calculate coefficients of quintic polynomial
                c = np.linalg.solve(a=a, b=b)

                # sampled n curve
                n_sample  = c[0] + c[1] * t_array[i] + c[2] * t_array[i] ** 2 + c[3] * t_array[i] ** 3 + c[4] * t_array[i] ** 4 + c[5] * t_array[i] ** 5
                n_dot_sample  = c[1] + 2 * c[2] * t_array[i] + 3 * c[3] * t_array[i] ** 2 + 4 * c[4] * t_array[i] ** 3 + 5 * c[5] * t_array[i] ** 4
                n_ddot_sample  = 2 * c[2] + 6 * c[3] * t_array[i] + 12 * c[4] * t_array[i] ** 2 + 20 * c[5] * t_array[i] ** 3

                if raceline_tendency:
                    # add raceline n data to sampled relative n curve
                    n = n_sample + n_rl_eval
                    n_dot = n_dot_sample + n_dot_rl_eval
                    n_ddot = n_ddot_sample + n_ddot_rl_eval
                else:
                    n = n_sample
                    n_dot = n_dot_sample
                    n_ddot = n_ddot_sample

                n_array[i, :] = n
                n_dot_array[i, :] = n_dot
                n_ddot_array[i, :] = n_ddot
                i += 1

        return n_array, n_dot_array, n_ddot_array
