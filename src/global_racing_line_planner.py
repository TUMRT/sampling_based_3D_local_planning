import numpy as np
import pandas as pd

class GlobalRacinglinePlanner():

    def __init__(
            self,
            track_handler,
            racing_line: str,
            horizon: float,
            N_steps: int = 150,
    ):
        self.track_handler = track_handler
        self.raceline_path = racing_line
        self.set_offline_raceline(racing_line)
        self.horizon = horizon
        self.N_steps = N_steps

    def set_offline_raceline(
            self,
            raceline_path: str
    ):
        rl_data_frame = pd.read_csv(raceline_path, comment='#', sep=',')
        self.s_offline_rl = rl_data_frame['s_opt'].to_numpy()
        self.v_offline_rl = rl_data_frame['v_opt'].to_numpy()
        self.n_offline_rl = rl_data_frame['n_opt'].to_numpy()
        self.chi_offline_rl = rl_data_frame['chi_opt'].to_numpy()
        self.ax_offline_rl = rl_data_frame['ax_opt'].to_numpy()
        self.ay_offline_rl = rl_data_frame['ay_opt'].to_numpy()
        self.t_offline_rl = rl_data_frame['t_opt'].to_numpy()
        self.jx_offline_rl = rl_data_frame['jx_opt'].to_numpy()
        self.jy_offline_rl = rl_data_frame['jy_opt'].to_numpy()

    def calc_raceline(
            self,
            s: float
    ):
        raceline = self.__gen_raceline(
            s=s,
            horizon=self.horizon
        )

        # calculate cartesian coordinates
        raceline['s'] = raceline['s'] % self.track_handler.s[-1]
        raceline_cartesian = self.track_handler.sn2cartesian(raceline['s'], raceline['n'])
        raceline['x'] = raceline_cartesian[:, 0]
        raceline['y'] = raceline_cartesian[:, 1]
        raceline['z'] = raceline_cartesian[:, 2]

        # calculate temporal derivatives
        Omega_z = np.interp(raceline['s'], self.track_handler.s, self.track_handler.Omega_z)
        dOmega_z = np.interp(raceline['s'], self.track_handler.s, self.track_handler.dOmega_z)

        raceline['s_dot'] = raceline['V'] * np.cos(raceline['chi']) / (1.0 - raceline['n'] * Omega_z)
        raceline['V_dot'] = raceline['ax']
        raceline['n_dot'] = raceline['V'] * np.sin(raceline['chi'])
        raceline['chi_dot'] = raceline['ay'] / raceline['V'] - Omega_z * raceline['s_dot']

        raceline['s_ddot'] = (raceline['V_dot'] * np.cos(raceline['chi']) - raceline['V'] * np.sin(raceline['chi']) *
                              raceline['chi_dot']) / (1.0 - raceline['n'] * Omega_z) - \
                             (raceline['V'] * np.cos(raceline['chi']) * (- raceline['n_dot'] * Omega_z - raceline[
                                 'n'] * dOmega_z * raceline['s_dot'])) / (1.0 - raceline['n'] * Omega_z)**2
        raceline['n_ddot'] = raceline['V'] * np.cos(raceline['chi']) * raceline['chi_dot'] + raceline['V_dot'] * \
                             np.sin(raceline['chi'])

        return raceline

    def __gen_raceline(
            self,
            s: float,
            horizon: float
    ):
        s_array = np.linspace(s, s + horizon, self.N_steps) 
        n_array = np.interp(s_array % self.track_handler.s[-1], self.s_offline_rl, self.n_offline_rl)
        V_array = np.interp(s_array % self.track_handler.s[-1], self.s_offline_rl, self.v_offline_rl)
        chi_array = np.interp(s_array % self.track_handler.s[-1], self.s_offline_rl, self.chi_offline_rl)
        ax_array = np.interp(s_array % self.track_handler.s[-1], self.s_offline_rl, self.ax_offline_rl)
        ay_array = np.interp(s_array % self.track_handler.s[-1], self.s_offline_rl, self.ay_offline_rl)
        t_array = np.interp(s_array % self.track_handler.s[-1], self.s_offline_rl, self.t_offline_rl)
        t_array -= t_array[0]
        t_array = np.unwrap(t_array, discont=max(t_array), period=self.t_offline_rl[-1])
        jx_array = np.interp(s_array % self.track_handler.s[-1], self.s_offline_rl, self.jx_offline_rl)
        jy_array = np.interp(s_array % self.track_handler.s[-1], self.s_offline_rl, self.jy_offline_rl)

        raceline = {
            's': s_array,
            'n': n_array,
            'V': V_array,
            'chi': chi_array,
            'ax': ax_array,
            'ay': ay_array,
            't': t_array,
            'jx': jx_array,
            'jy': jy_array,
            'epsilon_rho': np.zeros_like(s_array),
            'epsilon_V': np.zeros_like(s_array)
        }
        return raceline


# EOF
