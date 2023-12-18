import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib.patches import Rectangle
mpl.rcParams['lines.linewidth'] = 2


class Visualizer():

    def __init__(
            self,
            track_handler,
            gg_handler,
            gg_abs_margin,
            params,
            params_opp,
            params_sp,
            zoom_on_ego,
            zoom_margin
    ):
        self.track_handler = track_handler
        self.gg_handler = gg_handler
        self.gg_abs_margin = gg_abs_margin
        self.params = params
        self.params_opp = params_opp
        self.params_sp = params_sp
        self.zoom_on_ego = zoom_on_ego
        self.zoom_margin = zoom_margin
        
        # figure
        plt.ion()
        self.fig = plt.figure()
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1], figure=self.fig)

        # axis of track subplot
        self.axis_track = plt.subplot(gs[:, 0])
        self.axis_track.set_xlabel('$x$ in $m$')
        self.axis_track.set_ylabel('$y$ in $m$')
        self.axis_track.set_aspect(aspect='equal', adjustable='datalim')
        self.axis_track.grid()

        left, right = self.track_handler.get_track_bounds()
        self.axis_track.plot(left[0, :], left[1, :], 'black')
        self.axis_track.plot(right[0, :], right[1, :], 'black')

        self.rl_line, = self.axis_track.plot([0], [0], 'g-', label='Racing Line')
        if params_opp['start_s_opponents']:
            self.opp_lines = []
            self.opp_rects = []
            for i in range(len(params_opp['start_s_opponents'])):
                if i==0:
                    opp_line, = self.axis_track.plot([0], [0], 'r-', label='Prediction')
                else:
                    opp_line, = self.axis_track.plot([0], [0], 'r-')
                opp_rect = Rectangle((0, 0), params['vehicle_params']['total_length'], params['vehicle_params']['total_width'], angle=0, color='r', alpha=0.8)
                self.axis_track.add_patch(opp_rect)
                self.opp_lines.append(opp_line)
                self.opp_rects.append(opp_rect)
        self.ego_line, = self.axis_track.plot([0], [0], 'b-', label='Trajectory')
        self.ego_rect = Rectangle((0, 0), params['vehicle_params']['total_length'], params['vehicle_params']['total_width'], angle=0, color='b', alpha=0.8)
        self.axis_track.add_patch(self.ego_rect)
        self.axis_track.legend()

        # axis of velocity subplot
        self.axis_vel = plt.subplot(gs[0, 1])
        self.axis_vel.set_xlabel('$t$ in $s$')
        self.axis_vel.set_ylabel('$v$ in $m/s$')
        self.axis_vel.set_xlim([-0.5, params_sp['horizon'] + 0.5])
        self.axis_vel.set_ylim([0.0, 100.0])
        self.axis_vel.grid()

        self.vel_rl_line, = self.axis_vel.plot([0], [0], 'g-', label='Racing Line')
        self.vel_line, = self.axis_vel.plot([0], [0], 'b-', label='Trajectory')
        self.axis_vel.legend(loc='lower right')

        # axis of gg subplot
        self.axis_gg = plt.subplot(gs[1:3, 1])
        self.axis_gg.set_xlabel(r"$\tilde{a}_\mathrm{y}$ in $m/s^2$")
        self.axis_gg.set_ylabel(r"$\tilde{a}_\mathrm{x}$ in $m/s^2$")
        self.axis_gg.set_xlim([-40, 40])
        self.axis_gg.set_ylim([-40, 20])
        self.axis_gg.grid()

        self.diamond_line, = self.axis_gg.plot([0], [0], color='m', label='Friction Limits')
        self.axy_marker, = self.axis_gg.plot([0], [0], 'o', color='r', markersize=10, label='Acceleration')
        self.axis_gg.legend(loc='lower right')


    def update(
            self,
            state,
            trajectory,
            racing_line,
            prediction
    ):  
        # track subplot
        xyz_rl = self.track_handler.sn2cartesian(racing_line['s'], racing_line['n'])
        self.rl_line.set_xdata(xyz_rl[:, 0])
        self.rl_line.set_ydata(xyz_rl[:, 1])

        t = trajectory['t']

        ego_2d_heading = self.track_handler.calc_2d_heading_from_chi(state["chi"], state["s"])
        self.ego_rect.set_x(state['x'] - self.params['vehicle_params']['total_length']/2 * np.cos(ego_2d_heading) + self.params['vehicle_params']['total_width']/2 * np.sin(ego_2d_heading))
        self.ego_rect.set_y(state['y'] - self.params['vehicle_params']['total_length']/2 * np.sin(ego_2d_heading) - self.params['vehicle_params']['total_width']/2 * np.cos(ego_2d_heading))
        self.ego_rect.set_angle(np.rad2deg(ego_2d_heading))

        self.ego_line.set_xdata(trajectory['x'])
        self.ego_line.set_ydata(trajectory['y'])

        if self.zoom_on_ego:
            self.axis_track.set_xlim([min(trajectory['x']) - self.zoom_margin, max(trajectory['x']) + self.zoom_margin])
            self.axis_track.set_ylim([min(trajectory['y']) - self.zoom_margin, max(trajectory['y']) + self.zoom_margin])

        if self.params_opp['start_s_opponents']:
            i = -1
            for i, pred in enumerate(prediction.values()):
                self.opp_lines[i].set_visible(True)
                self.opp_lines[i].set_xdata(pred['x'])
                self.opp_lines[i].set_ydata(pred['y'])
                self.opp_rects[i].set_visible(True)
                opp_chi = np.arctan2((pred["n"][1] - pred["n"][0]), (pred["s"][1] - pred["s"][0]))
                opp_2d_heading = self.track_handler.calc_2d_heading_from_chi(opp_chi, pred["s"])
                self.opp_rects[i].set_x(pred['x'][0] - self.params['vehicle_params']['total_length']/2 * np.cos(opp_2d_heading) + self.params['vehicle_params']['total_width']/2 * np.sin(opp_2d_heading))
                self.opp_rects[i].set_y(pred['y'][0] - self.params['vehicle_params']['total_length']/2 * np.sin(opp_2d_heading) - self.params['vehicle_params']['total_width']/2 * np.cos(opp_2d_heading))
                self.opp_rects[i].set_angle(np.rad2deg(opp_2d_heading))
            for j in range(i+1, len(self.params_opp['start_s_opponents'])):
                self.opp_lines[j].set_visible(False)
                self.opp_rects[j].set_visible(False)
        
        # velocity subplot
        self.vel_rl_line.set_xdata(racing_line['t'])
        self.vel_rl_line.set_ydata(racing_line['V'])
        self.vel_line.set_xdata(t)
        self.vel_line.set_ydata(trajectory['V'])

        # gg subplot
        ax_tilde, ay_tilde, g_tilde = self.track_handler.calc_apparent_accelerations_numpy(
            s=state['s'],
            V=max(state['V'], 1.0),
            n=state['n'],
            chi=state['chi'],
            ax=state['ax'],
            ay=state['ay']
        )

        gg_exponent, ax_min, ax_max, ay_max  =  np.squeeze(self.gg_handler.acc_interpolator(np.array([state['V'], g_tilde])))
        ay_array = np.linspace(-ay_max, ay_max, 100)
        ax_array = - ax_min * np.power(
            1.0 - np.power(np.abs(ay_array) / ay_max, gg_exponent),
            1.0 / gg_exponent,
        )
        self.diamond_line.set_xdata([ay_array[0]] + list(ay_array) + [ay_array[-1]] + [None] + [ay_array[0]] + list(ay_array) + [ay_array[-1]])
        self.diamond_line.set_ydata([0] + list(np.minimum(ax_array + self.gg_abs_margin, ax_max + self.gg_abs_margin)) + [0] + [None] + [0] + list(-(ax_array + self.gg_abs_margin)) + [0])

        self.axy_marker.set_xdata([float(ay_tilde)])
        self.axy_marker.set_ydata([float(ax_tilde)])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
