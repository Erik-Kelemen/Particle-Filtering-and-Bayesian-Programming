"""
Simulator representing the world with a racetrack and a car

DO NOT EDIT THIS FILE!
"""

import sys
import numpy as np
from car import Car
from particle_filter import ParticleFilter
from kalman_filter import KalmanFilter
from utils import angle_bw
from racetrack import load_racetrack

WORLD_WIDTH = 1400
WORLD_HEIGHT = 800

class Simulator:
    def __init__(self, max_sensor_range=50, sensor_std=0.0, num_particles=50, gps_noise_var=10.0, gps_noise_width=20):
        self.racetrack = load_racetrack("data/racetrack.p")
        self.car = Car(max_sensor_range=max_sensor_range, sensor_std=sensor_std, gps_noise_var=gps_noise_var, gps_noise_width=gps_noise_width)

        self.max_sensor_range = max_sensor_range
        self.sensor_std = sensor_std
        self.num_particles = num_particles
        self.do_particle_filtering = False
        self.particle_filter = None
        self.x_est = None
        self.y_est = None
        self.orient_est = None

        self.do_kalman_filtering = False
        self.kalman_filter = None
        self.kf_state = None
        self.gps_noise_var = gps_noise_var
        self.gps_noise_width = gps_noise_width
        self.gps_noise_dist = "gaussian"

        self.r_count = 0
        self.cur_rightness = 0.5

        self.lap_data = []
        self.crossed_start = False
        self.lap_data_old = np.load("data/lap_data.npy")
        self.cur_i = 2
        self.recording = False
        self.replaying = False
    
    def init_particles(self):
        self.do_particle_filtering = True
        self.particle_filter = ParticleFilter(self.num_particles, 0, WORLD_WIDTH, 0, WORLD_HEIGHT)
    
    def stop_particles(self):
        self.do_particle_filtering = False
        self.particle_filter = None
    
    def init_kalman(self):
        self.do_kalman_filtering = True
        self.kalman_filter = KalmanFilter(self.car.gps_noise_var)
    
    def stop_kalman(self):
        self.do_kalman_filtering = False
        self.kalman_filter = None
    
    def toggle_particles(self):
        if self.do_particle_filtering:
            self.stop_particles()
        else:
            self.init_particles()
    
    def toggle_kalman(self):
        if self.do_kalman_filtering:
            self.stop_kalman()
        else:
            self.init_kalman()
        
    def toggle_gps_noise_dist(self):
        if self.gps_noise_dist == "gaussian":
            self.gps_noise_dist = "uniform"
        elif self.gps_noise_dist == "uniform":
            self.gps_noise_dist = "gaussian"
        else:
            raise ValueError
    
    def toggle_replay(self):
        self.replaying = not self.replaying
    
    def loop(self):
        if self.recording:
            self.lap_data.append(np.append(self.car.pos, self.car.orient))
            datum = self.racetrack.progress(self.car)
            progress = datum[0]
            if 1.1 < progress and progress < 1.5 and not self.crossed_start:
                self.crossed_start = True
            if 0.5 < progress and progress < 1.0 and self.crossed_start:
                np.save("data/lap_data.npy", np.array(self.lap_data))
                print("finished")
                self.crossed_start = False
        if self.replaying:
            if self.cur_i == len(self.lap_data_old):
                print("replay finished")
                return False
            
            dp = self.lap_data_old[self.cur_i]
            pos = dp[0:2]
            orient = dp[2:]

            dp_1 = self.lap_data_old[self.cur_i - 1]
            pos_1 = dp_1[0:2]

            dp_1 = self.lap_data_old[self.cur_i - 2]
            pos_2 = dp_1[0:2]

            self.car.pos = pos
            self.car.vel = pos - pos_1
            self.car.old_vel = pos_1 - pos_2
            self.car.orient = orient

            self.cur_i += 1
        
        if not self.replaying:
            self.car.update(self.racetrack.contour_inner, self.racetrack.contour_outer)
        d_orient = angle_bw(self.car.vel, self.car.old_vel) * np.pi / 180.0
        
        #######################################################################################################################################
        # particle filtering
        self.car.measure_sensor_dists(self.racetrack)

        if self.do_particle_filtering:
            sensor = self.racetrack.read_distances
            max_sensor_range = self.car.max_sensor_range
            sensor_std = self.car.sensor_std
            evidence = self.car.sensor_dists
            delta_angle = d_orient
            speed = np.linalg.norm(self.car.vel)
            self.x_est, self.y_est, self.orient_est = self.particle_filter.filtering_and_estimation(sensor, max_sensor_range, sensor_std, evidence, delta_angle, speed)
        
        #######################################################################################################################################
        # kalman filter
        if self.do_kalman_filtering:
            self.gps_measurement = self.car.measure_gps(noise_dist=self.gps_noise_dist)
            self.kf_state = self.kalman_filter.get_state(self.gps_measurement)
        
        return True
