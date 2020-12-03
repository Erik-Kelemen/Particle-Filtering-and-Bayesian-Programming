"""
GUI for interacting with simulation

DO NOT EDIT THIS FILE!
"""

import sys
import argparse
import math
from tkinter import *
from PIL import Image, ImageTk
from utils import load_image
from simulator import Simulator, WORLD_WIDTH, WORLD_HEIGHT
from racetrack import RaceTrack, Contour, Horizontals, load_racetrack

MAX_COUNT_SINCE = 5


class App(Tk):
    TICK_RATE = 40

    def __init__(self, max_sensor_range=50, sensor_std=0.0, num_particles=50, gps_noise_var=10.0, gps_noise_width=20):
        Tk.__init__(self, None, baseName=None,
                    className='Tk', useTk=1, sync=0, use=None)
        self.draw_occupancy = False
        self.draw_particles = False
        self.__canvas = Canvas(self, width=WORLD_WIDTH, height=WORLD_HEIGHT)
        self.__canvas.pack()
        self.__canvas.configure(background="red")
        self.title("Simulator")
        self.bind("<KeyPress>", self.keydown)
        self.bind("<KeyRelease>", self.keyup)
        self.history_chars = []
        _, self.bg = load_image("data/track.png")
        
        img_car_blue, _ = load_image("data/car_blue.png")
        self.car_blue_imgs = []
        for i in range(-180,180):
            imgtk = ImageTk.PhotoImage(img_car_blue.rotate(i))
            self.car_blue_imgs.append(imgtk)

        img_green_arrow, _ = load_image("data/green_arrow.png")
        self.green_arrow_imgs = []
        for i in range(-180,180):
            imgtk = ImageTk.PhotoImage(img_green_arrow.rotate(i))
            self.green_arrow_imgs.append(imgtk)

        self.count_since = 0
        self.max_count_since = MAX_COUNT_SINCE

        self.simulator = Simulator(max_sensor_range=max_sensor_range, sensor_std=sensor_std, num_particles=num_particles, gps_noise_var=gps_noise_var, gps_noise_width=gps_noise_width)
    
    def keyup(self, e):
        if e.keysym in self.history_chars:
            self.history_chars.pop(self.history_chars.index(e.keysym))

    def keydown(self, e):
        if not e.keysym in self.history_chars:
            self.history_chars.append(e.keysym)

    def process_input(self):
        self.count_since += 1
        self.count_since = min(self.count_since, self.max_count_since)
        if not self.simulator.replaying:
            if "Up" in self.history_chars:
                self.simulator.car.throttle_press()
            if "Down" in self.history_chars:
                self.simulator.car.brake_press()
            if "Left" in self.history_chars:
                self.simulator.car.steer("left")
            if "Right" in self.history_chars:
                self.simulator.car.steer("right")
        if "p" in self.history_chars:
            if self.count_since >= self.max_count_since:
                self.simulator.toggle_particles()
                self.count_since = 0
        if "k" in self.history_chars:
            if self.count_since >= self.max_count_since:
                self.simulator.toggle_kalman()
                self.count_since = 0
        if "o" in self.history_chars:
            if self.count_since >= self.max_count_since:
                self.draw_occupancy = not self.draw_occupancy
                self.count_since = 0
        if "r" in self.history_chars:
            if self.count_since >= self.max_count_since:
                self.draw_particles = not self.draw_particles
                self.count_since = 0
        if "d" in self.history_chars:
            if self.count_since >= self.max_count_since:
                self.simulator.toggle_gps_noise_dist()
                self.count_since = 0
                print("GPS noise dist is now {}".format(self.simulator.gps_noise_dist))
    
    def __loop(self):
        #######################################################################################################################################
        # loop overhead
        self.after(App.TICK_RATE, self.__loop)
        #######################################################################################################################################
        # update simulator
        self.process_input()
        self.simulator.loop()
        
        #######################################################################################################################################
        # graphics
        self.__canvas.delete(ALL)
        self.__canvas.create_image(0,0,image=self.bg,anchor=NW,tag="bg")

        car = self.simulator.car
        racetrack = self.simulator.racetrack
        
        car_angle = int(math.degrees(math.atan2(-car.orient[1], car.orient[0])))
        self.__canvas.create_image(car.pos[0], car.pos[1], image=self.car_blue_imgs[car_angle + 180])

        # draw grid and occupancy
        if self.draw_occupancy:
            for i in range(80):
                self.__canvas.create_line(
                    0, 10 * i, 1400, 10 * i, fill="gray")
            for i in range(140):
                self.__canvas.create_line(
                    10 * i, 0, 10 * i, 800, fill="gray")
            for i in range(140):
                for j in range(80):
                    if racetrack.occupancy[i,j] != 0:
                        self.__canvas.create_rectangle(i * 10, j * 10, i * 10 + 10, j * 10 + 10, fill="black")

        sensor_color = "red"
        sensor_width = 2
        if self.draw_occupancy:
            dists = self.simulator.car.sensor_dists
            self.__canvas.create_line(car.pos[0], car.pos[1], car.pos[0], car.pos[1]-dists[0], fill=sensor_color, width=sensor_width)
            self.__canvas.create_line(car.pos[0], car.pos[1], car.pos[0], car.pos[1]+dists[1], fill=sensor_color, width=sensor_width)
            self.__canvas.create_line(car.pos[0], car.pos[1], car.pos[0]-dists[2], car.pos[1], fill=sensor_color, width=sensor_width)
            self.__canvas.create_line(car.pos[0], car.pos[1], car.pos[0]+dists[3], car.pos[1], fill=sensor_color, width=sensor_width)
        
        if self.draw_particles:
            if self.simulator.do_particle_filtering:
                for p in self.simulator.particle_filter.particles:
                    self.__canvas.create_oval(p.pos[0]-2,p.pos[1]-2,p.pos[0]+2,p.pos[1]+2,fill="red")
        
        if self.simulator.do_particle_filtering:
            angle_est = int(math.degrees(math.atan2(-self.simulator.orient_est[1], self.simulator.orient_est[0])))
            self.__canvas.create_image(self.simulator.x_est, self.simulator.y_est, image=self.green_arrow_imgs[angle_est + 180])
        
        if self.simulator.do_kalman_filtering:
            measx = self.simulator.car.gps_measurement[0]
            measy = self.simulator.car.gps_measurement[1]
            self.__canvas.create_oval(measx-4,measy-4,measx+4,measy+4,fill="red")
            kfx = self.simulator.kf_state[0]
            kfy = self.simulator.kf_state[1]
            self.__canvas.create_oval(kfx-4,kfy-4,kfx+4,kfy+4,fill="lime green")
        
        
    def mainloop(self, n=0):
        self.__loop()
        Tk.mainloop(self, n)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_particles", default=50, type=int, help='Number of particles for particle filtering')
    parser.add_argument("-m", "--max_sensor_range", default=50, type=int, help='Maximum range of the car\'s sensors')
    parser.add_argument("-s", "--sensor_noise_std", default=0.0, type=float, help='Std dev of car\'s sensor noise')
    parser.add_argument("-gv", "--gps_noise_var", default=10.0, type=float, help='Variance of gaussian noise for GPS measurement (Kalman filter)')
    parser.add_argument("-gw", "--gps_noise_width", default=20, type=float, help='Width of uniformly random noise for GPS measurement (Kalman filter)')
    args = parser.parse_args()

    if args.help:
        parser.print_help()
        sys.exit(0)

    max_sensor_range = args.max_sensor_range
    sensor_std = args.sensor_noise_std
    num_particles = args.num_particles
    gps_noise_var = args.gps_noise_var
    gps_noise_width = args.gps_noise_width

    print("Running GUI with\n    Num particles = {}\n    Max sensor range = {}\n    Sensor noise std = {}\n    GPS gaussian noise var={}\n    GPS uniform noise width={}".format(num_particles, max_sensor_range, sensor_std, gps_noise_var, gps_noise_width))

    app = App(max_sensor_range=max_sensor_range, sensor_std=sensor_std, num_particles=num_particles, gps_noise_var=gps_noise_var, gps_noise_width=gps_noise_width)
    app.mainloop()


if __name__ == "__main__":
    main()
