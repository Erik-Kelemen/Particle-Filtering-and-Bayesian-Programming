"""
Racetrack in the simulator

DO NOT EDIT THIS FILE!
"""

import numpy as np
import pickle
from utils import angle_bw, add_noise

def load_racetrack(fname):
    racetrack = pickle.load(open(fname,"rb"))
    return racetrack


class Contour:
    def __init__(self):
        self.points = [np.array([0.0,0.0])]
        self.idx = 0
        self.moved = True

class Horizontals:
    def __init__(self):
        self.horizontals = [np.array([[0.0,0.0],[100.0,100.0]])]
        self.idx = 0
        self.pt = 0
        self.moved = True

class RaceTrack:
    def __init__(self):
        self.contour_inner = Contour()
        self.contour_outer = Contour()
        self.contour = 0
        self.occupancy = np.zeros((140,80))
        self.racing_line = Contour()
        self.finish_zone = np.array([[956,700],[1010,790]])
        self.split1 = np.array([[900,190],[950,280]])
        self.split2 = np.array([[20,300],[110,350]])
        self.horizontals = Horizontals()
    
    def is_car_in_box(self, car, box, direction):
        x = car.circle_front.pos[0] + car.circle_front.radius
        y = car.circle_front.pos[1] + car.circle_front.radius
        fx1 = box[0,0]
        fy1 = box[0,1]
        fx2 = box[1,0]
        fy2 = box[1,1]
        finished = fx1 <= x and x <= fx2 and fy1 <= y and y <= fy2
        if direction == "horizontal":
            dist = x - fx1
        elif direction == "vertical":
            dist = y - fy1
        else:
            raise ValueError
        return finished, dist
    
    def is_car_in_finish(self, car):
        return self.is_car_in_box(car, self.finish_zone, "horizontal")
    
    def is_car_in_split1(self, car):
        return self.is_car_in_box(car, self.split1, "horizontal")
    
    def is_car_in_split2(self, car):
        return self.is_car_in_box(car, self.split2, "vertical")
    
    def progress(self, car):
        pos = car.pos
        orient = car.orient
        speed = np.linalg.norm(car.vel)
        global MYTMP
        def does_proj(pt, horiz):
            ptmod = pt - horiz[0]
            vec = horiz[1] - horiz[0]
            unit_vec = vec / np.linalg.norm(vec)
            scalar = np.dot(ptmod, unit_vec)
            proj = scalar * unit_vec + horiz[0]
            minx = min(horiz[0,0],horiz[1,0])
            miny = min(horiz[0,1],horiz[1,1])
            maxx = max(horiz[0,0],horiz[1,0])
            maxy = max(horiz[0,1],horiz[1,1])

            tol = 10
            if minx - tol <= proj[0] and proj[0] <= maxx + tol and miny - tol <= proj[1] and proj[1] <= maxy + tol:
                return proj
            else:
                return None
        projs = []
        for i,h in enumerate(self.horizontals.horizontals):
            proj = does_proj(pos,h)
            projs.append((proj,h,i))
        
        best_dist1 = np.infty
        best_proj1 = None
        best_i1 = None
        best_h = None
        for p,h,i in projs:
            if p is not None:
                dist = np.linalg.norm(pos - p)
                if dist < best_dist1:# and angle_bw(pos - h[0], h[1] - h[0]) >= 0:
                    best_dist1 = dist
                    best_proj1 = p
                    best_h = h
                    best_i1 = i
        
        angle = angle_bw(pos - best_h[0], best_h[1] - best_h[0])
        best_i2 = None
        if angle >= 0:
            best_i2 = best_i1 + 1
        else:
            best_i2 = best_i1 - 1
        
        best_proj2 = projs[best_i2%len(projs)][0]

        dist_proj1 = best_dist1
        dist_proj2 = np.linalg.norm(pos - best_proj2)

        proj_behind = None
        proj_front = None
        dist_behind = None
        dist_front = None
        i_behind = None
        i_front = None

        if angle <= 0:
            proj_front = best_proj1
            proj_behind = best_proj2
            dist_front = dist_proj1
            dist_behind = dist_proj2
            i_front = best_i1
            i_behind = best_i2
        else:
            proj_front = best_proj2
            proj_behind = best_proj1
            dist_front = dist_proj2
            dist_behind = dist_proj1
            i_front = best_i2
            i_behind = best_i1
        
        h_front = self.horizontals.horizontals[i_front%len(projs)]
        h_behind = self.horizontals.horizontals[i_behind%len(projs)]

        dist_right_front = np.linalg.norm(proj_front - h_front[0])
        dist_left_front = np.linalg.norm(proj_front - h_front[1])
        dist_right_behind = np.linalg.norm(proj_behind - h_behind[0])
        dist_left_behind = np.linalg.norm(proj_behind - h_behind[1])
        
        weight_front = dist_behind / (dist_front + dist_behind)
        weight_behind = dist_front / (dist_front + dist_behind)

        if i_front == 0:
            i_behind = len(projs) - 1
            i_front = len(projs)
        progress = weight_front * i_front + weight_behind * i_behind
        
        rightness_front = dist_left_front / (dist_left_front + dist_right_front)
        rightness_behind = dist_left_behind / (dist_left_behind + dist_right_behind)
        rightness = weight_front * rightness_front + weight_behind * rightness_behind
        
        orient_rot = np.array([-orient[1],orient[0]])
        angle_front = angle_bw(orient_rot, h_front[1] - h_front[0])
        angle_behind = angle_bw(orient_rot, h_behind[1] - h_behind[0])
        angle = weight_front * angle_front + weight_behind * angle_behind
        
        if best_proj1 is None:
            print("ERROR best_proj")

        return np.array([progress,rightness,angle,speed])

    
    def read_distances(self, x, y, max_sensor_range, noisy=False, std=0):
        if x is np.nan or y is np.nan:
            return np.array([0,0,0,0])
        i = int(x // 10)
        j = int(y // 10)

        if i < 0 or i >= 140 or j < 0 or j >= 80:
            return np.array([0,0,0,0])
        
        if self.occupancy[i,j] != 0:
            return np.array([0.0,0.0,0.0,0.0])
        
        reading_up = y % 10
        reading_down = 10 - (y % 10)
        reading_left = x % 10
        reading_right = 10 - (x % 10)
        reading_NE = np.linalg.norm([10-(x%10),y%10])

        delta = 1
        while reading_up < max_sensor_range and j-delta >= 0 and self.occupancy[i,j-delta] == 0:
            reading_up += 10
            delta += 1
        reading_up = min(reading_up, max_sensor_range)

        delta = 1
        while reading_down < max_sensor_range and j+delta < 80 and self.occupancy[i,j+delta] == 0:
            reading_down += 10
            delta += 1
        reading_down = min(reading_down, max_sensor_range)

        delta = 1
        while reading_left < max_sensor_range and i-delta >= 0 and self.occupancy[i-delta,j] == 0:
            reading_left += 10
            delta += 1
        reading_left = min(reading_left, max_sensor_range)

        delta = 1
        while reading_right < max_sensor_range and i+delta < 140 and self.occupancy[i+delta,j] == 0:
            reading_right += 10
            delta += 1
        reading_right = min(reading_right, max_sensor_range)

        if noisy:
            reading_up = add_noise(x=reading_up, std=std)
            reading_down = add_noise(x=reading_down, std=std)
            reading_left = add_noise(x=reading_left, std=std)
            reading_right = add_noise(x=reading_right, std=std)
        readings = [reading_up,reading_down,reading_left,reading_right]


        return np.array(readings)
