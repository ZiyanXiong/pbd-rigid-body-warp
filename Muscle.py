import warp as wp
from DataTypes import *
from utils import SIM_NUM, EPS_SMALL, DEVICE

class Muscle():
    '''
        Muscle class on Host.
    '''
            
    def __init__(self, bodies, via_points, index):  
        self.bodies = bodies
        self.via_points = via_points
        self.index = index
 
class MuscleSpring(Muscle):
    def __init__(self, bodies, points, stiffness, rest_length, index):
        if len(points) < 4:
            raise ValueError("At least 2 via points are required to create a MuscleSpring. Points list should look like [Origin Via_point1 Via_point2 ... Insertion].")
        super().__init__(bodies, points[1:-1], index)
        rest_length -= wp.norm_l2(points[0] - points[1]) + wp.norm_l2(points[-2] - points[-1])
        self.stiffness = stiffness
        self.rest_length = rest_length