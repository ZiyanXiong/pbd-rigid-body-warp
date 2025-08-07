import warp as wp
from DataTypes import *
from utils import SIM_NUM, EPS_SMALL, DEVICE

class Joint():
    '''
        Joint class on Host.
    '''
            
    def __init__(self, parent, child, xl, index):  
        self.parent = parent
        self.child = child
        self.xl1 = xl
        self.xl2 = None
        self.index = index

class JointHingeWorld(Joint):
    def __init__(self, child, child_transform, xw, axis, index):
        super().__init__(None, child, xw, index)
        self.axis = axis
        self.xl2 = wp.transform_point(wp.transform_inverse(child_transform), self.xl1)
    
        


