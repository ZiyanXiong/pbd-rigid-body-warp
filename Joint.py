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
        self.has_target = False
        self.kp = 0
        self.kd = 0
        self.theta_target = None
        self.omega_target = None
        self.limits = None

    def setTarget(self, theta_target, omega_target, kp, kd):
        self.has_target = True
        self.kp = kp
        self.kd = kd
        self.theta_target = theta_target
        self.omega_target = omega_target
    
    def setLimits(self, limits):
        self.limits = limits
    
class JointHinge(Joint):
    def __init__(self, parent, parent_transform, child, child_transform, xl1, axis, index):
        super().__init__(parent, child, xl1, index)
        self.limits = wp.vec2(wp.PI, -wp.PI)
        # https://math.stackexchange.com/a/3582461
        g = wp.sign(axis[2])
        h = axis[2] + g
        b = wp.vec3(g - axis[0] * axis[0] / h, -axis[0] * axis[1] / h, -axis[0])
        self.axis1 = axis
        self.b1 = b
        self.xl2 = wp.transform_point(wp.transform_multiply(parent_transform, wp.transform_inverse(child_transform)), self.xl1)
        self.axis2 = wp.transform_vector(wp.transform_multiply(parent_transform, wp.transform_inverse(child_transform)), self.axis1)
        self.b2 = wp.transform_vector(wp.transform_multiply(parent_transform, wp.transform_inverse(child_transform)), self.b1)

class JointHingeWorld(Joint):
    def __init__(self, child, child_transform, xw, axis, index):
        super().__init__(None, child, xw, index)
        self.limits = wp.vec2(wp.PI, -wp.PI)
        # https://math.stackexchange.com/a/3582461
        g = wp.sign(axis[2])
        h = axis[2] + g
        b = wp.vec3(g - axis[0] * axis[0] / h, -axis[0] * axis[1] / h, -axis[0])
        self.axis1 = axis
        self.b1 = b
        self.xl2 = wp.transform_point(wp.transform_inverse(child_transform), self.xl1)
        self.axis2 = wp.transform_vector(wp.transform_inverse(child_transform), self.axis1)
        self.b2 = wp.transform_vector(wp.transform_inverse(child_transform), self.b1)
    

