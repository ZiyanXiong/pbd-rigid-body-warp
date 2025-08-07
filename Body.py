import warp as wp
from DataTypes import *
from utils import SIM_NUM, EPS_SMALL, DEVICE

class Body():
    '''
        Body class on Host recording the initial and history state of one body.
    '''
    class StateHistory:
        def __init__(self):
            self.transforms = []
            self.phis = []
            
    def __init__(self):  
        self.index = -1
        self.transform_host = None
        self.transform_variation = False
        self.init_transform = None
        self.init_phi = None
        self.I = None
        self.stateHistory = self.StateHistory()
    
    def init(self):
        self.computInertiaConst()
    
    def setInitialState(self, transform, phi, transform_host, variation):
        self.init_transform = wp.array(transform, dtype = Transform)
        self.init_phi = wp.array(phi, dtype = Vec6)
        self.transform_host = transform_host
        self.transform_variation = variation

    def getResults(self, transform, phi):
        self.stateHistory.transforms.append(transform.numpy().tolist())
        self.stateHistory.phis.append(phi.numpy().tolist())

    
class BodyRigid(Body):
    def __init__(self, shape, density, mu, index):
        super().__init__()
        self.shape = shape
        self.density = density
        self.mu = mu
        self.index = index

    def computInertiaConst(self):
        self.I = self.shape.computeInertia(self.density)
        
class BodyFixed():
    def __init__(self, shape, transform, index):  
        self.shape = shape
        self.index = index
        self.transform = wp.array(transform, dtype=Transform)

        
