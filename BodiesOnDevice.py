import warp as wp
import numpy as np
from DataTypes import *
from utils import SIM_NUM, EPS_SMALL, DEVICE

@wp.kernel
def stepUnconstraint(
    transform: wp.array(dtype=Transform, ndim=2),
    phi: wp.array(dtype=Vec6, ndim=2),
    I: wp.array(dtype=Vec6),
    f: Vec3,
    t: Vec3,
    gravity: Vec3,
    h: FP_DATA_TYPE
):
    i,j = wp.tid()
    if wp.math.norm_l1(I[i]) < EPS_SMALL:
        return
    I_r = Vec3(I[i][0], I[i][1], I[i][2])
    w = Vec3(phi[i,j][0], phi[i,j][1], phi[i,j][2])
    v = Vec3(phi[i,j][3], phi[i,j][4], phi[i,j][5])
    R = wp.quat_to_matrix(transform[i,j].q)
    Mr = wp.diag(I_r)
    Mr = R * Mr * wp.transpose(R)
    invMr = wp.diag(Vec3(1.0/I_r[0],1.0/I_r[1],1.0/I_r[2]))
    invMr = R * invMr * wp.transpose(R)
    f = f + I[i][3] * gravity
    t = t + wp.cross(wp.mul(Mr,w),w)
    w = w + h * (wp.mul(invMr,t))
    v =  v + h * f / I[i][3]
    phi[i,j] = Vec6(w[0], w[1], w[2], v[0], v[1], v[2])

@wp.kernel
def updateSubStepStates(
    phi: wp.array(dtype=Vec6, ndim=2),
    phi_dt: wp.array(dtype=Vec6, ndim=2),
    h: FP_DATA_TYPE
):
    i, j = wp.tid()
    phi_dt[i,j] = phi_dt[i,j] + h * phi[i,j]

@wp.kernel
def intergrateStates(
    transform: wp.array(dtype=Transform, ndim=2),
    phi_dt: wp.array(dtype=Vec6, ndim=2),
):
    i, j = wp.tid()
    dtheta = Vec3(phi_dt[i][j][0], phi_dt[i][j][1], phi_dt[i][j][2])
    dpos = Vec3(phi_dt[i][j][3], phi_dt[i][j][4], phi_dt[i][j][5])

    dtheta_norm = wp.math.norm_l2(dtheta)
    if(dtheta_norm < EPS_SMALL):
        dq = Quat(0.0,0.0,0.0,1.0)
    else:
        dq = wp.quat_from_axis_angle(dtheta / dtheta_norm, dtheta_norm)
    pPrime = transform[i,j].p + dpos
    qPrime = dq * transform[i,j].q
    transform[i,j] = Transform(pPrime, qPrime)

class BodiesOnDevice():
    '''
        Bodies class on Device representing all of the bodies.
    '''
    def __init__(self, bodies):
        self.body_num = len(bodies)
        self.transform = wp.empty(shape=(self.body_num, SIM_NUM), dtype=Transform)
        #self.axisAng = wp.empty(shape=(self.body_num, SIM_NUM), dtype=Vec3)
        self.phi = wp.empty(shape=(self.body_num, SIM_NUM), dtype=Vec6)
        self.phi_dt = wp.empty(shape=(self.body_num, SIM_NUM), dtype=Vec6)
        
        I_Host = np.empty(shape=self.body_num, dtype=Vec6)
        for i in range(self.body_num):
            wp.copy(self.transform[i], bodies[i].init_transform)
            wp.copy(self.phi[i], bodies[i].init_phi)
            I_Host[i] = bodies[i].I

        self.I = wp.array(I_Host, dtype=Vec6)

    def reset(self, bodies):
        for i in range(self.body_num):
            wp.copy(self.transform[i], bodies[i].init_transform)
            wp.copy(self.phi[i], bodies[i].init_phi)

    
    def stepUncons(self, h, gravity):
        f = Vec3(0.0,0.0,0.0)
        t = Vec3(0.0,0.0,0.0)
        
        wp.launch(
            kernel=stepUnconstraint,
            dim=(self.body_num, SIM_NUM),
            inputs=[
                self.transform,
                self.phi,
                self.I,
                f,
                t,
                gravity,
                h
            ]
        )
        wp.synchronize()
       
    def updateSubStepStates(self, h):
        wp.launch(
            kernel=updateSubStepStates,
            dim=(self.body_num, SIM_NUM),
            inputs=[
                self.phi,
                self.phi_dt,
                h
            ]
        )
        
    def intergrateStates(self,h):
        wp.launch(
            kernel=intergrateStates,
            dim=(self.body_num, SIM_NUM),
            inputs=[
                self.transform,
                self.phi_dt,
            ]
        )
        self.phi_dt.zero_()