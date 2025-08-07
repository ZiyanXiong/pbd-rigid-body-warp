import warp as wp
import numpy as np
from DataTypes import *
from CollisionDetection import collisionDetectionGroundCuboid, collisionDetectionCuboidCuboid
from utils import SIM_NUM, CONTACT_MAX, DEVICE, EPS_SMALL, getContactFrame, gamma, gammaDivM


@wp.kernel
def initGroundContact(
    con_count: wp.array(dtype=INT_DATA_TYPE),
    body_ind: wp.array(dtype=INT_DATA_TYPE, ndim=2),
    q: wp.array(dtype=Transform, ndim=2),
    I: wp.array(dtype=Vec6),
    xl: wp.array(dtype=Vec3, ndim=2),
    normal: wp.array(dtype=Vec3, ndim=2),
    # outputs
    J: wp.array(dtype=Mat36, ndim=2),
    J_div_m: wp.array(dtype=Mat36, ndim=2),
    w: wp.array(dtype=Vec3, ndim=2)
):
    i,j = wp.tid()
    if i > con_count[j]:
        return
    b_i = body_ind[i][j]
    cf = getContactFrame(normal[i][j])
    x_w = wp.transform_vector(q[b_i][j], xl[i][j])
    #wp.printf("x_w: %f %f %f\n", x_w[0], x_w[1], x_w[2])
    J[i][j] = cf * gamma(x_w)
    #J_div_m[i][j] = cf * gammaDivM(x_w, I[b_i])

    w_kernel = Vec3()
    J_div_m_kernel = Mat36()
    M_r = Vec3(I[b_i][0],I[b_i][1],I[b_i][2])
    M_p = I[b_i][3]
    for k in range(3):
        n_l = wp.transform_vector(wp.transform_inverse(q[b_i][j]), cf[k])
        rxn_l = wp.cross(xl[i][j], n_l)
        w_kernel[k] = wp.dot(rxn_l, wp.cw_div(rxn_l, M_r))  + 1.0 / M_p
        rxnI = wp.transform_vector(q[b_i][j], wp.cw_div(rxn_l, M_r))
        J_div_m_kernel[k] = Vec6(rxnI[0], rxnI[1], rxnI[2], cf[k][0] / M_p, cf[k][1] / M_p, cf[k][2] / M_p)

    w[i][j] = w_kernel
    J_div_m[i][j] = J_div_m_kernel

@wp.kernel
def solveGroundContactNormal(
    con_ind: INT_DATA_TYPE,
    con_count: wp.array(dtype=INT_DATA_TYPE),
    body_ind: wp.array(dtype=INT_DATA_TYPE, ndim=2),
    J: wp.array(dtype=Mat36),
    J_div_m: wp.array(dtype=Mat36),
    w: wp.array(dtype=Vec3),
    d: wp.array(dtype=FP_DATA_TYPE),
    h: FP_DATA_TYPE,
    #outputs
    lambdas: wp.array(dtype=Vec3),
    phi: wp.array(dtype=Vec6, ndim=2),
    phi_dt: wp.array(dtype=Vec6, ndim=2)
):
    i = wp.tid()
    if con_ind >= con_count[i]:
        return
    b_i = body_ind[con_ind][i]
    lambda_i = lambdas[i]

    c = wp.dot(J[i][0], phi[b_i][i] + phi_dt[b_i][i] / h) - d[i] / h
    dlambda_nor = -c / (w[i][0])
    # wp.printf("c: %f \n", c)
    if(lambda_i[0] + dlambda_nor < 0.0):
        dlambda_nor = -lambda_i[0]
    lambda_i[0] += dlambda_nor
    lambdas[i] = lambda_i
    # wp.printf("dlambda_nor: %f \n", dlambda_nor)
    #wp.atomic_add(phi[b_i],i , J_div_m[i][0] * lambdas[i][0])
    #update = J_div_m[i][0] * dlambda_nor
    #wp.printf("updates: %f %f %f %f %f %f\n", update[0], update[1], update[2], update[3], update[4], update[5])
    phi[b_i][i] += J_div_m[i][0] * dlambda_nor

@wp.kernel
def solveGroundContactTangent(
    con_ind: INT_DATA_TYPE,
    con_count: wp.array(dtype=INT_DATA_TYPE),
    body_ind: wp.array(dtype=INT_DATA_TYPE, ndim=2),
    J: wp.array(dtype=Mat36),
    J_div_m: wp.array(dtype=Mat36),
    w: wp.array(dtype=Vec3),
    mu: wp.array(dtype=FP_DATA_TYPE),
    h: FP_DATA_TYPE,
    #outputs
    lambdas: wp.array(dtype=Vec3),
    phi: wp.array(dtype=Vec6, ndim=2),
    phi_dt: wp.array(dtype=Vec6, ndim=2)
):
    i = wp.tid()
    if con_ind >= con_count[i]:
        return
    b_i = body_ind[con_ind][i]
    lambda_i = lambdas[i]

    c1 = wp.dot(J[i][1], phi[b_i][i] + phi_dt[b_i][i] / h)
    c2 = wp.dot(J[i][2], phi[b_i][i] + phi_dt[b_i][i] / h)

    dlambda_tan1 = -c1 / (w[i][1])
    dlambda_tan2 = -c2 / (w[i][2])
    lambda_tan = Vec2(lambda_i[1]+ dlambda_tan1, lambda_i[2]+ dlambda_tan2)
    lambda_tan_norm = wp.math.norm_l2(lambda_tan)
    if(lambda_tan_norm > wp.max(EPS_SMALL, mu[i] * lambda_i[0])):
        dlambda_tan1 = mu[i] * lambda_i[0] * lambda_tan[0] / lambda_tan_norm - lambda_i[1]
        dlambda_tan2 = mu[i] * lambda_i[0] * lambda_tan[1] / lambda_tan_norm - lambda_i[2]
    lambda_i[1] += dlambda_tan1
    lambda_i[2] += dlambda_tan2
    lambdas[i] = lambda_i
    # wp.printf("dlambda_tan: %f %f\n", dlambda_tan1, dlambda_tan2)
    phi[b_i][i] += J_div_m[i][1] * dlambda_tan1 + J_div_m[i][2] * dlambda_tan2

@wp.kernel
def initBodyContact(
    con_count: wp.array(dtype=INT_DATA_TYPE),
    body_ind1: wp.array(dtype=INT_DATA_TYPE, ndim=2),
    body_ind2: wp.array(dtype=INT_DATA_TYPE, ndim=2),
    q: wp.array(dtype=Transform, ndim=2),
    I: wp.array(dtype=Vec6),
    xl1: wp.array(dtype=Vec3, ndim=2),
    xl2: wp.array(dtype=Vec3, ndim=2),
    normal: wp.array(dtype=Vec3, ndim=2),
    # outputs
    J1: wp.array(dtype=Mat36, ndim=2),
    J_div_m1: wp.array(dtype=Mat36, ndim=2),
    w1: wp.array(dtype=Vec3, ndim=2),
    J2: wp.array(dtype=Mat36, ndim=2),
    J_div_m2: wp.array(dtype=Mat36, ndim=2),
    w2: wp.array(dtype=Vec3, ndim=2),
):
    i,j = wp.tid()
    if i > con_count[j]:
        return
    b1_i = body_ind1[i][j]
    b2_i = body_ind2[i][j]
    cf = getContactFrame(normal[i][j])
    x_w = wp.transform_vector(q[b1_i][j], xl1[i][j])
    J1[i][j] = cf * gamma(x_w)

    x_w = wp.transform_vector(q[b2_i][j], xl2[i][j])
    J2[i][j] = cf * gamma(x_w)

    w_kernel = Vec3()
    J_div_m_kernel = Mat36()
    M_r = Vec3(I[b1_i][0],I[b1_i][1],I[b1_i][2])
    M_p = I[b1_i][3]
    for k in range(3):
        n_l = wp.transform_vector(wp.transform_inverse(q[b1_i][j]), cf[k])
        rxn_l = wp.cross(xl1[i][j], n_l)
        w_kernel[k] = wp.dot(rxn_l, wp.cw_div(rxn_l, M_r)) + 1.0 / M_p
        rxnI = wp.transform_vector(q[b1_i][j], wp.cw_div(rxn_l, M_r))
        J_div_m_kernel[k] = Vec6(rxnI[0], rxnI[1], rxnI[2], cf[k][0] / M_p, cf[k][1] / M_p, cf[k][2] / M_p)
    w1[i][j] = w_kernel
    J_div_m1[i][j] = J_div_m_kernel

    M_r = Vec3(I[b2_i][0],I[b2_i][1],I[b2_i][2])
    M_p = I[b2_i][3]
    for k in range(3):
        n_l = wp.transform_vector(wp.transform_inverse(q[b2_i][j]), cf[k])
        rxn_l = wp.cross(xl2[i][j], n_l)
        w_kernel[k] = wp.dot(rxn_l, wp.cw_div(rxn_l, M_r))  + 1.0 / M_p
        rxnI = wp.transform_vector(q[b2_i][j], wp.cw_div(rxn_l, M_r))
        J_div_m_kernel[k] = Vec6(rxnI[0], rxnI[1], rxnI[2], cf[k][0] / M_p, cf[k][1] / M_p, cf[k][2] / M_p)
    w2[i][j] = w_kernel
    J_div_m2[i][j] = J_div_m_kernel

@wp.kernel
def solveBodyContactNormal(
    con_ind: INT_DATA_TYPE,
    con_count: wp.array(dtype=INT_DATA_TYPE),
    body_ind1: wp.array(dtype=INT_DATA_TYPE, ndim=2),
    J1: wp.array(dtype=Mat36),
    J_div_m1: wp.array(dtype=Mat36),
    w1: wp.array(dtype=Vec3),
    body_ind2: wp.array(dtype=INT_DATA_TYPE, ndim=2),
    J2: wp.array(dtype=Mat36),
    J_div_m2: wp.array(dtype=Mat36),
    w2: wp.array(dtype=Vec3),
    d: wp.array(dtype=FP_DATA_TYPE),
    h: FP_DATA_TYPE,
    #outputs
    lambdas: wp.array(dtype=Vec3),
    phi: wp.array(dtype=Vec6, ndim=2),
    phi_dt: wp.array(dtype=Vec6, ndim=2),
):
    i = wp.tid()
    if con_ind >= con_count[i]:
        return
    b1_i = body_ind1[con_ind][i]
    b2_i = body_ind2[con_ind][i]
    lambda_i = lambdas[i]

    c = wp.dot(J1[i][0], phi[b1_i][i] + phi_dt[b1_i][i] / h) - d[i] / h
    c -= wp.dot(J2[i][0], phi[b2_i][i] + phi_dt[b2_i][i] / h)
    dlambda_nor = -c / (w1[i][0] + w2[i][0])
    #wp.printf("c: %f \n", c)
    if(lambda_i[0] + dlambda_nor < 0.0):
        dlambda_nor = -lambda_i[0]
    lambda_i[0] += dlambda_nor
    lambdas[i] = lambda_i
    #wp.printf("dlambda_nor: %f \n", dlambda_nor)
    #wp.atomic_add(phi[b_i],i , J_div_m[i][0] * lambdas[i][0])
    #update = J_div_m[i][0] * dlambda_nor
    #wp.printf("updates: %f %f %f %f %f %f\n", update[0], update[1], update[2], update[3], update[4], update[5])
    phi[b1_i][i] += J_div_m1[i][0] * dlambda_nor
    phi[b2_i][i] -= J_div_m2[i][0] * dlambda_nor

@wp.kernel
def solveBodyContactTangent(
    con_ind: INT_DATA_TYPE,
    con_count: wp.array(dtype=INT_DATA_TYPE),
    body_ind1: wp.array(dtype=INT_DATA_TYPE, ndim=2),
    J1: wp.array(dtype=Mat36),
    J_div_m1: wp.array(dtype=Mat36),
    w1: wp.array(dtype=Vec3),
    body_ind2: wp.array(dtype=INT_DATA_TYPE, ndim=2),
    J2: wp.array(dtype=Mat36),
    J_div_m2: wp.array(dtype=Mat36),
    w2: wp.array(dtype=Vec3),
    h: FP_DATA_TYPE,
    mu: wp.array(dtype=FP_DATA_TYPE),
    #outputs
    lambdas: wp.array(dtype=Vec3),
    phi: wp.array(dtype=Vec6, ndim=2),
    phi_dt: wp.array(dtype=Vec6, ndim=2),
):
    i = wp.tid()
    if con_ind >= con_count[i]:
        return
    b1_i = body_ind1[con_ind][i]
    b2_i = body_ind2[con_ind][i]
    lambda_i = lambdas[i]

    c1 = wp.dot(J1[i][1], phi[b1_i][i] + phi_dt[b1_i][i] / h)
    c1 -= wp.dot(J2[i][1], phi[b2_i][i] + phi_dt[b2_i][i] / h)
    c2 = wp.dot(J1[i][2], phi[b1_i][i] + phi_dt[b1_i][i] / h)
    c2 -= wp.dot(J2[i][2], phi[b2_i][i] + phi_dt[b2_i][i] / h)

    dlambda_tan1 = -c1 / (w1[i][1] + w2[i][1])
    dlambda_tan2 = -c2 / (w1[i][1] + w2[i][1])
    lambda_tan = Vec2(lambda_i[1]+ dlambda_tan1, lambda_i[2]+ dlambda_tan2)
    lambda_tan_norm = wp.math.norm_l2(lambda_tan)
    if(lambda_tan_norm > wp.max(EPS_SMALL, mu[i] * lambda_i[0])):
        dlambda_tan1 = mu[i] * lambda_i[0] * lambda_tan[0] / lambda_tan_norm - lambda_i[1]
        dlambda_tan2 = mu[i] * lambda_i[0] * lambda_tan[1] / lambda_tan_norm - lambda_i[2]
    lambda_i[1] += dlambda_tan1
    lambda_i[2] += dlambda_tan2
    lambdas[i] = lambda_i
    # wp.printf("dlambda_tan: %f %f\n", dlambda_tan1, dlambda_tan2)
    phi[b1_i][i] += J_div_m1[i][1] * dlambda_tan1 + J_div_m1[i][2] * dlambda_tan2
    phi[b2_i][i] -= J_div_m2[i][1] * dlambda_tan1 + J_div_m2[i][2] * dlambda_tan2

@wp.kernel
def initWorldFix(
    con_count: INT_DATA_TYPE,
    body_ind: wp.array(dtype=INT_DATA_TYPE),
    q: wp.array(dtype=Transform, ndim=2),
    I: wp.array(dtype=Vec6),
    xw0: wp.array(dtype=Vec3),
    xl: wp.array(dtype=Vec3),
    axis: wp.array(dtype=Vec3),
    # outputs
    d: wp.array(dtype=Vec3, ndim=2),
    J: wp.array(dtype=Mat36, ndim=2),
    J_div_m: wp.array(dtype=Mat36, ndim=2),
    w: wp.array(dtype=Vec3, ndim=2)
):
    i,j = wp.tid()
    if i > con_count:
        return
    b_i = body_ind[i]
    cf = getContactFrame(wp.transform_vector(q[b_i][j], axis[i]))
    x_w = wp.transform_vector(q[b_i][j], xl[i])
    # q_ij = q[b_i][j]
    # wp.printf("q: %f %f %f %f %f %f %f\n", q_ij[0], q_ij[1], q_ij[2], q_ij[3], q_ij[4], q_ij[5], q_ij[6])
    # wp.printf("axis: %f %f %f\n", axis[i][0], axis[i][1], axis[i][2])
    # wp.printf("x_l: %f %f %f\n", xl[i][0], xl[i][1], xl[i][2])
    # wp.printf("x_w: %f %f %f\n", x_w[0], x_w[1], x_w[2])
    d[i][j] = cf * (xw0[i] - x_w - q[b_i][j].p)
    J[i][j] = cf * gamma(x_w)
    w_kernel = Vec3()
    J_div_m_kernel = Mat36()
    M_r = Vec3(I[b_i][0],I[b_i][1],I[b_i][2])
    M_p = I[b_i][3]
    for k in range(3):
        # wp.printf("c_fk: %f %f %f\n", cf[k][0], cf[k][1], cf[k][2])
        n_l = wp.transform_vector(wp.transform_inverse(q[b_i][j]), cf[k])
        # wp.printf("n_l: %f %f %f\n", n_l[0], n_l[1], n_l[2])
        rxn_l = wp.cross(xl[i], n_l)
        # wp.printf("rxn_l: %f %f %f\n", rxn_l[0], rxn_l[1], rxn_l[2])
        w_kernel[k] = wp.dot(rxn_l, wp.cw_div(rxn_l, M_r))  + 1.0 / M_p
        rxnI = wp.transform_vector(q[b_i][j], wp.cw_div(rxn_l, M_r))
        J_div_m_kernel[k] = Vec6(rxnI[0], rxnI[1], rxnI[2], cf[k][0] / M_p, cf[k][1] / M_p, cf[k][2] / M_p)

    w[i][j] = w_kernel
    J_div_m[i][j] = J_div_m_kernel

@wp.kernel
def solveWorldFix(
    con_count: INT_DATA_TYPE,
    body_ind: wp.array(dtype=INT_DATA_TYPE),
    J: wp.array(dtype=Mat36, ndim=2),
    J_div_m: wp.array(dtype=Mat36, ndim=2),
    w: wp.array(dtype=Vec3, ndim=2),
    d: wp.array(dtype=Vec3, ndim=2),
    h: FP_DATA_TYPE,
    #outputs
    lambdas: wp.array(dtype=Vec3, ndim=2),
    phi: wp.array(dtype=Vec6, ndim=2),
    phi_dt: wp.array(dtype=Vec6, ndim=2)
):
    j = wp.tid()

    for i in range(con_count):
        b_i = body_ind[i]
        d_ij = d[i][j]
        phi_ij = phi[b_i][j]
        w_ij = w[i][j]
        lambdas_ij=lambdas[i][j]
        bias = phi_dt[b_i][j] / h
        J_ij = J[i][j]
        J_div_m_ij = J_div_m[i][j]
        for k in range(3):
            c = -wp.dot(J_ij[k], (phi_ij +bias)) + d_ij[k] / h
            # wp.printf("c: %f = %f + %f\n", c, wp.dot(J_ij[k], (phi_ij +bias)), d_ij[k] / h)
            # wp.printf("d_ij: %f \n", d_ij[k])
            # wp.printf("bias: %f %f %f %f %f %f\n", bias[0], bias[1], bias[2], bias[3], bias[4], bias[5])
            # wp.printf("phi: %f %f %f %f %f %f\n", phi_ij[0], phi_ij[1], phi_ij[2], phi_ij[3], phi_ij[4], phi_ij[5])
            # wp.printf("Jijk: %f %f %f %f %f %f\n", J_ij[k][0], J_ij[k][1], J_ij[k][2], J_ij[k][3], J_ij[k][4], J_ij[k][5])
            dlambda = -c / w_ij[k]
            # wp.printf("dlambda: %f \n", dlambda)
            lambdas_ij[k] += dlambda
            phi_ij -= J_div_m_ij[k] * dlambda
        lambdas[i][j] = lambdas_ij
        phi[b_i][j] = phi_ij

@wp.kernel
def initWorldRotate(
    con_count:INT_DATA_TYPE,
    body_ind: wp.array(dtype=INT_DATA_TYPE),
    q: wp.array(dtype=Transform, ndim=2),
    I: wp.array(dtype=Vec6),
    axis: wp.array(dtype=Vec3),
    # outputs
    d: wp.array(dtype=Vec3, ndim=2),
    J: wp.array(dtype=Mat36, ndim=2),
    J_div_m: wp.array(dtype=Mat36, ndim=2),
    w: wp.array(dtype=Vec3, ndim=2)
):
    i,j = wp.tid()
    if i > con_count:
        return
    b_i = body_ind[i]
    cf = getContactFrame(wp.transform_vector(q[b_i][j], axis[i]))

    w_kernel = Vec3()
    J_div_m_kernel = Mat36()
    J_kernel = Mat36()
    M_r = Vec3(I[b_i][0],I[b_i][1],I[b_i][2])
    for k in range(3):
        n_l = wp.transform_vector(wp.transform_inverse(q[b_i][j]), cf[k])
        w_kernel[k] = 1.0 / M_r[k]
        rxnI = wp.transform_vector(q[b_i][j], wp.cw_div(n_l, M_r))
        J_div_m_kernel[k] = Vec6(rxnI[0], rxnI[1], rxnI[2], 0.0, 0.0, 0.0)
        J_kernel[k] = Vec6(cf[k][0], cf[k][1], cf[k][2], 0.0, 0.0, 0.0)

    w[i][j] = w_kernel
    J[i][j] = J_kernel
    J_div_m[i][j] = J_div_m_kernel

    r_axis, dtheta = wp.quat_to_axis_angle(q[b_i][j].q)
    d[i][j] = cf * r_axis * dtheta


@wp.kernel
def solveWorldRotate(
    con_count: INT_DATA_TYPE,
    body_ind: wp.array(dtype=INT_DATA_TYPE),
    J: wp.array(dtype=Mat36, ndim=2),
    J_div_m: wp.array(dtype=Mat36, ndim=2),
    w: wp.array(dtype=Vec3, ndim=2),
    d: wp.array(dtype=Vec3, ndim=2),
    h: FP_DATA_TYPE,
    #outputs
    lambdas: wp.array(dtype=Vec3, ndim=2),
    phi: wp.array(dtype=Vec6, ndim=2),
    phi_dt: wp.array(dtype=Vec6, ndim=2)
):
    j = wp.tid()

    for i in range(con_count):
        b_i = body_ind[i]
        d_ij = d[i][j]
        phi_ij = phi[b_i][j]
        w_ij = w[i][j]
        lambdas_ij=lambdas[i][j]
        bias = phi_dt[b_i][j] / h
        J_ij = J[i][j]
        J_div_m_ij = J_div_m[i][j]
        for k in range(3):
            if(k==0):
                continue
            c = wp.dot(J_ij[k], (phi_ij +bias)) - d_ij[k] / h
            # wp.printf("d: %f\n", d_ij[k])
            # wp.printf("bias: %f %f %f %f %f %f\n", bias[0], bias[1], bias[2], bias[3], bias[4], bias[5])
            dlambda = -c / w_ij[k]
            lambdas_ij[k] += dlambda
            #wp.printf("dlambda: %f %f %f\n", dlambda[0], dlambda[1], dlambda[2])
            phi_ij += J_div_m_ij[k] * dlambda
        lambdas[i][j] = lambdas_ij
        phi[b_i][j] = phi_ij

class ConstraintsOnDevice():
    def __init__(self):
        self.con_count = wp.zeros(shape=SIM_NUM, dtype=INT_DATA_TYPE)

    def fillConstraints(self):
        pass

    def computeC(self):
        pass

    def init(self):
        pass

    def solve(self):
        pass

    def solveVelocity(self):
        pass

class ConstraintsGroundContact(ConstraintsOnDevice):
    def __init__(self, fixed, bodies_on_host, bodies_on_device):
        super().__init__()
        self.fixed = fixed
        self.bodies_on_host = bodies_on_host
        self.bodies = bodies_on_device
        self.c = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=Vec3)
        self.lambdas = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=Vec3)
        self.body_ind = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=INT_DATA_TYPE)
        self.J = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=Mat36)
        self.J_div_m = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=Mat36)
        self.xl = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=Vec3)
        self.w = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=Vec3)
        self.normal = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=Vec3)
        self.d = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=FP_DATA_TYPE)
        self.mu = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=FP_DATA_TYPE)

    def fillConstraints(self):
        self.con_count.zero_()
        for i in range(len(self.fixed)):
            for j in range(len(self.bodies_on_host)):
                collisionDetectionGroundCuboid(self.bodies_on_host[j].index,
                                                self.bodies.transform[j],
                                                self.bodies_on_host[j].shape.sides,
                                                self.bodies_on_host[j].mu,
                                                self.fixed[i].transform,
                                                self.fixed[i].shape.normal,
                                                self.con_count,
                                                self.body_ind,
                                                self.d,
                                                self.normal,
                                                self.xl,
                                                self.mu)
                # print("countact_num: ",self.con_count)
                # print("d: ",self.d.numpy())
                # print("normal: ",self.normal.numpy())
                # print("xl: ",self.xl.numpy())

    def init(self):
        wp.launch(
            kernel=initGroundContact,
            dim=(CONTACT_MAX, SIM_NUM),
            inputs=[
                self.con_count,
                self.body_ind,
                self.bodies.transform,
                self.bodies.I,
                self.xl,
                self.normal
            ],
            outputs=[
                self.J,
                self.J_div_m,
                self.w
            ],
            device=DEVICE.GPU,
            record_tape=False,
        )
        # print("J: ",self.J.numpy())
        # print("J_div_m: ",self.J_div_m.numpy())
        # print("w: ",self.w.numpy())

    def computeC(self):
        # Compute contact constraints between ground and bodies
        pass


    def solve(self, h):
        for i in range(CONTACT_MAX):
            wp.launch(
                kernel=solveGroundContactNormal,
                dim=SIM_NUM,
                inputs=[
                    i,
                    self.con_count,
                    self.body_ind,
                    self.J[i],
                    self.J_div_m[i],
                    self.w[i],
                    self.d[i],
                    h
                ],
                outputs=[
                    self.lambdas[i],
                    self.bodies.phi,
                    self.bodies.phi_dt
                ],
                device=DEVICE.GPU,
                record_tape=False
            )
            wp.synchronize()
            # print("phi: ",self.bodies.phi.numpy())
            # print("lambdas: ",self.lambdas.numpy())

        for i in range(CONTACT_MAX):
            wp.launch(
                kernel=solveGroundContactTangent,
                dim=SIM_NUM,
                inputs=[
                    i,
                    self.con_count,
                    self.body_ind,
                    self.J[i],
                    self.J_div_m[i],
                    self.w[i],
                    self.mu[i],
                    h
                ],
                outputs=[
                    self.lambdas[i],
                    self.bodies.phi,
                    self.bodies.phi_dt
                ],
                device=DEVICE.GPU,
                record_tape=False
            )
            wp.synchronize()
            # print("phi: ",self.bodies.phi.numpy())

    def solveVelocity(self, h):
        d_zeros = wp.zeros(shape=SIM_NUM, dtype=FP_DATA_TYPE)
        for i in range(CONTACT_MAX):
            wp.launch(
                kernel=solveGroundContactNormal,
                dim=SIM_NUM,
                inputs=[
                    i,
                    self.con_count,
                    self.body_ind,
                    self.J[i],
                    self.J_div_m[i],
                    self.w[i],
                    d_zeros,
                    h
                ],
                outputs=[
                    self.lambdas[i],
                    self.bodies.phi,
                    self.bodies.phi_dt
                ],
                device=DEVICE.GPU,
                record_tape=False
            )
            wp.synchronize()
            # print("phi: ",self.bodies.phi.numpy())
            # print("lambdas: ",self.lambdas.numpy())

        for i in range(CONTACT_MAX):
            wp.launch(
                kernel=solveGroundContactTangent,
                dim=SIM_NUM,
                inputs=[
                    i,
                    self.con_count,
                    self.body_ind,
                    self.J[i],
                    self.J_div_m[i],
                    self.w[i],
                    self.mu[i],
                    h
                ],
                outputs=[
                    self.lambdas[i],
                    self.bodies.phi,
                    self.bodies.phi_dt
                ],
                device=DEVICE.GPU,
                record_tape=False
            )
            wp.synchronize()

        #print("phi: ",self.bodies.phi.numpy())

class ConstraintsBodyContact(ConstraintsOnDevice):
    def __init__(self, bodies_on_host, bodies_on_device, body_pairs):
        super().__init__()
        self.bodies_on_host = bodies_on_host
        self.bodies = bodies_on_device
        self.body_pairs = body_pairs
        self.c = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=Vec3)
        self.lambdas = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=Vec3)
        self.body_ind1 = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=INT_DATA_TYPE)
        self.body_ind2 = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=INT_DATA_TYPE)
        self.J1 = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=Mat36)
        self.J2 = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=Mat36)
        self.J_div_m1 = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=Mat36)
        self.J_div_m2 = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=Mat36)
        self.xl1 = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=Vec3)
        self.xl2 = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=Vec3)
        self.w1 = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=Vec3)
        self.w2 = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=Vec3)
        self.normal = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=Vec3)
        self.d = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=FP_DATA_TYPE)
        self.mu = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=FP_DATA_TYPE)

    def fillConstraints(self):
        self.con_count.zero_()
        for i in range(len(self.body_pairs)):
            body1, body2 = self.body_pairs[i]
            collisionDetectionCuboidCuboid(body1,
                                            self.bodies.transform[body1],
                                            self.bodies_on_host[body1].shape.sides,
                                            body2,
                                            self.bodies.transform[body2],
                                            self.bodies_on_host[body2].shape.sides,
                                            0.5 * (self.bodies_on_host[body1].mu + self.bodies_on_host[body2].mu),
                                            self.con_count,
                                            self.body_ind1,
                                            self.body_ind2,
                                            self.d,
                                            self.normal,
                                            self.xl1,
                                            self.xl2,
                                            self.mu)
            # print("d: ",self.d.numpy())
            # print("normal: ",self.normal.numpy())
            # print("xl1: ",self.xl1.numpy())
            # print("xl2: ",self.xl2.numpy())

    def computeC(self):
        # Compute contact constraints between ground and bodies
        pass

    def init(self):
        wp.launch(
            kernel=initBodyContact,
            dim=(CONTACT_MAX, SIM_NUM),
            inputs=[
                self.con_count,
                self.body_ind1,
                self.body_ind2,
                self.bodies.transform,
                self.bodies.I,
                self.xl1,
                self.xl2,
                self.normal
            ],
            outputs=[
                self.J1,
                self.J_div_m1,
                self.w1,
                self.J2,
                self.J_div_m2,
                self.w2
            ],
            device=DEVICE.GPU,
            record_tape=False,
        )
        # print("J1: ",self.J1.numpy())
        # print("J_div_m1: ",self.J_div_m1.numpy())
        # print("w1: ",self.w1.numpy())
        # print("J2: ",self.J2.numpy())
        # print("J_div_m2: ",self.J_div_m2.numpy())
        # print("w2: ",self.w2.numpy())

    def solve(self, h):
        for i in range(CONTACT_MAX):
            wp.launch(
                kernel=solveBodyContactNormal,
                dim=SIM_NUM,
                inputs=[
                    i,
                    self.con_count,
                    self.body_ind1,
                    self.J1[i],
                    self.J_div_m1[i],
                    self.w1[i],
                    self.body_ind2,
                    self.J2[i],
                    self.J_div_m2[i],
                    self.w2[i],
                    self.d[i],
                    h
                ],
                outputs=[
                    self.lambdas[i],
                    self.bodies.phi,
                    self.bodies.phi_dt
                ],
                device=DEVICE.GPU,
                record_tape=False
            )
            wp.synchronize()
            # print("phi: ",self.bodies.phi.numpy())
            # print("lambdas: ",self.lambdas.numpy())

        for i in range(CONTACT_MAX):
            wp.launch(
                kernel=solveBodyContactTangent,
                dim=SIM_NUM,
                inputs=[
                    i,
                    self.con_count,
                    self.body_ind1,
                    self.J1[i],
                    self.J_div_m1[i],
                    self.w1[i],
                    self.body_ind1,
                    self.J1[i],
                    self.J_div_m1[i],
                    self.w1[i],
                    h,
                    self.mu[i],
                ],
                outputs=[
                    self.lambdas[i],
                    self.bodies.phi,
                    self.bodies.phi_dt
                ],
                device=DEVICE.GPU,
                record_tape=False
            )
            wp.synchronize()
            # print("phi: ",self.bodies.phi.numpy())

    def solveVelocity(self, h):
        d_zeros = wp.zeros(shape=SIM_NUM, dtype=FP_DATA_TYPE)
        for i in range(CONTACT_MAX):
            wp.launch(
                kernel=solveBodyContactNormal,
                dim=SIM_NUM,
                inputs=[
                    i,
                    self.con_count,
                    self.body_ind1,
                    self.J1[i],
                    self.J_div_m1[i],
                    self.w1[i],
                    self.body_ind2,
                    self.J2[i],
                    self.J_div_m2[i],
                    self.w2[i],
                    d_zeros,
                    h
                ],
                outputs=[
                    self.lambdas[i],
                    self.bodies.phi,
                    self.bodies.phi_dt
                ],
                device=DEVICE.GPU,
                record_tape=False
            )
            wp.synchronize()
            # print("phi: ",self.bodies.phi.numpy())
            # print("lambdas: ",self.lambdas.numpy())

        for i in range(CONTACT_MAX):
            wp.launch(
                kernel=solveBodyContactTangent,
                dim=SIM_NUM,
                inputs=[
                    i,
                    self.con_count,
                    self.body_ind1,
                    self.J1[i],
                    self.J_div_m1[i],
                    self.w1[i],
                    self.body_ind1,
                    self.J1[i],
                    self.J_div_m1[i],
                    self.w1[i],
                    h,
                    self.mu[i],
                ],
                outputs=[
                    self.lambdas[i],
                    self.bodies.phi,
                    self.bodies.phi_dt
                ],
                device=DEVICE.GPU,
                record_tape=False
            )
            wp.synchronize()
            # print("phi: ",self.bodies.phi.numpy())

class ConstraintsWorldFix(ConstraintsOnDevice):
    def __init__(self, joints, bodies_on_device):
        self.con_count = len(joints)
        self.bodies = bodies_on_device

        body_ind = np.empty(shape=self.con_count, dtype=INT_DATA_TYPE)
        xw = np.empty(shape=self.con_count, dtype=Vec3)
        xl = np.empty(shape=self.con_count, dtype=Vec3)
        normal = np.empty(shape=self.con_count, dtype=Vec3)
        for i in range(self.con_count):
            body_ind[i] = joints[i].child
            xw[i] = joints[i].xl1
            xl[i] = joints[i].xl2
            normal[i] = joints[i].axis

        self.c = wp.empty(shape=(self.con_count, SIM_NUM), dtype=Vec3)
        self.lambdas = wp.empty(shape=(self.con_count, SIM_NUM), dtype=Vec3)
        self.body_ind = wp.array(body_ind, dtype=INT_DATA_TYPE)
        self.J = wp.empty(shape=(self.con_count, SIM_NUM), dtype=Mat36)
        self.J_div_m = wp.empty(shape=(self.con_count, SIM_NUM), dtype=Mat36)
        self.xw = wp.array(xw, dtype=Vec3)
        self.xl = wp.array(xl, dtype=Vec3)
        self.w = wp.empty(shape=(self.con_count, SIM_NUM), dtype=Vec3)
        self.normal = wp.array(normal, dtype=Vec3)
        self.d = wp.empty(shape=(self.con_count, SIM_NUM), dtype=Vec3)

    def init(self):
        wp.launch(
            kernel=initWorldFix,
            dim=(self.con_count, SIM_NUM),
            inputs=[
                self.con_count,
                self.body_ind,
                self.bodies.transform,
                self.bodies.I,
                self.xw,
                self.xl,
                self.normal
            ],
            outputs=[
                self.d,
                self.J,
                self.J_div_m,
                self.w
            ],
            device=DEVICE.GPU,
            record_tape=False,
        )
        # print("J: ",self.J.numpy())
        # print("J_div_m: ",self.J_div_m.numpy())
        # print("w: ",self.w.numpy())

    def computeC(self):
        # Compute contact constraints between ground and bodies
        pass

    def solve(self, h):
        wp.launch(
            kernel=solveWorldFix,
            dim=SIM_NUM,
            inputs=[
                self.con_count,
                self.body_ind,
                self.J,
                self.J_div_m,
                self.w,
                self.d,
                h
            ],
            outputs=[
                self.lambdas,
                self.bodies.phi,
                self.bodies.phi_dt
            ],
            device=DEVICE.GPU,
            record_tape=False
        )
        wp.synchronize()
        # print("phi: ",self.bodies.phi.numpy())
        # print("lambdas: ",self.lambdas.numpy())

    def solveVelocity(self, h):
        self.d.zero_()
        wp.launch(
            kernel=solveWorldFix,
            dim=SIM_NUM,
            inputs=[
                self.con_count,
                self.body_ind,
                self.J,
                self.J_div_m,
                self.w,
                self.d,
                h
            ],
            outputs=[
                self.lambdas,
                self.bodies.phi,
                self.bodies.phi_dt
            ],
            device=DEVICE.GPU,
            record_tape=False
        )
        wp.synchronize()
        self.lambdas.zero_()
        # print("phi: ",self.bodies.phi.numpy())
        # print("lambdas: ",self.lambdas.numpy())

class ConstraintsWorldRotate(ConstraintsOnDevice):
    def __init__(self, joints, bodies_on_device):
        self.con_count = len(joints)
        self.bodies = bodies_on_device

        body_ind = np.empty(shape=self.con_count, dtype=INT_DATA_TYPE)
        axis = np.empty(shape=self.con_count, dtype=Vec3)
        for i in range(self.con_count):
            body_ind[i] = joints[i].child
            axis[i] = joints[i].axis

        self.c = wp.empty(shape=(self.con_count, SIM_NUM), dtype=Vec3)
        self.lambdas = wp.empty(shape=(self.con_count, SIM_NUM), dtype=Vec3)
        self.body_ind = wp.array(body_ind, dtype=INT_DATA_TYPE)
        self.J = wp.empty(shape=(self.con_count, SIM_NUM), dtype=Mat36)
        self.J_div_m = wp.empty(shape=(self.con_count, SIM_NUM), dtype=Mat36)
        self.w = wp.empty(shape=(self.con_count, SIM_NUM), dtype=Vec3)
        self.axis = wp.array(axis, dtype=Vec3)
        self.d = wp.empty(shape=(self.con_count, SIM_NUM), dtype=Vec3)

    def init(self):
        wp.launch(
            kernel=initWorldRotate,
            dim=(self.con_count, SIM_NUM),
            inputs=[
                self.con_count,
                self.body_ind,
                self.bodies.transform,
                self.bodies.I,
                self.axis
            ],
            outputs=[
                self.d,
                self.J,
                self.J_div_m,
                self.w
            ],
            device=DEVICE.GPU,
            record_tape=False,
        )
        # print("J: ",self.J.numpy())
        # print("J_div_m: ",self.J_div_m.numpy())
        # print("w: ",self.w.numpy())
        # print("d: ",self.d.numpy())

    def computeC(self):
        # Compute contact constraints between ground and bodies
        pass


    def solve(self, h):
        wp.launch(
            kernel=solveWorldRotate,
            dim=SIM_NUM,
            inputs=[
                self.con_count,
                self.body_ind,
                self.J,
                self.J_div_m,
                self.w,
                self.d,
                h
            ],
            outputs=[
                self.lambdas,
                self.bodies.phi,
                self.bodies.phi_dt
            ],
            device=DEVICE.GPU,
            record_tape=False
        )
        wp.synchronize()
        # print("phi: ",self.bodies.phi.numpy())
        # print("phi_dt: ",self.bodies.phi_dt.numpy())
        # print("lambdas: ",self.lambdas.numpy())

    def solveVelocity(self, h):
        self.d.zero_()
        wp.launch(
            kernel=solveWorldRotate,
            dim=SIM_NUM,
            inputs=[
                self.con_count,
                self.body_ind,
                self.J,
                self.J_div_m,
                self.w,
                self.d,
                h
            ],
            outputs=[
                self.lambdas,
                self.bodies.phi,
                self.bodies.phi_dt
            ],
            device=DEVICE.GPU,
            record_tape=False
        )
        wp.synchronize()
        self.lambdas.zero_()
        # print("phi: ",self.bodies.phi.numpy())
        # print("lambdas: ",self.lambdas.numpy())