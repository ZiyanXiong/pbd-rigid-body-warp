import warp as wp
WORLD_FIX_CONSTRAINT = wp.constant(0)
WORLD_CONTACT_CONSTRAINT = wp.constant(1)
WORLD_ROTATE_CONSTRAINT = wp.constant(2)
WORLD_ROTATE_TARGET_CONSTRAINT = wp.constant(3)
BODY_FIX_CONSTRAINT = wp.constant(4)
BODY_CONTACT_CONSTRAINT = wp.constant(5)
BODY_ROTATE_CONSTRAINT = wp.constant(6)
BODY_ROTATE_TARGET_CONSTRAINT = wp.constant(7)
BODY_MUSCLE_CONSTRAINT = wp.constant(8)

import numpy as np
from DataTypes import *
from CollisionDetection import collisionDetectionGroundCuboid, collisionDetectionCuboidCuboid
from utils import SIM_NUM, CONTACT_MAX, EPS_SMALL, VIA_POINT_MAX, getContactFrame, gamma

@wp.kernel
def initConstraintsContact(
    con_count: wp.array(dtype=INT_DATA_TYPE),
    con_type: wp.array(dtype=INT_DATA_TYPE, ndim=2),
    body1_ind: wp.array(dtype=INT_DATA_TYPE, ndim=2),
    body2_ind: wp.array(dtype=INT_DATA_TYPE, ndim=2),
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
    if(con_type[i][j] == WORLD_CONTACT_CONSTRAINT):
        b_i = body2_ind[i][j]
        cf = getContactFrame(normal[i][j])
        x_w = wp.transform_vector(q[b_i][j], xl2[i][j])
        J2[i][j] = cf * gamma(x_w)

        w_kernel = Vec3()
        J_div_m_kernel = Mat36()
        M_r = Vec3(I[b_i][0],I[b_i][1],I[b_i][2])
        M_p = I[b_i][3]
        for k in range(3):
            # wp.printf("c_fk: %f %f %f\n", cf[k][0], cf[k][1], cf[k][2])
            n_l = wp.transform_vector(wp.transform_inverse(q[b_i][j]), cf[k])
            # wp.printf("n_l: %f %f %f\n", n_l[0], n_l[1], n_l[2])
            rxn_l = wp.cross(xl2[i][j], n_l)
            # wp.printf("rxn_l: %f %f %f\n", rxn_l[0], rxn_l[1], rxn_l[2])
            w_kernel[k] = wp.dot(rxn_l, wp.cw_div(rxn_l, M_r))  + 1.0 / M_p
            rxnI = wp.transform_vector(q[b_i][j], wp.cw_div(rxn_l, M_r))
            J_div_m_kernel[k] = Vec6(rxnI[0], rxnI[1], rxnI[2], cf[k][0] / M_p, cf[k][1] / M_p, cf[k][2] / M_p)

        w2[i][j] = w_kernel
        J_div_m2[i][j] = J_div_m_kernel
    elif(con_type[i][j] == BODY_CONTACT_CONSTRAINT):
        b1_i = body1_ind[i][j]
        b2_i = body2_ind[i][j]
        cf = getContactFrame(normal[i][j])
        J1[i][j] = cf * gamma(wp.transform_vector(q[b1_i][j], xl1[i][j]))
        J2[i][j] = cf * gamma(wp.transform_vector(q[b2_i][j], xl2[i][j]))
        
        w_kernel = Vec3()
        J_div_m_kernel = Mat36()
        M_r = Vec3(I[b1_i][0],I[b1_i][1],I[b1_i][2])
        M_p = I[b1_i][3]
        for k in range(3):
            # wp.printf("c_fk: %f %f %f\n", cf[k][0], cf[k][1], cf[k][2])
            n_l = wp.transform_vector(wp.transform_inverse(q[b1_i][j]), cf[k])
            # wp.printf("n_l: %f %f %f\n", n_l[0], n_l[1], n_l[2])
            rxn_l = wp.cross(xl1[i][j], n_l)
            # wp.printf("rxn_l: %f %f %f\n", rxn_l[0], rxn_l[1], rxn_l[2])
            w_kernel[k] = wp.dot(rxn_l, wp.cw_div(rxn_l, M_r))  + 1.0 / M_p
            rxnI = wp.transform_vector(q[b1_i][j], wp.cw_div(rxn_l, M_r))
            J_div_m_kernel[k] = Vec6(rxnI[0], rxnI[1], rxnI[2], cf[k][0] / M_p, cf[k][1] / M_p, cf[k][2] / M_p)
        w1[i][j] = w_kernel
        J_div_m1[i][j] = J_div_m_kernel
        
        M_r = Vec3(I[b2_i][0],I[b2_i][1],I[b2_i][2])
        M_p = I[b2_i][3]
        for k in range(3):
            # wp.printf("c_fk: %f %f %f\n", cf[k][0], cf[k][1], cf[k][2])
            n_l = wp.transform_vector(wp.transform_inverse(q[b2_i][j]), cf[k])
            # wp.printf("n_l: %f %f %f\n", n_l[0], n_l[1], n_l[2])
            rxn_l = wp.cross(xl2[i][j], n_l)
            # wp.printf("rxn_l: %f %f %f\n", rxn_l[0], rxn_l[1], rxn_l[2])
            w_kernel[k] = wp.dot(rxn_l, wp.cw_div(rxn_l, M_r))  + 1.0 / M_p
            rxnI = wp.transform_vector(q[b2_i][j], wp.cw_div(rxn_l, M_r))
            J_div_m_kernel[k] = Vec6(rxnI[0], rxnI[1], rxnI[2], cf[k][0] / M_p, cf[k][1] / M_p, cf[k][2] / M_p)
        w2[i][j] = w_kernel
        J_div_m2[i][j] = J_div_m_kernel

@wp.kernel
def solveConstraintsContact(
    con_count: wp.array(dtype=INT_DATA_TYPE),
    con_type: wp.array(dtype=INT_DATA_TYPE, ndim=2),
    body1_ind: wp.array(dtype=INT_DATA_TYPE, ndim=2),
    body2_ind: wp.array(dtype=INT_DATA_TYPE, ndim=2),
    J1: wp.array(dtype=Mat36, ndim=2),
    J_div_m1: wp.array(dtype=Mat36, ndim=2),
    w1: wp.array(dtype=Vec3, ndim=2),
    J2: wp.array(dtype=Mat36, ndim=2),
    J_div_m2: wp.array(dtype=Mat36, ndim=2),
    w2: wp.array(dtype=Vec3, ndim=2),
    d: wp.array(dtype=FP_DATA_TYPE, ndim=2),
    h: FP_DATA_TYPE,
    mu: wp.array(dtype=FP_DATA_TYPE, ndim=2),
    #outputs
    lambdas: wp.array(dtype=Vec3, ndim=2),
    phi: wp.array(dtype=Vec6, ndim=2),
    phi_dt: wp.array(dtype=Vec6, ndim=2),
):
    j = wp.tid()
    for i in range(con_count[j]):
        if(con_type[i][j] == WORLD_CONTACT_CONSTRAINT):
            b_i = body2_ind[i][j]
            d_ij = d[i][j]
            phi_ij = phi[b_i][j]
            w_ij = w2[i][j]
            lambdas_ij=lambdas[i][j]
            bias = phi_dt[b_i][j] / h
            J_ij = J2[i][j]
            J_div_m_ij = J_div_m2[i][j]

            c = -wp.dot(J_ij[0], (phi_ij +bias)) + d_ij / h
            # wp.printf("c: %f = %f + %f\n", c, wp.dot(J_ij[0], (phi_ij +bias)), d_ij / h)
            # wp.printf("d_ij: %f \n", d_ij[k])
            # wp.printf("bias: %f %f %f %f %f %f\n", bias[0], bias[1], bias[2], bias[3], bias[4], bias[5])
            # wp.printf("phi: %f %f %f %f %f %f\n", phi_ij[0], phi_ij[1], phi_ij[2], phi_ij[3], phi_ij[4], phi_ij[5])
            # wp.printf("Jijk: %f %f %f %f %f %f\n", J_ij[k][0], J_ij[k][1], J_ij[k][2], J_ij[k][3], J_ij[k][4], J_ij[k][5])
            dlambda_nor = -c / w_ij[0]
            # wp.printf("dlambda: %f \n", dlambda)
            if(lambdas_ij[0] + dlambda_nor < 0.0):
                dlambda_nor = -lambdas_ij[0]
            lambdas_ij[0] += dlambda_nor
            phi_ij -= J_div_m_ij[0] * dlambda_nor
            
            c1 = -wp.dot(J_ij[1], (phi_ij +bias)) 
            c2 = -wp.dot(J_ij[2], (phi_ij +bias)) 
            # wp.printf("c: %f = %f + %f\n", c, wp.dot(J_ij[k], (phi_ij +bias)), d_ij[k] / h)
            # wp.printf("d_ij: %f \n", d_ij[k])
            # wp.printf("bias: %f %f %f %f %f %f\n", bias[0], bias[1], bias[2], bias[3], bias[4], bias[5])
            # wp.printf("phi: %f %f %f %f %f %f\n", phi_ij[0], phi_ij[1], phi_ij[2], phi_ij[3], phi_ij[4], phi_ij[5])
            # wp.printf("Jijk: %f %f %f %f %f %f\n", J_ij[k][0], J_ij[k][1], J_ij[k][2], J_ij[k][3], J_ij[k][4], J_ij[k][5])
            dlambda_tan1 = -c1 / w_ij[1]
            dlambda_tan2 = -c2 / w_ij[2]
            # wp.printf("dlambda: %f \n", dlambda)
            lambda_tan = Vec2(lambdas_ij[1]+ dlambda_tan1, lambdas_ij[2]+ dlambda_tan2)
            lambda_tan_norm = wp.math.norm_l2(lambda_tan)
            if(lambda_tan_norm > wp.max(EPS_SMALL, mu[i][j] * lambdas_ij[0])):
                dlambda_tan1 = mu[i][j] * lambdas_ij[0] * lambda_tan[0] / lambda_tan_norm - lambdas_ij[1]
                dlambda_tan2 = mu[i][j] * lambdas_ij[0] * lambda_tan[1] / lambda_tan_norm - lambdas_ij[2]
            lambdas_ij[1] += dlambda_tan1
            lambdas_ij[2] += dlambda_tan2
            phi_ij -= J_div_m_ij[1] * dlambda_tan1 + J_div_m_ij[2] * dlambda_tan2
            lambdas[i][j] = lambdas_ij
            phi[b_i][j] = phi_ij
        elif(con_type[i][j] == BODY_CONTACT_CONSTRAINT):
            b1_i = body1_ind[i][j]
            b2_i = body2_ind[i][j]
            d_ij = d[i][j]
            phi1_ij = phi[b1_i][j]
            phi2_ij = phi[b2_i][j]
            w_ij = w1[i][j] + w2[i][j]
            lambdas_ij=lambdas[i][j]
            bias1 = phi_dt[b1_i][j] / h
            bias2 = phi_dt[b2_i][j] / h
            J1_ij = J1[i][j]
            J_div_m1_ij = J_div_m1[i][j]
            J2_ij = J2[i][j]
            J_div_m2_ij = J_div_m2[i][j]

            c = wp.dot(J1_ij[0], (phi1_ij +bias1)) - wp.dot(J2_ij[0], (phi2_ij +bias2)) + d_ij / h
            # wp.printf("c: %f = %f - %f + %f \n", c, wp.dot(J1_ij[0], (phi1_ij +bias1)), wp.dot(J2_ij[0], (phi2_ij +bias2)), d_ij / h)
            # wp.printf("w_ij: %f \n", w_ij[k])
            # wp.printf("d_ij: %f \n", d_ij[k])
            # wp.printf("bias2: %f %f %f %f %f %f\n", bias2[0], bias2[1], bias2[2], bias2[3], bias2[4], bias2[5])
            # wp.printf("phi1: %f %f %f %f %f %f\n", phi1_ij[0], phi1_ij[1], phi1_ij[2], phi1_ij[3], phi1_ij[4], phi1_ij[5])
            # wp.printf("phi2: %f %f %f %f %f %f\n", phi2_ij[0], phi2_ij[1], phi2_ij[2], phi2_ij[3], phi2_ij[4], phi2_ij[5])
            # wp.printf("J1ijk: %f %f %f %f %f %f\n", J1_ij[k][0], J1_ij[k][1], J1_ij[k][2], J1_ij[k][3], J1_ij[k][4], J1_ij[k][5])
            # wp.printf("J2ij0: %f %f %f %f %f %f\n", J2_ij[0][0], J2_ij[0][1], J2_ij[0][2], J2_ij[0][3], J2_ij[0][4], J2_ij[0][5])
            dlambda_nor = -c / w_ij[0]
            # wp.printf("dlambda: %f \n", dlambda)
            if(lambdas_ij[0] + dlambda_nor < 0.0):
                dlambda_nor = -lambdas_ij[0]
            lambdas_ij[0] += dlambda_nor
            phi1_ij += J_div_m1_ij[0] * dlambda_nor
            phi2_ij -= J_div_m2_ij[0] * dlambda_nor
            
            c1 = wp.dot(J1_ij[1], (phi1_ij +bias1)) - wp.dot(J2_ij[1], (phi2_ij +bias2))
            c2 = wp.dot(J1_ij[2], (phi1_ij +bias1)) - wp.dot(J2_ij[2], (phi2_ij +bias2))
            # wp.printf("c: %f = %f - %f\n", c, wp.dot(J1_ij[k], (phi1_ij +bias1)), wp.dot(J2_ij[k], (phi2_ij +bias2)))
            # wp.printf("w_ij: %f \n", w_ij[k])
            # wp.printf("d_ij: %f \n", d_ij[k])
            # wp.printf("bias: %f %f %f %f %f %f\n", bias[0], bias[1], bias[2], bias[3], bias[4], bias[5])
            # wp.printf("phi1: %f %f %f %f %f %f\n", phi1_ij[0], phi1_ij[1], phi1_ij[2], phi1_ij[3], phi1_ij[4], phi1_ij[5])
            # wp.printf("phi2: %f %f %f %f %f %f\n", phi2_ij[0], phi2_ij[1], phi2_ij[2], phi2_ij[3], phi2_ij[4], phi2_ij[5])
            # wp.printf("J1ijk: %f %f %f %f %f %f\n", J1_ij[k][0], J1_ij[k][1], J1_ij[k][2], J1_ij[k][3], J1_ij[k][4], J1_ij[k][5])
            # wp.printf("J2ijk: %f %f %f %f %f %f\n", J2_ij[k][0], J2_ij[k][1], J2_ij[k][2], J2_ij[k][3], J2_ij[k][4], J2_ij[k][5])
            dlambda_tan1 = -c1 / w_ij[1]
            dlambda_tan2 = -c2 / w_ij[2]
            # wp.printf("dlambda: %f \n", dlambda)
            lambda_tan = Vec2(lambdas_ij[1]+ dlambda_tan1, lambdas_ij[2]+ dlambda_tan2)
            lambda_tan_norm = wp.math.norm_l2(lambda_tan)
            if(lambda_tan_norm > wp.max(EPS_SMALL, mu[i][j] * lambdas_ij[0])):
                dlambda_tan1 = mu[i][j] * lambdas_ij[0] * lambda_tan[0] / lambda_tan_norm - lambdas_ij[1]
                dlambda_tan2 = mu[i][j] * lambdas_ij[0] * lambda_tan[1] / lambda_tan_norm - lambdas_ij[2]
            lambdas_ij[1] += dlambda_tan1
            lambdas_ij[2] += dlambda_tan2
            phi1_ij += J_div_m1_ij[1] * dlambda_tan1 + J_div_m1_ij[2] * dlambda_tan2
            phi2_ij -= J_div_m2_ij[1] * dlambda_tan1 + J_div_m2_ij[2] * dlambda_tan2
            lambdas[i][j] = lambdas_ij
            phi[b1_i][j] = phi1_ij
            phi[b2_i][j] = phi2_ij
            
@wp.kernel
def initConstraintsJoint(
    con_count: INT_DATA_TYPE,
    con_type: wp.array(dtype=INT_DATA_TYPE),
    body1_ind: wp.array(dtype=INT_DATA_TYPE),
    body2_ind: wp.array(dtype=INT_DATA_TYPE),
    phi: wp.array(dtype=Vec6, ndim=2),
    q: wp.array(dtype=Transform, ndim=2),
    I: wp.array(dtype=Vec6),
    xl1: wp.array(dtype=Vec3),
    xl2: wp.array(dtype=Vec3),
    axis1: wp.array(dtype=Vec3),
    axis2: wp.array(dtype=Vec3),
    b1: wp.array(dtype=Vec3),
    b2: wp.array(dtype=Vec3),
    theta_target: wp.array(dtype=FP_DATA_TYPE),
    oemga_target: wp.array(dtype=FP_DATA_TYPE),
    kp: wp.array(dtype=FP_DATA_TYPE),
    kd: wp.array(dtype=FP_DATA_TYPE),
    h: FP_DATA_TYPE,
    # outputs
    d: wp.array(dtype=Vec3, ndim=2),
    J1: wp.array(dtype=Mat36, ndim=2),
    J_div_m1: wp.array(dtype=Mat36, ndim=2),
    w1: wp.array(dtype=Vec3, ndim=2),
    J2: wp.array(dtype=Mat36, ndim=2),
    J_div_m2: wp.array(dtype=Mat36, ndim=2),
    w2: wp.array(dtype=Vec3, ndim=2),
):
    i,j = wp.tid()
    if i > con_count:
        return
    
    if(con_type[i] == WORLD_FIX_CONSTRAINT):
        b_i = body2_ind[i]
        cf = getContactFrame(axis1[i])
        x_w = wp.transform_vector(q[b_i][j], xl2[i])
        # q_ij = q[b_i][j]
        # wp.printf("q: %f %f %f %f %f %f %f\n", q_ij[0], q_ij[1], q_ij[2], q_ij[3], q_ij[4], q_ij[5], q_ij[6])
        # wp.printf("axis: %f %f %f\n", axis[i][0], axis[i][1], axis[i][2])
        # wp.printf("x_l: %f %f %f\n", xl[i][0], xl[i][1], xl[i][2])
        # wp.printf("x_w: %f %f %f\n", x_w[0], x_w[1], x_w[2])

        d[i][j] = cf * (xl1[i] - x_w - q[b_i][j].p)
        J2[i][j] = cf * gamma(x_w)
        # J2_ij = J2[i][j]
        # wp.printf("J2:\n %f %f %f %f %f %f\n", J2_ij[0][0], J2_ij[0][1], J2_ij[0][2], J2_ij[0][3], J2_ij[0][4], J2_ij[0][5])
        # wp.printf("%f %f %f %f %f %f\n", J2_ij[1][0], J2_ij[1][1], J2_ij[1][2], J2_ij[1][3], J2_ij[1][4], J2_ij[1][5])
        # wp.printf("%f %f %f %f %f %f\n", J2_ij[2][0], J2_ij[2][1], J2_ij[2][2], J2_ij[2][3], J2_ij[2][4], J2_ij[2][5])

        w_kernel = Vec3()
        J_div_m_kernel = Mat36()
        M_r = Vec3(I[b_i][0],I[b_i][1],I[b_i][2])
        M_p = I[b_i][3]
        for k in range(3):
            # wp.printf("c_fk: %f %f %f\n", cf[k][0], cf[k][1], cf[k][2])
            n_l = wp.transform_vector(wp.transform_inverse(q[b_i][j]), cf[k])
            # wp.printf("n_l: %f %f %f\n", n_l[0], n_l[1], n_l[2])
            rxn_l = wp.cross(xl2[i], n_l)
            # wp.printf("rxn_l: %f %f %f\n", rxn_l[0], rxn_l[1], rxn_l[2])
            w_kernel[k] = wp.dot(rxn_l, wp.cw_div(rxn_l, M_r))  + 1.0 / M_p
            rxnI = wp.transform_vector(q[b_i][j], wp.cw_div(rxn_l, M_r))
            J_div_m_kernel[k] = Vec6(rxnI[0], rxnI[1], rxnI[2], cf[k][0] / M_p, cf[k][1] / M_p, cf[k][2] / M_p)

        w2[i][j] = w_kernel
        J_div_m2[i][j] = J_div_m_kernel
    elif(con_type[i] == BODY_FIX_CONSTRAINT):
        b1_i = body1_ind[i]
        b2_i = body2_ind[i]
        cf = getContactFrame(wp.transform_vector(q[b1_i][j], axis1[i]))
        d[i][j] = cf * (wp.transform_point(q[b1_i][j], xl1[i]) - wp.transform_point(q[b2_i][j], xl2[i]))
        J1[i][j] = cf * gamma(wp.transform_vector(q[b1_i][j], xl1[i]))
        J2[i][j] = cf * gamma(wp.transform_vector(q[b2_i][j], xl2[i]))
        
        w_kernel = Vec3()
        J_div_m_kernel = Mat36()
        M_r = Vec3(I[b1_i][0],I[b1_i][1],I[b1_i][2])
        M_p = I[b1_i][3]
        for k in range(3):
            # wp.printf("c_fk: %f %f %f\n", cf[k][0], cf[k][1], cf[k][2])
            n_l = wp.transform_vector(wp.transform_inverse(q[b1_i][j]), cf[k])
            # wp.printf("n_l: %f %f %f\n", n_l[0], n_l[1], n_l[2])
            rxn_l = wp.cross(xl1[i], n_l)
            # wp.printf("rxn_l: %f %f %f\n", rxn_l[0], rxn_l[1], rxn_l[2])
            w_kernel[k] = wp.dot(rxn_l, wp.cw_div(rxn_l, M_r))  + 1.0 / M_p
            rxnI = wp.transform_vector(q[b1_i][j], wp.cw_div(rxn_l, M_r))
            J_div_m_kernel[k] = Vec6(rxnI[0], rxnI[1], rxnI[2], cf[k][0] / M_p, cf[k][1] / M_p, cf[k][2] / M_p)
        w1[i][j] = w_kernel
        J_div_m1[i][j] = J_div_m_kernel
        
        M_r = Vec3(I[b2_i][0],I[b2_i][1],I[b2_i][2])
        M_p = I[b2_i][3]
        for k in range(3):
            # wp.printf("c_fk: %f %f %f\n", cf[k][0], cf[k][1], cf[k][2])
            n_l = wp.transform_vector(wp.transform_inverse(q[b2_i][j]), cf[k])
            # wp.printf("n_l: %f %f %f\n", n_l[0], n_l[1], n_l[2])
            rxn_l = wp.cross(xl2[i], n_l)
            # wp.printf("rxn_l: %f %f %f\n", rxn_l[0], rxn_l[1], rxn_l[2])
            w_kernel[k] = wp.dot(rxn_l, wp.cw_div(rxn_l, M_r))  + 1.0 / M_p
            rxnI = wp.transform_vector(q[b2_i][j], wp.cw_div(rxn_l, M_r))
            J_div_m_kernel[k] = Vec6(rxnI[0], rxnI[1], rxnI[2], cf[k][0] / M_p, cf[k][1] / M_p, cf[k][2] / M_p)
        w2[i][j] = w_kernel
        J_div_m2[i][j] = J_div_m_kernel
    elif(con_type[i] == WORLD_ROTATE_CONSTRAINT or con_type[i] == WORLD_ROTATE_TARGET_CONSTRAINT):
            b_i = body2_ind[i]
            cf = getContactFrame(axis1[i])

            w_kernel = Vec3()
            J_div_m_kernel = Mat36()
            J_kernel = Mat36()
            M_r = Vec3(I[b_i][0],I[b_i][1],I[b_i][2])
            d_ij = Vec3(0.0)
            for k in range(3):
                n_l = wp.transform_vector(wp.transform_inverse(q[b_i][j]), cf[k])
                w_kernel[k] = wp.dot(n_l, wp.cw_div(n_l, M_r))
                rxnI = wp.transform_vector(q[b_i][j], wp.cw_div(n_l, M_r))
                J_div_m_kernel[k] = Vec6(rxnI[0], rxnI[1], rxnI[2], 0.0, 0.0, 0.0)
                J_kernel[k] = Vec6(cf[k][0], cf[k][1], cf[k][2], 0.0, 0.0, 0.0)

            w2[i][j] = w_kernel
            J2[i][j] = J_kernel
            J_div_m2[i][j] = J_div_m_kernel

            # r_axis, dtheta = wp.quat_to_axis_angle(wp.quat_inverse(q[b_i][j].q))
            # # wp.printf("dq: %f %f %f %f\n", q[b_i][j].q[0], q[b_i][j].q[1], q[b_i][j].q[2], q[b_i][j].q[3])
            # d[i][j] = cf * r_axis * dtheta
            d_ij = cf * wp.cross(wp.transform_vector(q[b_i][j], axis2[i]), axis1[i])
            if(con_type[i] == WORLD_ROTATE_TARGET_CONSTRAINT):
                # dtheta_align = wp.cross(wp.transform_vector(q[b_i][j], axis2[i]), axis1[i])
                # dtheta_align_norm = wp.norm_l2(dtheta_align)
                # if dtheta_align_norm > EPS_SMALL:
                #     dq_align = wp.quat_from_axis_angle(dtheta_align / dtheta_align_norm, dtheta_align_norm)
                # else:
                #     dq_align = wp.quat_identity()
                # d_target = wp.dot(cf[0], wp.cross(wp.quat_rotate(dq_align * q[b_i][j].q, b2[i]), b1[i]))
                d_target = wp.asin(wp.dot(cf[0], wp.cross(wp.quat_rotate(q[b_i][j].q, b2[i]), b1[i])))
                d_omega = -wp.dot(J_kernel[0], phi[b_i][j])
                a = h/(h*(h*kp[i]+kd[i])*wp.dot(J_kernel[0], -J_div_m_kernel[0]) + 1.0)
                d_ij[0] = a * h * (kp[i]*((d_target - theta_target[i]) + d_omega * h) + kd[i]*(d_omega - oemga_target[i])) - d_omega * h
                # wp.printf("d_target: %f\n", d_target)
                # wp.printf("d_omega: %f\n", d_omega) 
                # wp.printf("a: %f\n", a) 
                # wp.printf("dq_align: %f %f %f %f\n", dq_align[0], dq_align[1], dq_align[2], dq_align[3]) 
                # wp.printf("kp: %f\n", kp[i]) 
                # wp.printf("kd: %f\n", kd[i]) 
                # wp.printf("h*kp[i]+kd[i]: %f\n", h*kp[i]+kd[i]) 
                # wp.printf("wp.dot(J_kernel[0], -J_div_m_kernel[0]): %f\n", wp.dot(J_kernel[0], -J_div_m_kernel[0]))   
            d[i][j] = Vec3(d_ij[0], wp.asin(d_ij[1]), wp.asin(d_ij[2]))
            
    elif(con_type[i] == BODY_ROTATE_CONSTRAINT or con_type[i] == BODY_ROTATE_TARGET_CONSTRAINT):
            b1_i = body1_ind[i]
            b2_i = body2_ind[i]
            cf = getContactFrame(wp.transform_vector(q[b1_i][j], axis1[i]))
            w_kernel = Vec3()
            J_div_m_kernel = Mat36()
            J_kernel = Mat36()
            
            M_r = Vec3(I[b1_i][0],I[b1_i][1],I[b1_i][2])
            for k in range(3):
                n_l = wp.transform_vector(wp.transform_inverse(q[b1_i][j]), cf[k])
                w_kernel[k] = wp.dot(n_l, wp.cw_div(n_l, M_r))
                rxnI = wp.transform_vector(q[b1_i][j], wp.cw_div(n_l, M_r))
                J_div_m_kernel[k] = Vec6(rxnI[0], rxnI[1], rxnI[2], 0.0, 0.0, 0.0)
                J_kernel[k] = Vec6(cf[k][0], cf[k][1], cf[k][2], 0.0, 0.0, 0.0)
            w1[i][j] = w_kernel
            J1[i][j] = J_kernel
            J_div_m1[i][j] = J_div_m_kernel
            
            M_r = Vec3(I[b2_i][0],I[b2_i][1],I[b2_i][2])
            for k in range(3):
                n_l = wp.transform_vector(wp.transform_inverse(q[b2_i][j]), cf[k])
                w_kernel[k] = wp.dot(n_l, wp.cw_div(n_l, M_r))
                rxnI = wp.transform_vector(q[b2_i][j], wp.cw_div(n_l, M_r))
                J_div_m_kernel[k] = Vec6(rxnI[0], rxnI[1], rxnI[2], 0.0, 0.0, 0.0)
                J_kernel[k] = Vec6(cf[k][0], cf[k][1], cf[k][2], 0.0, 0.0, 0.0)
            w2[i][j] = w_kernel
            J2[i][j] = J_kernel
            J_div_m2[i][j] = J_div_m_kernel
            
            d_ij = cf * wp.cross(wp.transform_vector(q[b2_i][j], axis2[i]), wp.transform_vector(q[b1_i][j], axis1[i]))
            if(con_type[i] == BODY_ROTATE_TARGET_CONSTRAINT):
                d_J_div_m = J_div_m1[i][j] - J_div_m2[i][j]
                # dtheta_align = wp.cross(wp.transform_vector(q[b2_i][j], axis2[i]), wp.transform_vector(q[b1_i][j], axis1[i]))
                # dtheta_align_norm = wp.norm_l2(dtheta_align)
                # if dtheta_align_norm > EPS_SMALL:
                #     dq_align = wp.quat_from_axis_angle(dtheta_align / dtheta_align_norm, dtheta_align_norm)
                # else:
                #     dq_align = wp.quat_identity()
                # d_target = wp.dot(cf[0], wp.cross(wp.quat_rotate(dq_align * q[b_i][j].q, b2[i]), wp.transform_vector(q[b1_i][j], b1[i])))
                d_target = wp.asin(wp.dot(cf[0], wp.cross(wp.transform_vector(q[b2_i][j], b2[i]), wp.transform_vector(q[b1_i][j], b1[i]))))
                d_omega = wp.dot(J_kernel[0], phi[b1_i][j] - phi[b2_i][j])          
                a = h/(h*(h*kp[i]+kd[i])*wp.dot(J_kernel[0], d_J_div_m[0]) + 1.0)
                d_ij[0] = a * h * (kp[i]*((d_target - theta_target[i]) + d_omega * h) + kd[i]*(d_omega - oemga_target[i])) - d_omega * h
                d[i][j] = Vec3(d_ij[0], wp.asin(d_ij[1]), wp.asin(d_ij[2]))
        
@wp.kernel
def solveConstraintsJoint(
    con_count: INT_DATA_TYPE,
    con_type: wp.array(dtype=INT_DATA_TYPE),
    body1_ind: wp.array(dtype=INT_DATA_TYPE),
    body2_ind: wp.array(dtype=INT_DATA_TYPE),
    J1: wp.array(dtype=Mat36, ndim=2),
    J_div_m1: wp.array(dtype=Mat36, ndim=2),
    w1: wp.array(dtype=Vec3, ndim=2),
    J2: wp.array(dtype=Mat36, ndim=2),
    J_div_m2: wp.array(dtype=Mat36, ndim=2),
    w2: wp.array(dtype=Vec3, ndim=2),
    d: wp.array(dtype=Vec3, ndim=2),
    h: FP_DATA_TYPE,
    #outputs
    lambdas: wp.array(dtype=Vec3, ndim=2),
    phi: wp.array(dtype=Vec6, ndim=2),
    phi_dt: wp.array(dtype=Vec6, ndim=2)
):
    j = wp.tid()

    for i in range(con_count):
        if(con_type[i] == WORLD_FIX_CONSTRAINT or con_type[i] == WORLD_ROTATE_CONSTRAINT or con_type[i] == WORLD_ROTATE_TARGET_CONSTRAINT):
            b_i = body2_ind[i]
            d_ij = d[i][j]
            phi_ij = phi[b_i][j]
            w_ij = w2[i][j]
            lambdas_ij=lambdas[i][j]
            bias = phi_dt[b_i][j] / h
            J_ij = J2[i][j]
            J_div_m_ij = J_div_m2[i][j]
            for k in range(3):
                if(con_type[i] == WORLD_ROTATE_CONSTRAINT and k==0):
                    continue
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
        elif(con_type[i] == BODY_FIX_CONSTRAINT or con_type[i] == BODY_ROTATE_CONSTRAINT or con_type[i] == BODY_ROTATE_TARGET_CONSTRAINT):
            b1_i = body1_ind[i]
            b2_i = body2_ind[i]
            d_ij = d[i][j]
            phi1_ij = phi[b1_i][j]
            phi2_ij = phi[b2_i][j]
            w_ij = w1[i][j] + w2[i][j]
            lambdas_ij=lambdas[i][j]
            bias1 = phi_dt[b1_i][j] / h
            bias2 = phi_dt[b2_i][j] / h
            J1_ij = J1[i][j]
            J_div_m1_ij = J_div_m1[i][j]
            J2_ij = J2[i][j]
            J_div_m2_ij = J_div_m2[i][j]
            for k in range(3):
                if(con_type[i] == BODY_ROTATE_CONSTRAINT and k==0):
                    continue
                c = wp.dot(J1_ij[k], (phi1_ij +bias1)) - wp.dot(J2_ij[k], (phi2_ij +bias2)) + d_ij[k] / h
                # wp.printf("c: %f = %f - %f\n", c, wp.dot(J1_ij[k], (phi1_ij +bias1)), wp.dot(J2_ij[k], (phi2_ij +bias2)))
                # wp.printf("w_ij: %f \n", w_ij[k])
                # wp.printf("d_ij: %f \n", d_ij[k])
                # wp.printf("bias: %f %f %f %f %f %f\n", bias[0], bias[1], bias[2], bias[3], bias[4], bias[5])
                # wp.printf("phi1: %f %f %f %f %f %f\n", phi1_ij[0], phi1_ij[1], phi1_ij[2], phi1_ij[3], phi1_ij[4], phi1_ij[5])
                # wp.printf("phi2: %f %f %f %f %f %f\n", phi2_ij[0], phi2_ij[1], phi2_ij[2], phi2_ij[3], phi2_ij[4], phi2_ij[5])
                # wp.printf("J1ijk: %f %f %f %f %f %f\n", J1_ij[k][0], J1_ij[k][1], J1_ij[k][2], J1_ij[k][3], J1_ij[k][4], J1_ij[k][5])
                # wp.printf("J2ijk: %f %f %f %f %f %f\n", J2_ij[k][0], J2_ij[k][1], J2_ij[k][2], J2_ij[k][3], J2_ij[k][4], J2_ij[k][5])
                dlambda = -c / w_ij[k]
                # wp.printf("dlambda: %f \n", dlambda)
                lambdas_ij[k] += dlambda
                phi1_ij += J_div_m1_ij[k] * dlambda
                phi2_ij -= J_div_m2_ij[k] * dlambda
            lambdas[i][j] = lambdas_ij
            phi[b1_i][j] = phi1_ij
            phi[b2_i][j] = phi2_ij
              
@wp.kernel
def initConstraintsMuscle(
    con_count: INT_DATA_TYPE,
    con_type: wp.array(dtype=INT_DATA_TYPE),
    body_num: wp.array(dtype=INT_DATA_TYPE),
    body_inds: wp.array(dtype=INT_DATA_TYPE, ndim=2),
    stiffness: wp.array(dtype=FP_DATA_TYPE),
    q: wp.array(dtype=Transform, ndim=2),
    I: wp.array(dtype=Vec6),
    xls: wp.array(dtype=Vec3, ndim=2),
    l: wp.array(dtype=FP_DATA_TYPE),
    h: FP_DATA_TYPE,
    # outputs
    alpha: wp.array(dtype=FP_DATA_TYPE),
    b: wp.array(dtype=FP_DATA_TYPE, ndim=2),
    Js: wp.array(dtype=Vec6, ndim=3),
    J_div_ms: wp.array(dtype=Vec6, ndim=3),
    w: wp.array(dtype=FP_DATA_TYPE, ndim=2),
):
    i,j = wp.tid()
    if i > con_count:
        return
    
    if(con_type[i] == BODY_MUSCLE_CONSTRAINT):
        alpha[i] = 1.0 / (stiffness[i] * h * h)
        normals = Mat3(0.0)
        b[i][j] = -l[i]
        for k in range(body_num[i]-1):
            b_i = body_inds[i][k]
            normal = wp.transform_point(q[b_i][j], xls[i][k]) - wp.transform_point(q[body_inds[i][k+1]][j], xls[i][k+1])
            normal_len = wp.norm_l2(normal)
            if normal_len < EPS_SMALL:
                return
            normal = normal / normal_len
            normals[k] = normals[k] + normal
            normals[k+1] = normals[k+1] - normal
            b[i][j] = b[i][j] + normal_len
            
        w[i][j] = 0.0
        for k in range(body_num[i]):
            M_r = Vec3(I[b_i][0],I[b_i][1],I[b_i][2])
            M_p = I[b_i][3]
            # wp.printf("c_fk: %f %f %f\n", cf[k][0], cf[k][1], cf[k][2])
            n_l = wp.transform_vector(wp.transform_inverse(q[b_i][j]), normals[k])
            # wp.printf("n_l: %f %f %f\n", n_l[0], n_l[1], n_l[2])
            rxn_l = wp.cross(xls[i][k], n_l)
            # wp.printf("rxn_l: %f %f %f\n", rxn_l[0], rxn_l[1], rxn_l[2])
            w[i][j] += wp.dot(rxn_l, wp.cw_div(rxn_l, M_r))  + 1.0 / M_p
            rxn = wp.transform_vector(q[b_i][j], rxn_l)
            Js[i][j][k] = Vec6(rxn[0], rxn[1], rxn[2], normals[k][0], normals[k][1], normals[k][2])
            rxnI = wp.transform_vector(q[b_i][j], wp.cw_div(rxn_l, M_r))
            J_div_ms[i][j][k] = Vec6(rxnI[0], rxnI[1], rxnI[2], normals[k][0] / M_p, normals[k][1] / M_p, normals[k][2] / M_p)
            
        b[i][j] = b[i][j] / h

@wp.kernel
def solveConstraintsMuscle(
    con_count: INT_DATA_TYPE,
    con_type: wp.array(dtype=INT_DATA_TYPE),
    body_num: wp.array(dtype=INT_DATA_TYPE),
    body_inds: wp.array(dtype=INT_DATA_TYPE, ndim=2),
    alpha: wp.array(dtype=FP_DATA_TYPE),
    Js: wp.array(dtype=Vec6, ndim=3),
    J_div_ms: wp.array(dtype=Vec6, ndim=3),
    w: wp.array(dtype=FP_DATA_TYPE, ndim=2),
    b: wp.array(dtype=FP_DATA_TYPE, ndim=2),
    h: FP_DATA_TYPE,
    #outputs
    lambdas: wp.array(dtype=FP_DATA_TYPE, ndim=2),
    phi: wp.array(dtype=Vec6, ndim=2),
):
    j = wp.tid()

    for i in range(con_count):
        if(con_type[i] == BODY_MUSCLE_CONSTRAINT):
            c = b[i][j] + alpha[i] * lambdas[i][j]
            for k in range(body_num[i]):
                b_i = body_inds[i][k]
                c += wp.dot(Js[i][j][k], phi[b_i][j])
            dlambda = -c / (w[i][j] + alpha[i])
            # wp.printf("alpha: %f \n", alpha[i])
            # wp.printf("c: %f = %f + %f + %f + %f \n", c, wp.dot(J1[i][j], phi[b1_i][j]), -wp.dot(J2[i][j], phi[b2_i][j]), b[i][j], alpha[i] * lambdas[i][j])
            # wp.printf("c: %f\n", c)
            # wp.printf("b: %f\n", b[i][j])
            # wp.printf("w_ij: %f\n", w[i][j] + alpha[i])
            # wp.printf("dlambda: %f \n", dlambda)
            lambdas[i][j] += dlambda
            for k in range(body_num[i]):
                b_i = body_inds[i][k]
                phi[b_i][j] += J_div_ms[i][j][k] * dlambda

class ConstraintsOnDevice():
    def __init__(self):
        self.con_count = wp.zeros(shape=SIM_NUM, dtype=INT_DATA_TYPE)

    def update(self, step):
        pass

    def computeC(self):
        pass
    
    def init(self):
        pass

    def solve(self):
        pass

    def solveVelocity(self):
        pass

class ConstraintsContact(ConstraintsOnDevice):
    def __init__(self, fixed, bodies_on_host, bodies_on_device, body_pairs):
        super().__init__()
        self.fixed = fixed
        self.bodies_on_host = bodies_on_host
        self.bodies = bodies_on_device
        self.body_pairs = body_pairs
        self.con_type = wp.empty(shape=(CONTACT_MAX, SIM_NUM), dtype=INT_DATA_TYPE)
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

    def update(self, step):
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
                                                self.con_type,
                                                self.body_ind2,
                                                self.d,
                                                self.normal,
                                                self.xl2,
                                                self.mu)
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
                                            self.con_type,
                                            self.body_ind1,
                                            self.body_ind2,
                                            self.d,
                                            self.normal,
                                            self.xl1,
                                            self.xl2,
                                            self.mu)
            # print("con_count: ",self.con_count.numpy())
            # print("con_type: ",self.con_type.numpy())
            # print("d: ",self.d.numpy())
            # print("normal: ",self.normal.numpy())
            # print("xl1: ",self.xl1.numpy())
            # print("xl2: ",self.xl2.numpy())

    def computeC(self):
        # Compute contact constraints between ground and bodies
        pass

    def init(self, h):
        wp.launch(
            kernel=initConstraintsContact,
            dim=(CONTACT_MAX, SIM_NUM),
            inputs=[
                self.con_count,
                self.con_type,
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
            record_tape=False,
        )
        # print("J1: ",self.J1.numpy())
        # print("J_div_m1: ",self.J_div_m1.numpy())
        # print("w1: ",self.w1.numpy())
        # print("J2: ",self.J2.numpy())
        # print("J_div_m2: ",self.J_div_m2.numpy())
        # print("w2: ",self.w2.numpy())

    def solve(self, h):
        wp.launch(
            kernel=solveConstraintsContact,
            dim=SIM_NUM,
            inputs=[
                self.con_count,
                self.con_type,
                self.body_ind1,
                self.body_ind2,
                self.J1,
                self.J_div_m1,
                self.w1,
                self.J2,
                self.J_div_m2,
                self.w2,
                self.d,
                h,
                self.mu,
            ],
            outputs=[
                self.lambdas,
                self.bodies.phi,
                self.bodies.phi_dt
            ],
            record_tape=False
        )
        wp.synchronize()
        # print("phi: ",self.bodies.phi.numpy())
        # print("lambdas: ",self.lambdas.numpy())

    def solveVelocity(self, h):
        self.solve(h)

class ConstraintsJoint(ConstraintsOnDevice):
    def __init__(self, joints, bodies_on_device):
        self.con_count = 2 * len(joints)
        self.bodies = bodies_on_device
        self.joints = joints

        body1_ind = np.zeros(shape=self.con_count, dtype=INT_DATA_TYPE)
        body2_ind = np.zeros(shape=self.con_count, dtype=INT_DATA_TYPE)
        constraint_type = np.empty(shape=self.con_count, dtype=INT_DATA_TYPE)
        xl1 = np.empty(shape=self.con_count, dtype=Vec3)
        xl2 = np.empty(shape=self.con_count, dtype=Vec3)
        axis1 = np.empty(shape=self.con_count, dtype=Vec3)
        axis2 = np.empty(shape=self.con_count, dtype=Vec3)
        b1 = np.empty(shape=self.con_count, dtype=Vec3)
        b2 = np.empty(shape=self.con_count, dtype=Vec3)
        kp = np.zeros(shape=self.con_count, dtype=FP_DATA_TYPE)
        kd = np.zeros(shape=self.con_count, dtype=FP_DATA_TYPE)
        ci = 0
        
        for i in range(len(joints)):
            if(joints[i].parent is None):
                body2_ind[ci] = joints[i].child
                xl1[ci] = joints[i].xl1
                xl2[ci] = joints[i].xl2
                axis1[ci] = joints[i].axis1
                axis2[ci] = joints[i].axis2
                b1[ci] = joints[i].b1
                b2[ci] = joints[i].b2
                constraint_type[ci] = WORLD_FIX_CONSTRAINT
                ci +=1
                
                body2_ind[ci] = joints[i].child
                xl1[ci] = joints[i].xl1
                xl2[ci] = joints[i].xl2
                axis1[ci] = joints[i].axis1
                axis2[ci] = joints[i].axis2
                b1[ci] = joints[i].b1
                b2[ci] = joints[i].b2
                if(joints[i].has_target):
                    constraint_type[ci] = WORLD_ROTATE_TARGET_CONSTRAINT
                    kp[ci] = joints[i].kp
                    kd[ci] = joints[i].kd
                else:
                    constraint_type[ci] = WORLD_ROTATE_CONSTRAINT
                ci +=1
            else:
                body1_ind[ci] = joints[i].parent
                body2_ind[ci] = joints[i].child
                xl1[ci] = joints[i].xl1
                xl2[ci] = joints[i].xl2
                axis1[ci] = joints[i].axis1
                axis2[ci] = joints[i].axis2
                b1[ci] = joints[i].b1
                b2[ci] = joints[i].b2
                constraint_type[ci] = BODY_FIX_CONSTRAINT
                ci +=1
                
                body1_ind[ci] = joints[i].parent
                body2_ind[ci] = joints[i].child
                xl1[ci] = joints[i].xl1
                xl2[ci] = joints[i].xl2
                axis1[ci] = joints[i].axis1
                axis2[ci] = joints[i].axis2
                b1[ci] = joints[i].b1
                b2[ci] = joints[i].b2
                if(joints[i].has_target):
                    constraint_type[ci] = BODY_ROTATE_TARGET_CONSTRAINT
                    kp[ci] = joints[i].kp
                    kd[ci] = joints[i].kd
                else:
                    constraint_type[ci] = BODY_ROTATE_CONSTRAINT
                ci +=1
                

        self.body1_ind = wp.array(body1_ind, dtype=INT_DATA_TYPE)
        self.body2_ind = wp.array(body2_ind, dtype=INT_DATA_TYPE)
        self.xl1 = wp.array(xl1, dtype=Vec3)
        self.xl2 = wp.array(xl2, dtype=Vec3)
        self.axis1 = wp.array(axis1, dtype=Vec3)
        self.axis2 = wp.array(axis2, dtype=Vec3)
        self.b1 = wp.array(b1, dtype=Vec3)
        self.b2 = wp.array(b2, dtype=Vec3)
        self.con_type = wp.array(constraint_type, dtype=INT_DATA_TYPE)
        self.kp = wp.array(kp, dtype=FP_DATA_TYPE)
        self.kd = wp.array(kd, dtype=FP_DATA_TYPE)
        self.theta_target = wp.empty(shape=self.con_count, dtype=FP_DATA_TYPE)
        self.omega_target = wp.empty(shape=self.con_count, dtype=FP_DATA_TYPE)
        
        self.c = wp.empty(shape=(self.con_count, SIM_NUM), dtype=Vec3)
        self.lambdas = wp.empty(shape=(self.con_count, SIM_NUM), dtype=Vec3)
        self.J1 = wp.empty(shape=(self.con_count, SIM_NUM), dtype=Mat36)
        self.J_div_m1 = wp.empty(shape=(self.con_count, SIM_NUM), dtype=Mat36)
        self.w1 = wp.empty(shape=(self.con_count, SIM_NUM), dtype=Vec3)
        self.J2 = wp.empty(shape=(self.con_count, SIM_NUM), dtype=Mat36)
        self.J_div_m2 = wp.empty(shape=(self.con_count, SIM_NUM), dtype=Mat36)
        self.w2 = wp.empty(shape=(self.con_count, SIM_NUM), dtype=Vec3)
        self.d = wp.empty(shape=(self.con_count, SIM_NUM), dtype=Vec3)

    def update(self, step):
        theta_target = np.zeros(shape=self.con_count, dtype=FP_DATA_TYPE)
        omega_target = np.zeros(shape=self.con_count, dtype=FP_DATA_TYPE)
        ci = 0
        for i in range(len(self.joints)):
            if(self.joints[i].has_target):
                theta_target[ci+1] = self.joints[i].theta_target[step]
                omega_target[ci+1] = self.joints[i].omega_target[step]
            ci+=2
        self.theta_target = wp.array(theta_target, dtype=FP_DATA_TYPE)
        self.omega_target = wp.array(omega_target, dtype=FP_DATA_TYPE)

    def init(self, h):
        wp.launch(
            kernel=initConstraintsJoint,
            dim=(self.con_count, SIM_NUM),
            inputs=[
                self.con_count,
                self.con_type,
                self.body1_ind,
                self.body2_ind,
                self.bodies.phi,
                self.bodies.transform,
                self.bodies.I,
                self.xl1,
                self.xl2,
                self.axis1,
                self.axis2,
                self.b1,
                self.b2,
                self.theta_target,
                self.omega_target,
                self.kp,
                self.kd,
                h,
            ],
            outputs=[
                self.d,
                self.J1,
                self.J_div_m1,
                self.w1,
                self.J2,
                self.J_div_m2,
                self.w2
            ],
            record_tape=False,
        )
        # print("J1: ",self.J1.numpy())
        # print("J_div_m1: ",self.J_div_m1.numpy())
        # print("w1: ",self.w1.numpy())
        # print("J2: ",self.J2.numpy())
        # print("J_div_m2: ",self.J_div_m2.numpy())
        # print("w2: ",self.w2.numpy())
        # print("d: ",self.d.numpy())

    def computeC(self):
        # Compute contact constraints between ground and bodies
        pass

    def solve(self, h):
        wp.launch(
            kernel=solveConstraintsJoint,
            dim=SIM_NUM,
            inputs=[
                self.con_count,
                self.con_type,
                self.body1_ind,
                self.body2_ind,
                self.J1,
                self.J_div_m1,
                self.w1,
                self.J2,
                self.J_div_m2,
                self.w2,
                self.d,
                h
            ],
            outputs=[
                self.lambdas,
                self.bodies.phi,
                self.bodies.phi_dt
            ],
            record_tape=False
        )
        wp.synchronize()
        # print("phi: ",self.bodies.phi.numpy())
        # print("lambdas: ",self.lambdas.numpy())

    def solveVelocity(self, h):
        wp.launch(
            kernel=solveConstraintsJoint,
            dim=SIM_NUM,
            inputs=[
                self.con_count,
                self.con_type,
                self.body1_ind,
                self.body2_ind,
                self.J1,
                self.J_div_m1,
                self.w1,
                self.J2,
                self.J_div_m2,
                self.w2,
                self.d,
                h
            ],
            outputs=[
                self.lambdas,
                self.bodies.phi,
                self.bodies.phi_dt
            ],
            record_tape=False
        )
        wp.synchronize()
        # print("lambdas: ",self.lambdas.numpy())
        
class ConstraintMuscle(ConstraintsOnDevice):
    def __init__(self, muscles, bodies_on_device):
        self.con_count = len(muscles)
        self.bodies = bodies_on_device
        self.muscles = muscles

        body_num = np.zeros(shape=self.con_count, dtype=INT_DATA_TYPE)
        body_inds = np.zeros(shape=(self.con_count, VIA_POINT_MAX), dtype=INT_DATA_TYPE)
        constraint_type = np.empty(shape=self.con_count, dtype=INT_DATA_TYPE)
        xls = np.empty(shape=(self.con_count, VIA_POINT_MAX), dtype=Vec3)
        stiffness = np.zeros(shape=self.con_count, dtype=FP_DATA_TYPE)
        rest_length = np.zeros(shape=self.con_count, dtype=FP_DATA_TYPE)

        ci = 0
        for i in range(len(muscles)):
            body_num[ci] = len(muscles[i].bodies)
            for j in range(body_num[ci]):
                body_inds[ci][j] = muscles[i].bodies[j]
                xls[ci][j] = muscles[i].via_points[j]
            stiffness[ci] = self.muscles[i].stiffness
            rest_length[ci] = self.muscles[i].rest_length
            constraint_type[ci] = BODY_MUSCLE_CONSTRAINT
            ci += 1

        self.body_num = wp.array(body_num, dtype=INT_DATA_TYPE)
        self.body_inds = wp.array(body_inds, dtype=INT_DATA_TYPE)
        self.xls = wp.array(xls, dtype=Vec3)
        self.con_type = wp.array(constraint_type, dtype=INT_DATA_TYPE)
        self.stiffness = wp.array(stiffness, dtype=FP_DATA_TYPE)
        self.rest_length = wp.array(rest_length, dtype=FP_DATA_TYPE)

        self.alpha = wp.empty(shape=self.con_count, dtype=FP_DATA_TYPE)
        self.lambdas = wp.empty(shape=(self.con_count, SIM_NUM), dtype=FP_DATA_TYPE)
        self.Js = wp.empty(shape=(self.con_count, SIM_NUM, VIA_POINT_MAX), dtype=Vec6)
        self.J_div_ms = wp.empty(shape=(self.con_count, SIM_NUM, VIA_POINT_MAX), dtype=Vec6)
        self.w = wp.empty(shape=(self.con_count, SIM_NUM), dtype=FP_DATA_TYPE)
        self.d = wp.empty(shape=(self.con_count, SIM_NUM), dtype=FP_DATA_TYPE)
        self.b = wp.empty(shape=(self.con_count, SIM_NUM), dtype=FP_DATA_TYPE)

    def init(self, h):
        wp.launch(
            kernel=initConstraintsMuscle,
            dim=(self.con_count, SIM_NUM),
            inputs=[
                self.con_count,
                self.con_type,
                self.body_num,
                self.body_inds,
                self.stiffness,
                self.bodies.transform,
                self.bodies.I,
                self.xls,
                self.rest_length,
                h
            ],
            outputs=[
                self.alpha,
                self.b,
                self.Js,
                self.J_div_ms,
                self.w,
            ],
            record_tape=False,
        )
        # print("J1: ",self.Js.numpy())
        # print("J_div_ms: ",self.J_div_ms.numpy())
        # print("ws: ",self.ws.numpy())
        # print("J2: ",self.J2.numpy())
        # print("J_div_m2: ",self.J_div_m2.numpy())
        # print("w2: ",self.w2.numpy())
        # print("d: ",self.d.numpy())

    def computeC(self):
        # Compute contact constraints between ground and bodies
        pass

    def solve(self, h):
        wp.launch(
            kernel=solveConstraintsMuscle,
            dim=SIM_NUM,
            inputs=[
                self.con_count,
                self.con_type,
                self.body_num,
                self.body_inds,
                self.alpha,
                self.Js,
                self.J_div_ms,
                self.w,
                self.b,
                h
            ],
            outputs=[
                self.lambdas,
                self.bodies.phi,
            ],
            record_tape=False
        )
        wp.synchronize()
        # print("phi: ",self.bodies.phi.numpy())
        # print("lambdas: ",self.lambdas.numpy())

    def solveVelocity(self, h):
        self.solve(h)