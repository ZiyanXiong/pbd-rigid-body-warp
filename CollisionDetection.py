import warp as wp
from DataTypes import *
from utils import SIM_NUM, CONTACT_MAX, DEVICE, EPS_BIG
from odeBoxBox import dBoxBoxDistance, generateContactPoints
from Constraints import WORLD_CONTACT_CONSTRAINT, BODY_CONTACT_CONSTRAINT

@wp.kernel
def planeBoxCollision(
    body1: INT_DATA_TYPE,
    q1: wp.array(dtype= Transform),
    side1: Vec3,
    body1_mu: FP_DATA_TYPE,
    q2: wp.array(dtype= Transform),
    normal_plane: Vec3,
    offset:FP_DATA_TYPE,
    # outputs
    contact_num: wp.array(dtype=INT_DATA_TYPE),
    contact_type: wp.array(dtype=INT_DATA_TYPE, ndim=2),
    body_ind: wp.array(dtype=INT_DATA_TYPE, ndim=2),
    depth: wp.array(dtype=FP_DATA_TYPE, ndim=2),
    normal: wp.array(dtype=Vec3, ndim=2),
    contact_points: wp.array(dtype=Vec3, ndim=2),
    contact_mu: wp.array(dtype=FP_DATA_TYPE, ndim=2)
):
    tid = wp.tid()
    t = wp.transform_multiply(wp.transform_inverse(q2[tid]), q1[tid])
    #wp.printf("t: %f %f %f %f %f %f %f\n", t[0], t[1], t[2], t[3], t[4], t[5], t[6])
    sign = Vec2(-1.0, 1.0)
    for sign_x in range(2):
        for sign_y in range(2):
            for sign_z in range(2):
                # Compute the corner of the box
                corner = Vec3(
                    sign[sign_x] * side1[0] / 2.0,
                    sign[sign_y] * side1[1] / 2.0,
                    sign[sign_z] * side1[2] / 2.0
                )
                
                # Transform the corner to plane space
                corner_plane = wp.transform_point(t, corner)
                
                # Compute the distance from the plane
                distance = wp.dot(normal_plane, corner_plane)
                if distance < offset:
                    # If the corner is below the plane, we have a contact
                    con_ind = wp.atomic_add(contact_num, tid, 1)
                    if con_ind < CONTACT_MAX:
                        body_ind[con_ind][tid] = body1
                        depth[con_ind][tid] = distance
                        normal[con_ind][tid] = -wp.transform_vector(q2[tid], normal_plane)
                        contact_points[con_ind][tid] = corner
                        contact_mu[con_ind][tid] = body1_mu
                        contact_type[con_ind][tid] = WORLD_CONTACT_CONSTRAINT
                    else:
                        print("Number of rigid contacts exceeded limit. Increase utils.CONTACT_MAX.")
                    
@wp.kernel
def boxBoxCollision(
    body1: INT_DATA_TYPE,
    q1: wp.array(dtype= Transform),
    side1:Vec3,
    body2: INT_DATA_TYPE,
    q2: wp.array(dtype= Transform),
    side2:Vec3,
    offset:FP_DATA_TYPE,
    contact_mu_const: FP_DATA_TYPE,
    # outputs
    contact_num: wp.array(dtype=INT_DATA_TYPE),
    contact_type: wp.array(dtype=INT_DATA_TYPE, ndim=2),
    body_id1: wp.array(dtype=INT_DATA_TYPE, ndim=2),
    body_id2: wp.array(dtype=INT_DATA_TYPE, ndim=2),
    depth: wp.array(dtype=FP_DATA_TYPE,  ndim=2),
    normal: wp.array(dtype=Vec3, ndim=2),
    contact_points1: wp.array(dtype=Vec3,  ndim=2),
    contact_points2: wp.array(dtype=Vec3,  ndim=2),
    contact_mu: wp.array(dtype=FP_DATA_TYPE, ndim=2)
):
    i = wp.tid()
    displacement = 0.0
    p1 = q1[i].p
    R1 = wp.quat_to_matrix(q1[i].q)
    p2 = q2[i].p
    R2 = wp.quat_to_matrix(q2[i].q)

    s, code, normal_kernel = dBoxBoxDistance(p1, R1, side1, p2, R2, side2)
    # wp.printf("code: %d \n",code)
    # wp.printf("distance: %f\n", s)
    # wp.printf("normal: %f %f %f \n", normal_kernel[0], normal_kernel[1], normal_kernel[2])

    if(s > offset):
        return
    elif s > -offset:
        displacement = offset 
    p1 = p1 + displacement * normal_kernel

    contact_num_kernel = generateContactPoints(p1, R1, side1, p2, R2, side2, code, normal_kernel, 8, contact_num[i], i, contact_points1, depth)
    if contact_num_kernel + contact_num[i] > CONTACT_MAX:
        print("Number of rigid contacts exceeded limit. Increase utils.CONTACT_MAX.")
        return
    else:
        for j in range(contact_num_kernel):
            con_ind = contact_num[i]
            body_id1[con_ind][i] = body2
            body_id2[con_ind][i] = body1
            normal[con_ind][i] = normal_kernel
            # wp.printf("code: %d \n",code)
            # wp.printf("distance: %f\n", depth[con_ind][i])
            # wp.printf("displacement: %f\n", displacement)
            # wp.printf("normal: %f %f %f \n", normal_kernel[0], normal_kernel[1], normal_kernel[2])
            # xl1 = contact_points1[con_ind][i]
            # wp.printf("xl1: %f %f %f \n", xl1[0], xl1[1], xl1[2])
            
            if(code <=3):
                contact_points1[con_ind][i] = contact_points1[con_ind][i]  + (depth[con_ind][i] - displacement) * normal_kernel
            elif(code <=6):
                contact_points1[con_ind][i] = contact_points1[con_ind][i] - displacement * normal_kernel
            else:
                contact_points1[con_ind][i] = contact_points1[con_ind][i] + (0.5 * depth[con_ind][i] - displacement) * normal_kernel
            depth[con_ind][i] = -(depth[con_ind][i] - displacement)
            contact_points2[con_ind][i] = wp.transform_point(wp.transform_inverse(q1[i]), contact_points1[con_ind][i])
            contact_points1[con_ind][i] = wp.transform_point(wp.transform_inverse(q2[i]), contact_points1[con_ind][i] + depth[con_ind][i] * normal_kernel)
            contact_mu[con_ind][i] = contact_mu_const
            contact_type[con_ind][i] = BODY_CONTACT_CONSTRAINT
            contact_num[i] += 1


def collisionDetectionGroundCuboid(body1, q1, side1, body1_mu, q2, normal_plane, contact_num, contact_type, body_ind, depth, normal, contact_points, contact_mu):

    offset = 2e-1
    wp.launch(
        kernel=planeBoxCollision,
        dim=SIM_NUM,
        inputs=[
            body1,
            q1,
            side1,
            body1_mu,
            q2,
            normal_plane,
            offset,
        ],
        outputs=[
            contact_num,
            contact_type,
            body_ind,
            depth,
            normal,
            contact_points,
            contact_mu
        ],
        record_tape=False,
    )
    
def collisionDetectionCuboidCuboid(body1, q1, side1, body2, q2, side2, contact_mu_const, contact_num, contact_type, body_id1, body_id2, depth, normal, contact_points1, contact_points2, contact_mu):

    offset = 2e-1
    wp.launch(
        kernel=boxBoxCollision,
        dim=SIM_NUM,
        inputs=[
            body1,
            q1,
            side1,
            body2,
            q2,
            side2,
            offset,
            contact_mu_const,
        ],
        outputs=[
            contact_num,
            contact_type,
            body_id1,
            body_id2,
            depth,
            normal,
            contact_points1,
            contact_points2,
            contact_mu
        ],
        record_tape=False,
    )