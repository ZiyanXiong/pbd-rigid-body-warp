import warp as wp
from DataTypes import *
from utils import SIM_NUM, EPS_SMALL, DEVICE
import numpy as np

@wp.func
def dLineClosestApproach(pa: Vec3, ua: Vec3, pb: Vec3, ub: Vec3):
    """
    Finds the closest points on two lines.
    Returns (alpha, beta) such that the closest points are
    qa = pa + alpha * ua and qb = pb + beta * ub.
    """
    p = pb - pa
    uaub = wp.dot(ua, ub)
    q1 = wp.dot(ua,p)
    q2 = wp.dot(ub,p)
    d = 1.0 - uaub * uaub
    if d <= EPS_SMALL:
        alpha = 0.0
        beta = 0.0
    else:
        inv_d = 1.0 / d
        alpha = (q1 + uaub * q2) * inv_d
        beta = (uaub * q1 + q2) * inv_d
    return alpha, beta

@wp.func
def intersectRectQuad(h:Vec2, p:Mat42):
    """
    Clips a 2D quadrilateral against a 2D rectangle, closely mimicking the C++ implementation.

    Args:
        h (Vec2): A list of 2 floats for the rectangle's half-widths [hx, hy].
        p (wp.array(dtype=Vec2)): A list of 4 Vec2 for the input quad's vertices.

    Returns:
        int: The number of vertices in the final clipped polygon.
        wp.array(dtype=Vec2): The vertices of the clipped polygon.
    """
    # q (and r) contain nq (and nr) coordinate points for the current (and chopped) polygons

    ret = Mat82()   
    buffer = Mat82()
    signs = Vec2(-1.0, 1.0)
    nq = 4
    for i in range(nq):
        buffer[i] = p[i]
    
    # We start by reading from 'p' and writing to the output buffer 'ret'.
    # q = p
    # r = ret 
    for dir_ in range(2):  
        sign = signs[0]
        pq_idx = INT_DATA_TYPE(0)
        pr_idx = INT_DATA_TYPE(0)
        nr = INT_DATA_TYPE(0)  # number of vertices in the newly-formed polygon 'r'
        # pq_idx = 0
        # pr_idx = 0
        # nr = 0
        for i in range(nq, 0, -1):
            # Go through all points in q and all lines between adjacent points
            current_point_val = buffer[pq_idx, dir_]

            # Check if the current point is inside the clipping line
            if sign * current_point_val < h[dir_]:
                # This point is inside, so copy it to the output list 'r'
                ret[pr_idx] = buffer[pq_idx]
                pr_idx += 1
                nr += 1
                # Check for buffer overflow
                if nr >= 8:
                    return nr, ret

            # Get the next point, with wrap-around for the last edge
            nextq_idx = pq_idx + 1 if i > 1 else 0
            next_point_val = buffer[nextq_idx, dir_]

            # Check if the line segment crosses the clipping line using XOR
            is_current_inside = (sign * current_point_val < h[dir_])
            is_next_inside = (sign * next_point_val < h[dir_])

            if is_current_inside != is_next_inside:
                # The line crosses the chopping line, so calculate the intersection point
                p_other_coord = buffer[pq_idx, 1 - dir_]
                n_other_coord = buffer[nextq_idx, 1 - dir_]
                
                denominator = next_point_val - current_point_val
                # The interpolated coordinate
                ret[pr_idx,  1 - dir_] = p_other_coord + (n_other_coord - p_other_coord) / \
                    denominator * (sign * h[dir_] - current_point_val)
                # The coordinate on the clipping line
                ret[pr_idx, dir_] = sign * h[dir_]

                pr_idx += 1
                nr += 1
                # Check for buffer overflow
                if nr >= 8:
                    return nr, ret

            # Move to the next point in 'q'
            pq_idx += 1

        # --- Buffer Swap ---
        # The clipped polygon 'r' becomes the input polygon 'q' for the next iteration.
        # q = ret
        # r = buffer
        nq = nr
            
        sign = signs[1]
        pq_idx = INT_DATA_TYPE(0)
        pr_idx = INT_DATA_TYPE(0)
        nr = INT_DATA_TYPE(0)  # number of vertices in the newly-formed polygon 'r'
        # pq_idx = 0
        # pr_idx = 0
        # nr = 0
        for i in range(nq, 0, -1):
            # Go through all points in q and all lines between adjacent points
            current_point_val = ret[pq_idx, dir_]

            # Check if the current point is inside the clipping line
            if sign * current_point_val < h[dir_]:
                # This point is inside, so copy it to the output list 'r'
                buffer[pr_idx] = ret[pq_idx]
                pr_idx += 1
                nr += 1
                # Check for buffer overflow
                if nr >= 8:
                    return nr, buffer

            # Get the next point, with wrap-around for the last edge
            nextq_idx = pq_idx + 1 if i > 1 else 0
            next_point_val = ret[nextq_idx, dir_]

            # Check if the line segment crosses the clipping line using XOR
            is_current_inside = (sign * current_point_val < h[dir_])
            is_next_inside = (sign * next_point_val < h[dir_])

            if is_current_inside != is_next_inside:
                # The line crosses the chopping line, so calculate the intersection point
                p_other_coord = ret[pq_idx, 1 - dir_]
                n_other_coord = ret[nextq_idx, 1 - dir_]
                
                denominator = next_point_val - current_point_val
                # The interpolated coordinate
                buffer[pr_idx,  1 - dir_] = p_other_coord + (n_other_coord - p_other_coord) / \
                    denominator * (sign * h[dir_] - current_point_val)
                # The coordinate on the clipping line
                buffer[pr_idx, dir_] = sign * h[dir_]

                pr_idx += 1
                nr += 1
                # Check for buffer overflow
                if nr >= 8:
                    return nr, buffer

            # Move to the next point in 'q'
            pq_idx += 1

        # --- Buffer Swap ---
        # The clipped polygon 'r' becomes the input polygon 'q' for the next iteration.
        # q = buffer
        # r = ret
        nq = nr
            
            
    return nr, buffer

@wp.func
def cullPoints(n:INT_DATA_TYPE, p:wp.array(dtype=Vec2), m:INT_DATA_TYPE, i0:INT_DATA_TYPE):
    """
    Reduces a set of 2D points to a smaller representative set.
    
    Args:
        n(int): number of points
        p(wp.array(dtype=Vec2)): list of 2D points 
        m(int): desired number of points
        i0(int): index of the point to always keep

    Returns:
        int: The number of vertices in the final clipped polygon.
        wp.array(dtype=int): The index of vertices of the clipped polygon.
    """
    if n <= m:
        return list(range(n))

    # Compute centroid
    if n == 1:
        cx, cy = p[0,0], p[0,1]
    elif n == 2:
        cx, cy = 0.5 * (p[0,0] + p [0,1]), 0.5 * (p[0,0] + p [0,1])
    else:
        a, cx, cy = 0, 0, 0
        for i in range(n):
            p_current = p[i]
            p_next = p[(i + 1) % n]  # Wrap around to the first point
            
            p_cross = wp.cross(p_current,p_next)
            a += p_cross
            cx += (p_current[0] + p_next[0]) * p_cross
            cy += (p_current[1] + p_next[1]) * p_cross
        
        if abs(a) < EPS_SMALL: # Fallback for degenerate polygons
            cx = sum(point[0] for point in p) / n
            cy = sum(point[1] for point in p) / n
        else:
            a = 1.0 / (3.0 * a)
            cx = a * cx
            cy = a * cy
            
    angles = [wp.atan2(point[1] - cy, point[0] - cx) for point in p]
    
    avail = wp.array(n,dtype=bool)
    avail.fill_(True)
    avail[i0] = False
    iret = wp.array(m,dtype=INT_DATA_TYPE)
    iret[0] = [i0]
    
    n_iret = 1
    for _ in range(1, m):
        target_angle = angles[i0] + (_ * (2 * wp.pi / m))
        if target_angle > wp.pi: target_angle -= 2 * wp.pi
        
        best_i =  -1
        max_diff = EPS_SMALL
        for i in range(n):
            if avail[i]:
                diff = abs(angles[i] - target_angle)
                if diff > wp.pi: diff = 2 * wp.pi - diff
                if diff < max_diff:
                    max_diff = diff
                    best_i = i
        if best_i != -1:
            avail[best_i] = False
            iret[n_iret] = best_i
            n_iret += 1
            
    return n_iret, iret

@wp.func
def dBoxBoxDistance(p1:Vec3, R1:Mat3, side1: Vec3, p2:Vec3, R2:Mat3, side2:Vec3):
    """
    Performs collision detection between two oriented boxes.
    p1, p2: Box center positions (list-like).
    R1, R2: 3x3 rotation matrices (list of lists, row-major).
    side1, side2: Box side lengths (list-like).
    max_contacts: Maximum number of contacts to generate.
    Returns: A list of ContactGeom objects.
    """
    fudge_factor = 0.05
    A = 0.5* side1 # Half side lengths for box 1
    B = 0.5* side2 # Half side lengths for box 2

    p = p2-p1
    pp = wp.transpose(R1) * p # Vector from p1 to p2 in R1's frame
    
    # Relative rotation matrix C = R1^T * R2
    R = wp.transpose(R1) * R2
    Q = Mat3(wp.abs(R[0,0]), wp.abs(R[0,1]), wp.abs(R[0,2]), 
            wp.abs(R[1,0]), wp.abs(R[1,1]), wp.abs(R[1,2]), 
            wp.abs(R[2,0]), wp.abs(R[2,1]), wp.abs(R[2,2]))


    s = -wp.inf
    normal = Vec3(0.0, 0.0, 0.0)
    invert_normal = False
    code = 0

    # Test axes parallel to box 1's axes
    for i in range(3):
        expr1 = pp[i]
        expr2 = A[i] + wp.dot(B, Q[i])
        s2 = abs(expr1) - expr2
        if s2 > s:
            s = s2
            normal = Vec3(R1[0, i], R1[1, i], R1[2, i])
            invert_normal = (expr1 < 0)
            code = i + 1

    # Test axes parallel to box 2's axes
    for i in range(3):
        expr1 = p[0] * R2[0, i] + p[1] * R2[1, i] + p[2] * R2[2, i]
        expr2 = A[0] * Q[0, i] + A[1] * Q[1, i] + A[2] * Q[2, i] + B[i]
        s2 = abs(expr1) - expr2
        if s2 > s:
            s = s2
            normal = Vec3(R2[0, i], R2[1, i], R2[2, i])
            invert_normal = (expr1 < 0)
            code = i + 4
    
    # Axes 7-15: Edge-Edge cross products
    # Axis 7: u1 x v1
    n = wp.vec3(0.0, -R[2,0], R[1,0])
    expr1 = pp[2]*R[1,0] - pp[1]*R[2,0]
    expr2 = A[1]*Q[2,0] + A[2]*Q[1,0] + B[1]*Q[0,2] + B[2]*Q[0,1]
    l = wp.norm_l2(n)
    if l > EPS_SMALL:
        s2 = (wp.abs(expr1) - expr2) / l
        if s2-fudge_factor*wp.abs(s2) > s: s, code, normal, invert_normal = s2, 7, R1*n/l, (expr1 < 0)

    # Axis 8: u1 x v2
    n = wp.vec3(0.0, -R[2,1], R[1,1])
    expr1 = pp[2]*R[1,1] - pp[1]*R[2,1]
    expr2 = A[1]*Q[2,1] + A[2]*Q[1,1] + B[0]*Q[0,2] + B[2]*Q[0,0]
    l = wp.norm_l2(n)
    if l > EPS_SMALL:
        s2 = (wp.abs(expr1) - expr2) / l
        if s2-fudge_factor*wp.abs(s2) > s: s, code, normal, invert_normal = s2, 8, R1*n/l, (expr1 < 0)

    # Axis 9: u1 x v3
    n = wp.vec3(0.0, -R[2,2], R[1,2])
    expr1 = pp[2]*R[1,2] - pp[1]*R[2,2]
    expr2 = A[1]*Q[2,2] + A[2]*Q[1,2] + B[0]*Q[0,1] + B[1]*Q[0,0]
    l = wp.norm_l2(n)
    if l > EPS_SMALL:
        s2 = (wp.abs(expr1) - expr2) / l
        if s2-fudge_factor*wp.abs(s2) > s: s, code, normal, invert_normal = s2, 9, R1*n/l, (expr1 < 0)

    # Axis 10: u2 x v1
    n = wp.vec3(R[2,0], 0.0, -R[0,0])
    expr1 = pp[0]*R[2,0] - pp[2]*R[0,0]
    expr2 = A[0]*Q[2,0] + A[2]*Q[0,0] + B[1]*Q[1,2] + B[2]*Q[1,1]
    l = wp.norm_l2(n)
    if l > EPS_SMALL:
        s2 = (wp.abs(expr1) - expr2) / l
        if s2-fudge_factor*wp.abs(s2) > s: s, code, normal, invert_normal = s2, 10, R1*n/l, (expr1 < 0)  
       
    # Axis 11: u2 x v2
    n = wp.vec3(R[2,1], 0.0, -R[0,1])
    expr1 = pp[0]*R[2,1] - pp[2]*R[0,1]
    expr2 = A[0]*Q[2,1] + A[2]*Q[0,1] + B[0]*Q[1,2] + B[2]*Q[1,0]
    l = wp.norm_l2(n)
    if l > EPS_SMALL:
        s2 = (wp.abs(expr1) - expr2) / l
        if s2-fudge_factor*wp.abs(s2) > s: s, code, normal, invert_normal = s2, 11, R1*n/l, (expr1 < 0)

    # Axis 12: u2 x v3
    n = wp.vec3(R[2,2], 0.0, -R[0,2])
    expr1 = pp[0]*R[2,2] - pp[2]*R[0,2]
    expr2 = A[0]*Q[2,2] + A[2]*Q[0,2] + B[0]*Q[1,1] + B[1]*Q[1,0]
    l = wp.norm_l2(n)
    if l > EPS_SMALL:
        s2 = (wp.abs(expr1) - expr2) / l
        if s2-fudge_factor*wp.abs(s2) > s: s, code, normal, invert_normal = s2, 12, R1*n/l, (expr1 < 0)
                
    # Axis 13: u3 x v1
    n = wp.vec3(-R[1,0], R[0,0], 0.0)
    expr1 = pp[1]*R[0,0] - pp[0]*R[1,0]
    expr2 = A[0]*Q[1,0] + A[1]*Q[0,0] + B[1]*Q[2,2] + B[2]*Q[2,1]
    l = wp.norm_l2(n)
    if l > EPS_SMALL:
        s2 = (wp.abs(expr1) - expr2) / l
        if s2-fudge_factor*wp.abs(s2) > s: s, code, normal, invert_normal = s2, 13, R1*n/l, (expr1 < 0)   
            
    # Axis 14: u3 x v2
    n = wp.vec3(-R[1,1], R[0,1], 0.0)
    expr1 = pp[1]*R[0,1] - pp[0]*R[1,1]
    expr2 = A[0]*Q[1,1] + A[1]*Q[0,1] + B[0]*Q[2,2] + B[2]*Q[2,0]
    l = wp.norm_l2(n)
    if l > EPS_SMALL:
        s2 = (wp.abs(expr1) - expr2) / l
        if s2-fudge_factor*wp.abs(s2) > s: s, code, normal, invert_normal = s2, 14, R1*n/l, (expr1 < 0)
        
    # Axis 15: u3 x v3
    n = wp.vec3(-R[1,2], R[0,2], 0.0)
    expr1 = pp[1]*R[0,2] - pp[0]*R[1,2]
    expr2 = A[0]*Q[1,2] + A[1]*Q[0,2] + B[0]*Q[2,1] + B[1]*Q[2,0]
    l = wp.norm_l2(n)
    if l > EPS_SMALL:
        s2 = (wp.abs(expr1) - expr2) / l
        if s2-fudge_factor*wp.abs(s2) > s: s, code, normal, invert_normal = s2, 15, R1*n/l, (expr1 < 0)
   
    if invert_normal:
        normal = -normal
    
    #wp.printf("normal: %f %f %f \n", normal[0], normal[1], normal[2])
    return s, code, normal

@wp.func
def generateContactPoints(p1:Vec3, R1:Mat3, side1: Vec3, p2:Vec3, R2:Mat3, side2:Vec3,  code:INT_DATA_TYPE, normal: Vec3, max_contacts: INT_DATA_TYPE,
                          row_ind: INT_DATA_TYPE, col_ind: INT_DATA_TYPE, contact_points: wp.array(dtype=Vec3, ndim=2), depth: wp.array(dtype=FP_DATA_TYPE, ndim=2)):
    A = 0.5* side1 # Half side lengths for box 1
    B = 0.5* side2 # Half side lengths for box 2

    if code > 6:
        # Edge-edge contact
        pa = p1; pb = p2
        for i in range(3):
            sign = 1.0 if (normal[i] * R1[0,i] + normal[i] * R1[0,i] + normal[i] * R1[0,i]) > 0.0 else -1.0
            pa += sign * A[i] * Vec3(R1[0, i], R1[1, i], R1[2, i])
        for i in range(3):
            sign = 1.0 if (normal[i] * R2[0,i] + normal[i] * R2[0,i] + normal[i] * R2[0,i]) > 0.0 else -1.0
            pa += sign * B[i] * Vec3(R2[0, i], R2[1, i], R2[2, i])

        uai = (code-7)//3
        ubi = (code-7)%3
        ua = Vec3(R1[0, uai], R1[1, uai], R1[2, uai])
        ub = Vec3(R1[0, ubi], R1[1, ubi], R1[2, ubi])
        
        alpha,beta = dLineClosestApproach(pa, ua, pb, ub)
        
        pa += ua * alpha
        pb += ub * beta
        
        contact_points[row_ind][col_ind]=(0.5 * (pa + pb))
        return 1

    # Face-something contact
    Ra, Rb, pa, pb, Sa, Sb = R1, R2, p1, p2, A, B
    is_swapped = 0
    if code > 3:
        Ra, Rb, pa, pb, Sa, Sb = R2, R1, p2, p1, B, A
        is_swapped = 1
    
    normal2 = normal if is_swapped == 0 else -normal
    
    nr = wp.transpose(Rb) * normal2
    anr = wp.abs(nr)
    
    lanr, a1, a2 = 0, 1, 2
    if anr[1] > anr[0]:
        if anr[1] > anr[2]: lanr, a1, a2 = 1, 0, 2
        else: lanr, a1, a2 = 2, 0, 1
    elif anr[0] > anr[2]: lanr, a1, a2 = 0, 1, 2
    else: lanr, a1, a2 = 2, 0, 1

    center = pb - pa
    center_offset_dir = 1.0 if nr[lanr] < 0.0 else -1.0
    center += center_offset_dir * Sb[lanr] * Vec3(Rb[0, lanr], Rb[1, lanr], Rb[2, lanr])
    
    codeN = code - 1 if code <= 3 else code - 4
    code1, code2 = (codeN+1)%3, (codeN+2)%3
    
    quad = Mat42()
    c1 = center[0] * Ra[0, code1] + center[1] * Ra[1, code1] + center[2] * Ra[2, code1]
    c2 = center[0] * Ra[0, code2] + center[1] * Ra[1, code2] + center[2] * Ra[2, code2]
    m11 = Ra[0, code1] * Rb[0,a1] + Ra[1, code1] * Rb[1,a1] + Ra[2, code1] * Rb[2,a1]
    m12 = Ra[0, code1] * Rb[0,a2] + Ra[1, code1] * Rb[1,a2] + Ra[2, code1] * Rb[2,a2]
    m21 = Ra[0, code2] * Rb[0,a1] + Ra[1, code2] * Rb[1,a1] + Ra[2, code2] * Rb[2,a1]
    m22 = Ra[0, code2] * Rb[0,a2] + Ra[1, code2] * Rb[1,a2] + Ra[2, code2] * Rb[2,a2]
    
    k1, k2 = m11*Sb[a1], m21*Sb[a1]
    k3, k4 = m12*Sb[a2], m22*Sb[a2]
    quad[0] = Vec2(c1-k1-k3, c2-k2-k4); quad[1] = Vec2(c1-k1+k3, c2-k2+k4)
    quad[2] = Vec2(c1+k1+k3, c2+k2+k4); quad[3] = Vec2(c1+k1-k3, c2+k2-k4)
    
    rect = Vec2(Sa[code1], Sa[code2])
    n_intersect,ret = intersectRectQuad(rect, quad)
    
    if n_intersect < 1: return 0

    cnum = INT_DATA_TYPE(0)

    det1 = 1.0 / (m11*m22 - m12*m21)
    m11_d, m12_d, m21_d, m22_d = m11*det1, m12*det1, m21*det1, m22*det1
    
    for j in range(n_intersect):
        k1 =  m22_d*(ret[j][0]-c1) - m12_d*(ret[j][1]-c2)
        k2 = -m21_d*(ret[j][0]-c1) + m11_d*(ret[j][1]-c2)
        
        pt = center + k1*Vec3(Rb[0,a1], Rb[1,a1], Rb[2,a1]) + k2*Vec3(Rb[0,a2], Rb[1,a2], Rb[2,a2])
        d = Sa[codeN] - wp.dot(normal2, pt)
        
        if d > -EPS_SMALL:
            if cnum < max_contacts:
                contact_points[row_ind+cnum][col_ind] = pt + pa
                depth[row_ind+cnum][col_ind] = d
                cnum += 1
  
    if cnum < 1:
        return 0
    else:
        return cnum
    
    # # Cull points if we have too many
    # i1 = 0
    # maxdepth = -wp.inf
    # for i in range(cnum):
    #     if dep[i] > maxdepth:
    #         maxdepth = dep[i]
    #         i1 = i

    # max_contacts, iret = cullPoints(cnum, ret, max_contacts, i1)
    
    # for j in range(max_contacts):
    #     idx = iret[j]
    #     contact_points.append(point[idx] + pa)
    # return max_contacts
    
@wp.kernel
def odeBoxBox(
    q1: wp.array(dtype= Transform),
    side1:wp.array(dtype=Vec3),
    q2: wp.array(dtype= Transform),
    side2:wp.array(dtype=Vec3),
    offset:FP_DATA_TYPE,
    # outputs
    contact_num: wp.array(dtype=INT_DATA_TYPE),
    depth: wp.array(dtype=FP_DATA_TYPE,  ndim=2),
    normal: wp.array(dtype=Vec3),
    contact_points: wp.array(dtype=Vec3,  ndim=2)
):
    i = wp.tid()
    displacement = 0.0
    p1 = q1[i].p
    R1 = wp.quat_to_matrix(q1[i].q)
    p2 = q2[i].p
    R2 = wp.quat_to_matrix(q2[i].q)
    s, code = dBoxBoxDistance(p1, R1, side1[i], p2, R2, side2[i], normal[i])
    
    if(s > offset):
        return
    elif s > EPS_SMALL:
        displacement = s 
    p1 = p1 + displacement * normal[i]
    contact_num[i] = generateContactPoints(p1, R1, side1[i], p2, R2, side2[i], code, normal[i], 8, contact_points[i], depth[i])
    for j in range(contact_num[i]):
        if(code <=3):
            contact_points[i][j] = contact_points[i][j] - (depth[i][j] + displacement) * normal[i]
        elif(code <=6):
            contact_points[i][j] = contact_points[i][j] - displacement * normal[i]
        else:
            contact_points[i][j] = contact_points[i][j] - (0.5 * depth[i][j] + displacement) * normal[i]
    
if __name__ == "__main__":
    def testOdeBoxBoxDistance():
        p1 = Vec3(0.0,0.0,0.5)
        R1 = wp.diag(Vec3(1.0,1.0,1.0))
        side1 = Vec3(1.0,1.0,1.0)
        p2 = Vec3(0.0,0.0,1.0)
        R2 = wp.diag(Vec3(1.0,1.0,1.0))
        side2 = Vec3(1.0,1.0,1.0)
        
        print("distance, code, normal: ", dBoxBoxDistance(p1, R1, side1, p2, R2, side2))
        
    def testOdeBoxBox():
        p1 = Vec3(0.0,0.0,1.0)
        R1 = wp.quat_to_matrix(wp.quat_from_axis_angle(Vec3(0.0,0.0,1.0), wp.pi/ 4))
        side1 = Vec3(1.0,1.0,1.0)
        p2 = Vec3(0.25,0.1,0.1)
        R2 = wp.diag(Vec3(1.0,1.0,1.0))
        side2 = Vec3(1.0,1.0,1.0)
        offset = 2e-1
        displacement = 0.0

        normal = Vec3(0.0,0.0,0.0)
        s, code = dBoxBoxDistance(p1, R1, side1, p2, R2, side2, normal)
        if(s > offset):
            print("No Collision")
            return
        elif s > EPS_SMALL:
            displacement = s 
            s = 0
        p1 = p1 + displacement * normal
        contact_points = wp.array(shape=8, dtype=Vec3)
        depth = wp.array(shape=8, dtype=FP_DATA_TYPE)
        contact_num = generateContactPoints(p1, R1, side1, p2, R2, side2, code, normal,8, contact_points, depth)
        
        contact_points = contact_points.numpy()
        depth = depth.numpy()

        for i in range(contact_num):
            if(code <=3):
                contact_points[i] = contact_points[i] - (s + displacement) * normal
            elif(code <=6):
                contact_points[i] = contact_points[i] - displacement * normal 
            else:
                contact_points[i] = contact_points[i] - (0.5 * s + displacement) * normal 
            print(f"contac point {i}: {contact_points[i]}, normal: {normal}")

    def testOdeBoxBoxKernel():
        body_q1_npa = np.empty(shape=SIM_NUM, dtype=Transform)
        body_q2_npa = np.empty(shape=SIM_NUM, dtype=Transform)
        side1_npa = np.empty(shape=SIM_NUM, dtype=Vec3)
        side2_npa = np.empty(shape=SIM_NUM, dtype=Vec3)
        offset = 2e-1
        for i in range(SIM_NUM):
            body_q1_npa[i] = Transform(Vec3(0.,0.,1.0),wp.quat_from_axis_angle(Vec3(0.0,0.0,1.0), wp.pi/ 4))
            body_q2_npa[i] = Transform(Vec3(0.25,0.1,0.1),wp.quat_from_axis_angle(Vec3(0.0,0.0,1.0), 0.))
            side1_npa[i] = Vec3(1.0,1.0,1.0)
            side2_npa[i] = Vec3(1.0,1.0,1.0)
        
        body_q1 = wp.array(body_q1_npa, dtype=Transform)
        body_q2 = wp.array(body_q2_npa, dtype=Transform)   
        side1 = wp.array(side1_npa, dtype=Vec3)
        side2 = wp.array(side2_npa, dtype=Vec3)
        
        contact_num= wp.empty(shape=SIM_NUM, dtype=INT_DATA_TYPE)
        depth= wp.empty(shape=(SIM_NUM, 8),dtype=FP_DATA_TYPE)
        normal= wp.empty(shape=SIM_NUM, dtype=Vec3)
        contact_points= wp.empty(shape=(SIM_NUM, 8),dtype=Vec3)
        
        wp.launch(
            kernel=odeBoxBox,
            dim=SIM_NUM,
            inputs=[
                body_q1,
                side1,
                body_q2,
                side2,
                offset
            ],
            outputs=[
                contact_num,
                depth,
                normal,
                contact_points
            ],
            device=DEVICE.GPU,
            record_tape=False,
        )
        
        print("Contact count: ",contact_num.numpy())
        #print("Contact depth: ", depth.numpy())
        #print("Contact normal: ", normal.numpy())
        #print("Contact points: ", contact_points.numpy())

        
        
    #testOdeBoxBoxDistance()
    #testOdeBoxBox()

    with wp.ScopedDevice(DEVICE.GPU):
        testOdeBoxBoxKernel()