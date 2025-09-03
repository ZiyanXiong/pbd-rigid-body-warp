SIM_NUM =  1
CONTACT_MAX =  150
VIA_POINT_MAX =  3
EPS_BIG = 1E-3
EPS_SMALL= 1E-6

class DEVICE:
    GPU =  "cuda"
    CPU =  "cpu"
    
import warp as wp
from DataTypes import *

@wp.func
def getContactFrame(n:Vec3):
    """
    Computes the contact frame based on the normal vector. 
    The contact frame is defined as a coordinate system aligned with the contact normal.
    Each row of the returned matrix corresponds to a basis vector of the contact frame.
    The first row is the normal vector, the second is orthogonal to the normal in the plane of contact.
    """
    cf = Mat3()
    if wp.abs(n[2]) < EPS_SMALL:
        cf[0] = Vec3(0.0, 0.0, 1.0)
    else:
        cf[0] = Vec3(1.0, 0.0, 0.0)
        
    cf[2] = wp.normalize(wp.cross(n, cf[0]))
    cf[1] = wp.cross(cf[2], n)
    cf[0] = n
    return cf

@wp.func
def gamma(xi: Vec3):
    G = Mat36(0.0, xi[2], -xi[1], 1.0, 0.0, 0.0,
              -xi[2], 0.0, xi[0], 0.0, 1.0, 0.0,
              xi[1], -xi[0], 0.0, 0.0, 0.0, 1.0)
    return G

