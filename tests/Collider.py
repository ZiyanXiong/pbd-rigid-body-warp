import warp as wp
import numpy as np
from DataTypes import *
from utils import SIM_NUM, EPS_SMALL, DEVICE

from warp.sim.collide import broadphase_collision_pairs, handle_contact_pairs
from warp.sim.model import ModelShapeGeometry

def test_collider():
    num_bodies = 2
    shape_pairs = 1
    num_shapes= 2
    
    body_q_npa = np.empty(shape=num_bodies, dtype=Transform)
    shape_body_npa = np.empty(shape=num_shapes, dtype=INT_DATA_TYPE)
    for i in range(num_bodies):
        body_q_npa[i] = Transform(0.,0.,0.5 + i*2,0.,0.,0.,1.)
        shape_body_npa[i] = i
    
    shape_contact_pairs_npa = np.empty(shape=(shape_pairs,2), dtype=INT_DATA_TYPE)
    for i in range(shape_pairs):
        shape_contact_pairs_npa[i,0] = i
        shape_contact_pairs_npa[i,1] = i+1
        
    
    body_q = wp.array(body_q_npa, dtype=Transform)
    body_mass = wp.empty(shape=num_bodies, dtype=FP_DATA_TYPE)
    shape_transform = wp.array(shape=num_shapes, dtype=Transform)
    shape_body= wp.array(shape_body_npa, dtype=INT_DATA_TYPE)
    shape_geo = ModelShapeGeometry()
    shape_geo.type= wp.empty(shape=num_shapes, dtype=INT_DATA_TYPE)
    shape_geo.scale= wp.empty(shape=num_shapes, dtype=Vec3)
    shape_geo.source= wp.empty(shape=num_shapes, dtype=ADD_DATA_TYPE)
    shape_geo.thickness = wp.empty(shape=num_shapes, dtype=FP_DATA_TYPE)
    shape_contact_pairs= wp.array(shape_contact_pairs_npa, dtype=INT_DATA_TYPE)
    shape_radius= wp.empty(shape=num_shapes, dtype=FP_DATA_TYPE)

    rigid_contact_max= 50
    rigid_contact_margin= 0.1
    iterate_mesh_vertices= True
    # outputs
    rigid_contact_count= wp.empty(shape=1, dtype=INT_DATA_TYPE)
    rigid_pair_shape0= wp.empty(shape=rigid_contact_max, dtype=INT_DATA_TYPE)
    rigid_pair_shape1= wp.empty(shape=rigid_contact_max, dtype=INT_DATA_TYPE)
    rigid_pair_point_id= wp.empty(shape=rigid_contact_max, dtype=INT_DATA_TYPE)
    rigid_pair_point_limit= wp.empty(shape=rigid_contact_max, dtype=INT_DATA_TYPE)
    
    
    body_mass.fill_(1.0)
    shape_transform.fill_(Transform(0.,0.,0.,0.,0.,0.,1.))
    shape_geo.type.fill_(wp.sim.GEO_BOX)
    shape_geo.scale.fill_(1.0)
    shape_geo.thickness.fill_(1)
    shape_radius.fill_(wp.sqrt(3.)/2.)
    rigid_pair_point_limit.fill_(100)
        
    wp.launch(
        kernel=broadphase_collision_pairs,
        dim=shape_pairs,
        inputs=[
            shape_contact_pairs,
            body_q,
            shape_transform,
            shape_body,
            body_mass,
            num_shapes,
            shape_geo,
            shape_radius,
            rigid_contact_max,
            rigid_contact_margin,
            rigid_contact_max,
            iterate_mesh_vertices,
        ],
        outputs=[
            rigid_contact_count,
            rigid_pair_shape0,
            rigid_pair_shape1,
            rigid_pair_point_id,
            rigid_pair_point_limit,
        ],
        device=DEVICE.GPU,
        record_tape=False,
    )
    print("Contact_count: ",rigid_contact_count.numpy())
    print("Contact shape0: ", rigid_pair_shape0.numpy())
    print("Contact shape1: ", rigid_pair_shape1.numpy())
    print("Contact point_id: ", rigid_pair_point_id.numpy())
    print("Contact point_limit: ", rigid_pair_point_limit.numpy())
    
    
    # outputs
    rigid_contact_count.zero_()
    rigid_contact_shape0= wp.empty(shape=rigid_contact_max, dtype=INT_DATA_TYPE)
    rigid_contact_shape1= wp.empty(shape=rigid_contact_max, dtype=INT_DATA_TYPE)
    rigid_contact_point0=wp.empty(shape=rigid_contact_max, dtype=Vec3)
    rigid_contact_point1=wp.empty(shape=rigid_contact_max, dtype=Vec3)
    rigid_contact_offset0=wp.empty(shape=rigid_contact_max, dtype=Vec3)
    rigid_contact_offset1=wp.empty(shape=rigid_contact_max, dtype=Vec3)
    rigid_contact_normal=wp.empty(shape=rigid_contact_max, dtype=Vec3)
    rigid_contact_thickness=wp.empty(shape=rigid_contact_max, dtype=FP_DATA_TYPE)
    rigid_contact_pairwise_counter=wp.empty(shape=rigid_contact_max, dtype=INT_DATA_TYPE)
    rigid_contact_tids=wp.empty(shape=rigid_contact_max, dtype=INT_DATA_TYPE)

    rigid_contact_shape0.fill_(-1)
    rigid_contact_shape1.fill_(-1)
    rigid_contact_pairwise_counter.zero_()
    
    wp.launch(
        kernel=handle_contact_pairs,
        dim=rigid_contact_max,
        inputs=[
            body_q,
            shape_transform,
            shape_body,
            shape_geo,
            rigid_contact_margin,
            rigid_pair_shape0,
            rigid_pair_shape1,
            num_shapes,
            rigid_pair_point_id,
            rigid_pair_point_limit,
            10,
        ],
        outputs=[
            rigid_contact_count,
            rigid_contact_shape0,
            rigid_contact_shape1,
            rigid_contact_point0,
            rigid_contact_point1,
            rigid_contact_offset0,
            rigid_contact_offset1,
            rigid_contact_normal,
            rigid_contact_thickness,
            rigid_contact_pairwise_counter,
            rigid_contact_tids,
        ],
        device=DEVICE.GPU,
    )
    
    print("Contact_count: ",rigid_contact_count.numpy())
    print("Contact shape0: ", rigid_contact_shape0.numpy())
    print("Contact shape1: ", rigid_contact_shape1.numpy())
    print("Contact point0: ", rigid_contact_point0.numpy())
    print("Contact point1: ", rigid_contact_point1.numpy())
    print("Contact offset0: ", rigid_contact_offset0.numpy())
    print("Contact offset1: ", rigid_contact_offset1.numpy())
    print("Contact normal: ", rigid_contact_normal.numpy())
    print("Contact thickness: ", rigid_contact_thickness.numpy())
    print("Contact pairwise_counter: ", rigid_contact_pairwise_counter.numpy())
    print("Contact contact_tids: ", rigid_contact_tids.numpy())
    
    
with wp.ScopedDevice(DEVICE.GPU):
    test_collider()