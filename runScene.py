import warp as wp

from DataTypes import *
from Model import Model
from Shape import Box, Plane
from utils import SIM_NUM, DEVICE

def runScene(sceneId):
    if(sceneId == 0):
        with wp.ScopedDevice(DEVICE.CPU):
            # alloc and launch on "cpu"
            sides = Vec3(4.0,4.0,4.0)
            box_shape = Box(sides)
            plane_shape = Plane(Vec3(0.0, 0.0, 1.0), Vec3(0.0, 0.0, 0.0))


        with wp.ScopedDevice(DEVICE.GPU):
            model = Model()
            model.addFixedBody(plane_shape, Transform(Vec3(0.0, 0.0, 0.0), Quat(0.0, 0.0, 0.0, 1.0)))

            body1 = model.addBody(box_shape, 1.0)
            model.setBodyInitialState(body1, Transform(Vec3(0.0, 0.0, 2.0), Quat(0.0, 0.0, 0.0, 1.0)), Vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            body2 = model.addBody(box_shape, 1.0)
            model.setBodyInitialState(body2, Transform(Vec3(0.0, 0.0, 6.0), Quat(0.0, 0.0, 0.0, 1.0)), Vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            model.collision_pairs.append((body1,body2))
            
            body1 = model.addBody(box_shape, 1.0)
            model.setBodyInitialState(body1, Transform(Vec3(0.0, 0.0, 2.0), Quat(0.0, 0.0, 0.0, 1.0)), Vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

            model.init()
            model.step(model.steps)
            model.saveResults("Results\\")
            
    elif(sceneId==1):
        with wp.ScopedDevice(DEVICE.CPU):
            # alloc and launch on "cpu"
            sides = Vec3(6.0,1.0,1.0)
            box_shape = Box(sides)


        with wp.ScopedDevice(DEVICE.GPU):
            model = Model()
                        
            body1 = model.addBody(box_shape, 1.0)
            model.setBodyInitialState(body1, Transform(Vec3(6.0, 0.0, 15.0), Quat(0.0, 0.0, 0.0, 1.0)), Vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            model.addJoint(None, body1, Vec3(3.0, 0.0, 15.0), Vec3(0.0, 1.0, 0.0))

            model.init()
            model.step(model.steps)
            model.saveResults("Results\\")
        
runScene(1)
