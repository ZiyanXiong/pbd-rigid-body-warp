import warp as wp
import numpy as np

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
            n = 10
            bodies = []
            for i in range(n):
                body = model.addBody(box_shape, 1.0)
                model.setBodyInitialState(body, Transform(Vec3(0.5 * i, 0.0, 2.0 + i * 4.0), Quat(0.0, 0.0, 0.0, 1.0)), Vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                bodies.append(body)

            for i in range(n):
                for j in range(i+1, n):
                    model.collision_pairs.append((bodies[i], bodies[j]))

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
            model.gravity = Vec3(0.0, 0.0, -980.0)
            model.steps = 300
            body1 = model.addBody(box_shape, 1.0)
            model.setBodyInitialState(body1, Transform(Vec3(6.0, 0.0, 15.0), Quat(0.0, 0.0, 0.0, 1.0)), Vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            body2 = model.addBody(box_shape, 1.0)
            model.setBodyInitialState(body2, Transform(Vec3(12.0, 0.0, 15.0), Quat(0.0, 0.0, 0.0, 1.0)), Vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            body3 = model.addBody(box_shape, 1.0)
            model.setBodyInitialState(body3, Transform(Vec3(18.0, 0.0, 15.0), Quat(0.0, 0.0, 0.0, 1.0)), Vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            joint1 = model.addJoint(None, body1, Vec3(3.0, 0.0, 15.0), Vec3(0.0, 1.0, 0.0), Vec2(wp.PI/4, -wp.PI/4))
            joint2 = model.addJoint(body1, body2, Vec3(3.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), Vec2(wp.PI/4, -wp.PI/4))
            joint3 = model.addJoint(body2, body3, Vec3(3.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), Vec2(wp.PI/4, -wp.PI/4))
            bodies = [body1, body2, body3]
            points = [Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, -0.5), Vec3(0.0, 0.0, -0.5), Vec3(2.0, 0.0, -0.5), Vec3(0.0, 0.0, 0.0)]
            model.addMuscle(bodies, points, 5000.0, 12.0)
            # model.setJointTargets(joint1, np.pi / 4.0 * np.ones(shape=model.steps, dtype=FP_DATA_TYPE), np.zeros(shape=model.steps, dtype=FP_DATA_TYPE), 1000.0, 10.0)
            # model.setJointTargets(joint2, np.pi / 4.0 * np.ones(shape=model.steps, dtype=FP_DATA_TYPE), np.zeros(shape=model.steps, dtype=FP_DATA_TYPE), 1000.0, 10.0)
            
            model.init()
            model.step(model.steps)
            model.saveResults("Results\\")
    
    elif(sceneId==2):
        with wp.ScopedDevice(DEVICE.CPU):
            # alloc and launch on "cpu"
            sides = Vec3(4.0, 4.0, 4.0)
            box_shape = Box(sides)

        with wp.ScopedDevice(DEVICE.GPU):
            model = Model()
            model.gravity = Vec3(0.0, 0.0, 0.0)
            body1 = model.addBody(box_shape, 1.0)
            model.setBodyInitialState(body1, Transform(Vec3(6.0, 0.0, 2.0), Quat(0.0, 0.0, 0.0, 1.0)), Vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            body2 = model.addBody(box_shape, 1.0)
            model.setBodyInitialState(body2, Transform(Vec3(-6.0, 0.0, 2.0), Quat(0.0, 0.0, 0.0, 1.0)), Vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            bodies = [body1, body2]
            points = [Vec3(0.0, 0.0, 0.0), Vec3(-2.0, 0.0, 0.0), Vec3(2.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0)]
            model.addMuscle(bodies, points, 5000.0, 9.0)
            model.init()
            model.step(model.steps)
            model.saveResults("Results\\")
        
runScene(1)
