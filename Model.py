import warp as wp
import os
import numpy as np
from DataTypes import *
from Body import BodyRigid, BodyFixed
from Joint import JointHinge, JointHingeWorld
from BodiesOnDevice import BodiesOnDevice
from Constraints import ConstraintsContact, ConstraintsJoint,  ConstraintMuscle
from Muscle import MuscleSpring
from utils import SIM_NUM

class Model:
    def __init__(self):        
        self.bodies = []
        self.bodies_fixed = []
        self.joints = []
        self.muscles = []
        self.constraints = []
        self.collision_pairs = []
        self.bodies_on_device = None
        self.gravity = Vec3(0.0, 0.0, -980.0)
        self.f = None
        self.tau = None
        self.lambdaLen = 0
        self.dof_force = 0
        self.dof_torque = 0
        
        self.t = 0 # Current time step
        self.h = 1/60 # Time step
        self.steps = 60 # Total steps 
        self.pos_iters = 10 # Iteration number for PGS, substeps for TGS
        self.vel_sub_steps = 10 # Iteration number for velocities
        self.sub_steps = 50 
        
    def addBody(self, shape, density, mu=0.5):
        body = BodyRigid(shape, density, mu, len(self.bodies))
        self.bodies.append(body)
        return body.index
    
    def addFixedBody(self, shape, transform, variantion=False):
        if variantion:
            init_transforms = transform
        else:
            init_transforms = np.empty(shape=SIM_NUM, dtype=Transform)
            for i in range(SIM_NUM):
                init_transforms[i] = transform

        fixed = BodyFixed(shape, init_transforms, len(self.bodies_fixed))
        self.bodies_fixed.append(fixed)
        return fixed.index
    
    def addJoint(self, parent, child, xl, axis):
        if(parent is None):
            joint = JointHingeWorld(child, self.bodies[child].transform_host, xl, axis, len(self.joints))
            self.joints.append(joint)
        else:
            joint = JointHinge(parent, self.bodies[parent].transform_host, child, self.bodies[child].transform_host, xl, axis, len(self.joints))
            self.joints.append(joint)
        return joint.index

    def addMuscle(self, bodies, points, stiffness, rest_length):
        muscle = MuscleSpring(bodies, points, stiffness, rest_length, len(self.muscles))
        self.muscles.append(muscle)
        return muscle.index

    def setBodyInitialState(self, index, transform, phi, variantion=False):
        if index < 0 or index >= len(self.bodies):
            raise IndexError("Body index out of range")

        if variantion:
            init_transforms = transform
            init_phis = phi
        else:
            init_transforms = np.empty(shape=SIM_NUM, dtype=Transform)
            init_phis = np.empty(shape=SIM_NUM, dtype=Vec6)
            for i in range(SIM_NUM):
                init_transforms[i] = transform
                init_phis[i] = phi
        self.bodies[index].setInitialState(init_transforms, init_phis, transform, variantion)
        
    def setJointTargets(self, index, theta_target, omega_target, kp, kd):
        self.joints[index].setTarget(theta_target, omega_target, kp, kd)
    
    def init(self):
        #for robot in self.robots:
            #robot.init() 
        #self.initForceTorque()
        for body in self.bodies:
            body.init()
        self.bodies_on_device = BodiesOnDevice(self.bodies)
        self.constraints.append(ConstraintsJoint(self.joints, self.bodies_on_device))
        self.constraints.append(ConstraintMuscle(self.muscles, self.bodies_on_device))
        self.constraints.append(ConstraintsContact(self.bodies_fixed, self.bodies, self.bodies_on_device, self.collision_pairs))

    def step(self, num_steps: int):
        for step in range(num_steps):
            self.initConstraints()
            self.stepUncons()
            self.solveTGS()
            self.solveVelocity()
        self.t = self.t + num_steps
        
    def stepUncons(self):
        # for actuator in self.actuators:
        #     actuator.applyForceTorque(self.t) 
        self.bodies_on_device.stepUncons(self.h, self.gravity)
    
    def initConstraints(self):
        for constraint in self.constraints:
            constraint.update(self.t)
            constraint.init(self.h)
        # self.ground_contact_constraints.fillConstraints(self.fixed, self.bodies)
        # self.ground_contact_constraints.init()
        
    def solveTGS(self):
        for i in range(self.sub_steps):
            for constraint in self.constraints:
                constraint.solve(self.h / self.sub_steps)
            self.bodies_on_device.updateSubStepStates(self.h / self.sub_steps)
            wp.synchronize()
        print("phi: ",self.bodies_on_device.phi.numpy())
        self.bodies_on_device.intergrateStates(self.h)
        wp.synchronize()
        
            
    def solveVelocity(self):
        for constraint in self.constraints:
            constraint.d.zero_()
        for i in range(self.vel_sub_steps):
            for constraint in self.constraints:
                constraint.solveVelocity(self.h / self.vel_sub_steps)
        for constraint in self.constraints:
            constraint.lambdas.zero_() 
        print("phi: ",self.bodies_on_device.phi.numpy())
        for body in self.bodies:
            body.getResults(self.bodies_on_device.transform[body.index, :],
                            self.bodies_on_device.phi[body.index, :])
            
    
    def saveResults(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = f"Body_States.txt"
        
        frames = self.steps
        body_num = len(self.bodies)
        # Open the file for writing
        with open(output_dir + output_file, "w") as f:
            f.write(f"#Body number: {body_num}\n")
            f.write(f"#Body Shape: ")
            for body in self.bodies:
                f.write(body.shape.name + " " + " ".join(str(x) for x in body.shape.sides) + " ")
            f.write("\n")
            f.write(f"#Simulation number: {SIM_NUM}\n")
            f.write(f"#Step number: {frames}\n")
            # Run simulation for a few seconds to see the stacking
            for frame in range(frames):                
                # Write the states of each box
                frame_data = []
                for i in range(body_num):
                    for j in range(SIM_NUM):
                        frame_data.extend(self.bodies[i].stateHistory.transforms[frame][j])

                # Write the line to file
                f.write(" ".join(str(x) for x in frame_data) + "\n")
                
        