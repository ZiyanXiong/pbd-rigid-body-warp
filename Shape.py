import warp as wp
from DataTypes import *

class Box():
    def __init__(self, sides):          
        self.sides = sides
        
    def computeInertia(self, density):
            I = Vec6()
            mass = density * self.sides.x * self.sides.y * self.sides.z
            I[0] = (1/12.0) * mass * (self.sides.y*self.sides.y + self.sides.z*self.sides.z)
            I[1] = (1/12.0) * mass * (self.sides.x*self.sides.x + self.sides.z*self.sides.z)
            I[2] = (1/12.0) * mass * (self.sides.x*self.sides.x + self.sides.y*self.sides.y)
            I[3:6] = Vec3(mass)
            return I
        
class Plane():
    def __init__(self, normal, translation):
        self.normal = normal
        self.translation = translation

    def computeInertia(self, density):
        # For a plane, inertia is not defined in the same way as for a rigid body.
        # This is a placeholder for any specific inertia calculations if needed.
        return Vec6(0.0)  # No inertia for a plane
