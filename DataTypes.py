import warp.types as wpt

FP_DATA_TYPE =  wpt.float32
INT_DATA_TYPE = wpt.int32
ADD_DATA_TYPE = wpt.uint64

class Vec2(wpt.vector(length=2, dtype=FP_DATA_TYPE)):
    pass
class Vec3(wpt.vector(length=3, dtype=FP_DATA_TYPE)):
    pass
class Vec3i(wpt.vector(length=3, dtype=INT_DATA_TYPE)):
    pass
class Vec4(wpt.vector(length=4, dtype=FP_DATA_TYPE)):
    pass
class Vec6(wpt.vector(length=6, dtype=FP_DATA_TYPE)):
    pass
class Mat2(wpt.matrix(shape=(2,2), dtype=FP_DATA_TYPE)):
    pass
class Mat3(wpt.matrix(shape=(3,3), dtype=FP_DATA_TYPE)):
    pass
class Mat4(wpt.matrix(shape=(4,4), dtype=FP_DATA_TYPE)):
    pass
class Mat36(wpt.matrix(shape=(3,6), dtype=FP_DATA_TYPE)):
    pass
class Mat42(wpt.matrix(shape=(4,2), dtype=FP_DATA_TYPE)):
    pass
class Mat82(wpt.matrix(shape=(8,2), dtype=FP_DATA_TYPE)):
    pass
class Quat(wpt.quaternion(dtype=FP_DATA_TYPE)):
    pass
class Transform(wpt.transformation(dtype=FP_DATA_TYPE)):
    pass