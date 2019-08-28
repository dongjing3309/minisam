# test sophus

from __future__ import print_function

from minisam import *
import math
import numpy as np


# ##################################
print('=======================================')
print('SO2 ctors')

s1 = sophus.SO2()
print('s =', s1)
print('s.theta =', s1.theta())
print('s.matrix =', s1.matrix())
print('s.complex =', s1.unit_complex())

print(type(s1.unit_complex()))

s2 = sophus.SO2(3.0 * math.pi)
print('s =', s2)
print('s.theta =', s2.theta())
print('s.matrix =', s2.matrix())
print('s.complex =', s2.unit_complex())

s2 = sophus.SO2(1.0)
print('s =', s2)
print('s.theta =', s2.theta())
print('s.matrix =', s2.matrix())
print('s.complex =', s2.unit_complex())


# ##################################
print('=======================================')
print('SO2 lie')

s = sophus.SO2.exp(0.321)

print('s =', s)
print('log(s) =', s.log())

omega = sophus.SO2.hat(0.321)

print('hat =', omega)
print('s.vee =', sophus.SO2.vee(omega))


# ##################################
print('=======================================')
print('SO2 group on self')

s3 = sophus.SO2(0.4)
s3i = s3.inverse()
s4 = s3 * s2
s5 = s2 * s3
s3 *= s2

print('s =', s3i)
print('s.theta =', s3i.theta())
print('s.matrix =', s3i.matrix())
print('s.complex =', s3i.unit_complex())

print('s =', s4)
print('s.theta =', s4.theta())
print('s.matrix =', s4.matrix())
print('s.complex =', s4.unit_complex())

print('s =', s5)
print('s.theta =', s5.theta())
print('s.matrix =', s5.matrix())
print('s.complex =', s5.unit_complex())

print('s =', s3)
print('s.theta =', s3.theta())
print('s.matrix =', s3.matrix())
print('s.complex =', s3.unit_complex())


# ##################################
print('=======================================')
print('SO2 group on point')

s = sophus.SO2(0.25 * math.pi)
p = np.array([1, 0], dtype=np.float)

sp = s * p
print('sp =', sp)

s = sophus.SO2(0.5 * math.pi)
p = np.array([4, 7], dtype=np.float)

sp = s * p
print('sp =', sp)


# ##################################
print('=======================================')
print('SE2')

p = np.array([1, 0], dtype=np.float)

se = sophus.SE2()
print(se)
print(se.so2())
print(se.translation())

sp = se * p
print('sp =', sp)


se = sophus.SE2.rot(0.25 * math.pi)
print(se)
print(se.so2())
print(se.translation())

sp = se * p
print('sp =', sp)


se = sophus.SE2.trans(np.array([3,4]))
print(se)
print(se.so2())
print(se.translation())

sp = se * p
print('sp =', sp)


# ##################################
print('=======================================')
print('SO3')

p = np.array([1, 0, 0], dtype=np.float)

so = sophus.SO3()
print(so)
print(so.matrix())
print(so.params())
print(so.unit_quaternion())

sp = so * p
print('sp =', sp)


so = sophus.SO3.rotX(0.4)
print(so)
print(so.matrix())
print(so.params())
print(so.unit_quaternion())

sp = so * p
print('sp =', sp)


so = sophus.SO3.rotY(0.5 * math.pi)
print(so)
print(so.matrix())
print(so.params())
print(so.unit_quaternion())

sp = so * p
print('sp =', sp)


so = sophus.SO3.rotY(0.5 * math.pi) * sophus.SO3.rotY(0.5 * math.pi)
print(so)
print(so.matrix())
print(so.params())
print(so.unit_quaternion())

sp = so * p
print('sp =', sp)


so = sophus.SO3(0.1, 0.2, -0.3, 0.927361849549570)
print(so)
print(so.matrix())
print(so.params())
print(so.unit_quaternion())


# ##################################
print('=======================================')
print('SE3')

p = np.array([1, 0, 0], dtype=np.float)

se = sophus.SE3()
print(se)
print(se.so3())
print(se.translation())

sp = se * p
print('sp =', sp)


se = sophus.SE3.rotY(0.25 * math.pi)
print(se)
print(se.so3())
print(se.translation())

sp = se * p
print('sp =', sp)


se = sophus.SE3.trans(np.array([3,4,5]))
print(se)
print(se.so3())
print(se.translation())

sp = se * p
print('sp =', sp)



# ##################################
print('=======================================')
print('miniSAM Variables')

var = Variables()

var.add(key('x', 1), s1)
var.add(key('x', 2), se)
var.add(key('x', 3), so)
var.add(key('x', 4), s4)

print(var)

print(var.at(key('x', 1)))
# print(var.at_SO2_(key('x', 1)))

print(var.at(key('x', 2)))
# print(var.at_SE3_(key('x', 2)))

print(var.at(key('x', 3)))
# print(var.at_SO3_(key('x', 3)))

print(var.at(key('x', 4)))
# print(var.at_SO2_(key('x', 4)))
