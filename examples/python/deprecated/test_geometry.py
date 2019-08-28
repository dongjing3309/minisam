# test mvg

from __future__ import print_function

from minisam import *
import math
import numpy as np


# ##################################
print('=======================================')
print('Calib')

c = CalibBundler(100, 0.0, 0.01)

print(c)
print(CalibBundler.dim())
print(c.f())
print(c.k1())
print(c.k2())
print(c.matrix())
print(c.inverse_matrix())

pw = np.array([0.23, 0.34])

pi = c.project(pw)
print(pi)

pwn = c.unproject(pi)
print(pwn)


# ##################################
print('=======================================')
print('projectBundler')

pw = np.array([0.23, 0.34, -2.0])

pose = sophus.SE3()

pi = projectBundler(pose, c, pw)
print(pi)

J_pose, J_calib, J_pw = projectBundlerJacobians(pose, c, pw)

print(J_pose)
print(J_calib)
print(J_pw)


# ##################################
print('=======================================')
print('project : CalibK')

c = CalibK(100, 100, 30, 20)

pw = np.array([0.23, 0.34, 1.0])

pose = sophus.SE3()

pi = project(pose, c, pw)
# pi = project_CalibK_(pose, c, pw)
print(pi)

J_pose, J_calib, J_pw = projectJacobians(pose, c, pw)
# J_pose, J_calib, J_pw = projectJacobians_CalibK_(pose, c, pw)     # OK

print(J_pose)
print(J_calib)
print(J_pw)


# ##################################
print('=======================================')
print('project factor : CalibK')


pf = ReprojectionPoseFactor(key('x', 0), key('l', 0), c, pi)
# pf = ReprojectionPoseFactor_CalibK_(key('x', 0), key('l', 0), c, pi, None)

vars = Variables()
vars.add(key('x', 0), pose)
vars.add(key('l', 0), pw)

print(pf)

print(pf.keys())
print(pf.dim())
print(pf.lossFunction())
print(pf.error(vars))
print(pf.jacobians(vars))
print(pf.weightedError(vars))
print(pf.weightedJacobiansError(vars))


pf = ReprojectionPoseFactor(key('x', 0), key('l', 0), c, pi, ScaleLoss.Scale(2.0))
# pf = ReprojectionPoseFactor_CalibK_(key('x', 0), key('l', 0), c, pi, ScaleLoss.Scale(2.0))

vars = Variables()
vars.add(key('x', 0), pose)
vars.add(key('l', 0), np.array([0.46, 0.68, 1.0]))

print(pf)

print(pf.keys())
print(pf.dim())
print(pf.lossFunction())
print(pf.error(vars))
print(pf.jacobians(vars))
print(pf.weightedError(vars))
print(pf.weightedJacobiansError(vars))
