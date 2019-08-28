# test loss function

from __future__ import print_function

from minisam import *
import numpy as np


e = np.array([0.3, 0.4, 0.5], dtype=np.float)
Js = [np.array([[1,2], [5,6], [7,8]], dtype=np.float), np.array([[3.0], [9], [11]], dtype=np.float)]


# ##################################
print('=======================================')
print('gaussian loss function')


gl1 = GaussianLoss.Covariance(np.array([[4,3,7], [8,3,5], [1,2,5]]))
print(gl1)

ew = gl1.weightError(e)
print('ew =', ew)

Jsw = gl1.weightJacobians(Js, e)
print('Jsw =', Jsw)


# in-place
print('====  in-place  ====')

ecopy = e.copy()
gl1.weightInPlace(ecopy)
print('ew =', ecopy)

ecopy = e.copy()
Jscopy = []
for J in Js:
    Jscopy.append(J.copy())

gl1.weightInPlace(Jscopy, ecopy)

print('Jsw =', Jscopy)
print('ew =', ecopy)

# ##################################
print('=======================================')
print('scale loss function')


sl1 = ScaleLoss.Scale(2.0)
print(sl1)

ew = sl1.weightError(e)
print('ew =', ew)

Jsw = sl1.weightJacobians(Js, e)
print('Jsw =', Jsw)


# in-place
print('====  in-place  ====')

ecopy = e.copy()
sl1.weightInPlace(ecopy)
print('ew =', ecopy)

ecopy = e.copy()
Jscopy = []
for J in Js:
    Jscopy.append(J.copy())

sl1.weightInPlace(Jscopy, ecopy)

print('Jsw =', Jscopy)
print('ew =', ecopy)


# ##################################
print('=======================================')
print('Huber loss function')


hl1 = HuberLoss.Huber(0.1)
print(hl1)

ew = hl1.weightError(e)
print('ew =', ew)

Jsw = hl1.weightJacobians(Js, e)
print('Jsw =', Jsw)


# in-place
print('====  in-place  ====')

ecopy = e.copy()
hl1.weightInPlace(ecopy)
print('ew =', ecopy)

ecopy = e.copy()
Jscopy = []
for J in Js:
    Jscopy.append(J.copy())

hl1.weightInPlace(Jscopy, ecopy)

print('Jsw =', Jscopy)
print('ew =', ecopy)


# ##################################
print('=======================================')
print('Composed loss function')


cl1 = ComposedLoss.Compose(sl1, sl1)
print(cl1)

ew = cl1.weightError(e)
print('ew =', ew)

Jsw = cl1.weightJacobians(Js, e)
print('Jsw =', Jsw)


# in-place
print('====  in-place  ====')

ecopy = e.copy()
cl1.weightInPlace(ecopy)
print('ew =', ecopy)

ecopy = e.copy()
Jscopy = []
for J in Js:
    Jscopy.append(J.copy())

cl1.weightInPlace(Jscopy, ecopy)

print('Jsw =', Jscopy)
print('ew =', ecopy)
