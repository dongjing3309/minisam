# example to test variables

from __future__ import print_function

from minisam import *
import numpy as np


# ##################################
print('=======================================')
print('empty variables')

var = Variables()

print(var)
print('size =', var.size())
print(var.exists(key('v', 1)))


# ##################################
print('=======================================')
print('add variables')

var.add(key('d', 1), 1.2)
var.add(key('v', 2), np.array([1,2]))
var.add(key('v', 3), np.array([1,2,3]))
var.add(key('v', 4), np.array([1,2,4,5]))
var.add(key('v', 5), np.array([1,2,4,5,10]))

print(var)
print('size =', var.size())


# ##################################
print('=======================================')
print('query (at const) variables')

v2 = var.at(key('v', 2))              # OK
# v2 = var.at_Vector2_(key('v', 2))     # OK
# v2 = var.at_Vector_(key('v', 2))      # runtime error

print(v2)

v5 = var.at(key('v', 5))              # OK
# v5 = var.at_Vector_(key('v', 5))      # OK

print(v5)

# check copy
v2 = np.array([3,4])
print(v2)

print(var)

# ##################################
print('=======================================')
print('update (at non-const) variables')

var.update(key('d', 1), 5.6)
var.update(key('v', 2), np.array([5,9]))

print(var)


# ##################################
print('=======================================')
print('manifold')

print('dim =', var.dim())

delta = 0.01 * np.ones(15, dtype=np.float)

vo = VariableOrdering([key('d', 1), key('v', 2), key('v', 3), key('v', 4), key('v', 5)])

var2 = var.retract(delta, vo)

print(var2)

delta_inv = var2.local(var, vo)

print('delta_inv =', delta_inv)

# ##################################
print('=======================================')
print('erase variables')

var.erase(key('d', 1))
var.erase(key('v', 2))

print(var)
print('size =', var.size())
