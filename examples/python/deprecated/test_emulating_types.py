# example to test emulating types

from __future__ import print_function

from minisam import *
import numpy as np


# ##################################
print('=======================================')
print('variables: add')

var = Variables()

var[key('d', 1)] = 1.2
var[key('v', 2)] = np.array([1,2])
var[key('v', 3)] = np.array([1,2,3])
var[key('v', 4)] = np.array([1,2,4,5])
var[key('v', 5)] = np.array([1,2,4,5,10])

print(len(var))
print(var)


# ##################################
print('=======================================')
print('variables: update')

var[key('d', 1)] = 89.22
var[key('v', 2)] = np.array([7,8])
var[key('v', 3)] = np.array([7,8,9])

print(len(var))
print(var)


# ##################################
print('=======================================')
print('variables: del')

del var[key('d', 1)]

print(len(var))
print(var)


# ##################################
print('=======================================')
print('factor graph: len')

graph = FactorGraph()

graph.add(PriorFactor(key('x', 1), np.array([0, 1]), None))
graph.add(PriorFactor(key('x', 2), np.array([1, 2]), None))
graph.add(PriorFactor(key('x', 3), np.array([2, 3]), None))
graph.add(PriorFactor(key('x', 4), np.array([3, 0]), None))

print(len(graph))
print(graph)


# ##################################
print('=======================================')
print('factor graph: update')

graph[1] = PriorFactor(key('x', 22), np.array([2, 22]), None)
graph[2] = PriorFactor(key('x', 33), np.array([3, 33]), None)

print(len(graph))
print(graph)


# ##################################
print('=======================================')
print('factor graph: del')

del graph[1]

print(len(graph))
print(graph)


# ##################################
print('=======================================')
print('factor graph: loop')

for f in graph:
    print(f)
