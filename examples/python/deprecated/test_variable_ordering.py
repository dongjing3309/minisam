# example to test variable ordering

from __future__ import print_function

from minisam import *


# ##################################
print('=======================================')
print('empty list')

vo1 = VariableOrdering()

print(vo1)
print('len =', len(vo1))
print('size =', vo1.size())


# ##################################
print('=======================================')
print('non-empty list')

volist = [key('a', 1), key('b', 2), key('c', 3), key('d', 4)]

vo2 = VariableOrdering(volist)

print(vo2)
print('len =', len(vo2))
print('size =', vo2.size())


# ##################################
print('=======================================')
print('append list')

vo2.push_back(key('G', 7))
vo2.push_back(key('F', 6))
vo2.push_back(key('E', 5))

print(vo2)
print('len =', len(vo2))
print('size =', vo2.size())


# ##################################
print('=======================================')
print('index operator')

print('vo2[', 0, '] =', keyString(vo2[0]))
print('vo2[', 3, '] =', keyString(vo2[3]))
print('vo2[', 6, '] =', keyString(vo2[6]))


# ##################################
print('=======================================')
print('search')

print('vo2.search', keyString(vo2[0]), ' =', vo2.searchKey(vo2[0]))
print('vo2.search', keyString(vo2[3]), ' =', vo2.searchKey(vo2[3]))
print('vo2.search', keyString(vo2[6]), ' =', vo2.searchKey(vo2[6]))

