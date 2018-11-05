import turtle as t
# import timer
# import numpy as np
import sys

# sys.setrecursionlimit(3000)



def tree(hm_branch,dist):
    if hm_branch>0:
        t.fd(dist)
        t.rt(45)
        tree(hm_branch-1,dist/1.5)
        t.lt(90)
        tree(hm_branch-1, dist/1.5)
        t.rt(45)
        t.fd(-dist)


    else:
        return
t.speed(0)
t.penup()
t.goto(0,-180)
t.left(90)
t.pendown()
tree(3,150)

t.done()
# np.random.seed(10)
# tabela = np.random.randint(1,400000,2900)

# def listsum(numList):
#     if len(numList) == 1:
#         return numList[0]
#     else:
#         return numList[0] + listsum(numList[1:])


# def suma(tabela):
#     zm1 = 0
#     for x in tabela:
#         zm1 += x
#     return (zm1)


# print(listsum(tabela))
# print(tabela)

