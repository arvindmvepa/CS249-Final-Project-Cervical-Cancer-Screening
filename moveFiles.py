from __future__ import division, print_function, absolute_import

import numpy as np
from numpy import genfromtxt
import os

testLoc = '/home/ubuntu/cs249_final_project/image_files/test/'
trainLoc = '/home/ubuntu/cs249_final_project/image_files/train/'

type1 = trainLoc + 'Type_1/'
type2 = trainLoc + 'Type_2/'
type3 = trainLoc + 'Type_3/'

rows = genfromtxt('solution_stg1_release.csv', dtype=None, delimiter=",", skip_header=1)

type1Count = 0
type2Count = 0
type3Count = 0
counter = 0
for row in rows:
    fileName = str(counter)+ ".jpg"
    newFileName = str(counter) + "_.jpg"
    print(testLoc+fileName)
    if int(row[1]) == 1:
        os.rename(testLoc+fileName, type1+newFileName)
        type1Count += 1
        counter += 1
    if int(row[2]) == 1:
        os.rename(testLoc+fileName, type2+newFileName)
        type2Count += 1
        counter += 1
    if int(row[3]) == 1:
        os.rename(testLoc+fileName, type3+newFileName)
        type3Count += 1
        counter += 1
print(type1Count)
print(type2Count)
print(type3Count)
print(counter)
