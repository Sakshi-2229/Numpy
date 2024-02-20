# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 09:27:43 2023

@author: sai
"""

'''
What is Numpy ?

The numpy is open source python library
library used for scientific computing application, and it stands for 
Numerical python.

Consisting of multi dimensional array, objects and a collection of 
routines for processing those arrays.

'''
#Install python NumPy library
#goto base terminal and on prompt
#pip install numpy
#Install NumPy using conda
#conda install numpy


'''while a program list can contain different data types within a 
single list, all of the elements in a numpy array should be homogenous.
'''

#Arrays in NumPy
#create ndarray

import numpy as np
arr = np.array([10,20,30])
arr

#create a multidimensional array
arr=np.array([[10,20,30],[40,50,60]])
arr

#Represent the minimm dimensions
#use ndmin param to specify how many minimum dimensions you wanted to
#create an array with minimum dimension.
arr=np.array([10,20,30], ndmin=3)
arr

#change the datatype 
#dtype parameter
arr=np.array([10,20,30], dtype=complex)
arr

#get the dimensions of array
arr=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
arr.ndim
arr

#finding the size of each item in array
arr=np.array([10,20,30])
print("Each item contain in bytes:", arr.itemsize)
arr


#finding datatype of each item in array
arr=np.array([10,20,30])
print("Each item type in array:", arr.dtype)

#get the shape and size of an array
arr=np.array([[1,2,3,4],[5,6,7,8]])
print("Size of array:", arr.size)
print("Shape of array:", arr.shape)

#create a sequence of integers using arange()
#create a sequence of integers from 0 to 20 with steps of 3
arr=np.arange(0,20,3)
print("A sequence of integers with step of 3:\n", arr)


#access single element using index
arr=np.arange(11)
arr

arr[2]

arr[-2]

#Multi dimensional array indexing
#dimensionality reduction

#Accessing multi dimensional array element using array indexing

arr=np.array([[10,20,30,40,50],[20,30,50,10,30]])
arr

arr.shape

arr[1,1]
arr[0,0]
arr[1,2]

arr[1,-1]
#output=30

#Accessing array elements using slicing

arr=np.array([0,1,2,3,4,5,6,7,8,9])
x=arr[1:8:2]  #will start from 1 to 8 with step 2
x


x=arr[-2:3:-1] #start last but one(-2) upto 3 but not 3 in step of 1
x
# Output: array([8, 7, 6, 5, 4])

x=arr[-2:10] #start last but one(-2) and upto 10 but not 10
x
#Output: array([8, 9])

#indexing in numpy

multi_arr=np.array([[10,20,10,40],[40,50,70,90],[60,10,70,80],[30,90,40,30]])
multi_arr

#Slicing array

#for multi dimensional NumPy arrays,
#you can access the elements as below

multi_arr [1,2] #To access the value at row 1 and column 2
multi_arr [1,:] #To get the value at row 1 and all columns.
multi_arr [:,1] #Access the value at all rows and columns 1.


#columns from 0-3 and every alternate row
x=multi_arr[:3, ::2] #All rows three columns, in all selected rows and  
x

# output 
#([[10,20,10,40],[40,50,70,90],[60,10,70,80],[30,90,40,30]])
#array([[10, 10],
#       [40, 70],
#       [60, 70]])

#integer array indexing
arr=np.arange(35).reshape(5, 7) #it will create an array from 0-34, having 5 rows 
                                 #and 7 columns
arr


#Boolean array indexing
arr=np.arange(12).reshape(3,4)
arr
rows=np.array([False, True,True])
wanted_rows = arr[rows, :]
wanted_rows


#Convert NumPy array to Python List
array=np.array([10,20,30,40])
print("Array:",array)
print(type(array))
#Convert list
lst=array.tolist()
print("List:", lst)
print(type(lst))

#Convert Multi dimensional array to list
#create array

array=np.array([[10,20,30,40],
                [50,60,70,80]])
print("Array:",array)
print(type(array))

lst=array.tolist()
print("list:",lst)
print(type(lst))
#output will be list of list.  list[list]


#convert python list to numpy array

#Two types
#numpy.array()
#numpy.asarray()

#create list
list=[20,40,60,80]
print(type(list))
#convert array
array=np.array(list)
print("Array:", array)
print(type(array))


#numpy.asarray()
list=[20,40,60,80]
array=np.asarray(list)
print("Array:", array)
print(type(array))


#Numpy Array Properties
#ndarray.shape
#ndarray.ndim
#ndarray.size
#ndarray.type

#shape
array=np.array([[1,2,3],[4,5,6]])
print(array.shape)

#Resize the array
array=np.array([[10,20,30],[40,50,60]])
array.shape=(3,2)
array

#reshape usage
array=np.array([[10,20,30],[40,50,60]])
new_array=array.reshape(3,2)
new_array



'''
#numpy's operations are divided into 3 main categories

1. Fourier Transform and shape Manipulation # (used for voice NLP)
2. Mathematical and Logical Operations
3. Linear Algebra and Random Number Generation.

'''

#Arithmetic Operations

#Write a numpy program to get the Numpy version and show the NumPy

import numpy as np
print(np.__version__)

#Write a NumPy program to test whether none of the elements of 
# a given array are zero.

import numpy as np
x=np.array([1,2,3,4])
print("Original array:")
x
print("Test if none of the elements of the said array is zero:")
print(np.all(x))


x=np.array([0,1,2,3])
print("Original array:")
x
print("Test if none of the elements of the said array is zero:")
print(np.all(x))

#######################################################################

#write numpy program to test if any of the given array are non zero

import numpy as np
x=np.array([0,0,0,4])
print("Original array:")
x
print("Test if any of the elements of the said array is non zero:")
print(np.any(x))


import numpy as np
x=np.array([0,0,0,0])
print("Original array:")
x
print("Test if any of the elements of the said array is non zero:")
print(np.any(x))

###########################################################################################
#program to test a given array element wise for finiteness (not infinity or not a number)

import numpy as np
a=np.array([1,0, np.nan, np.inf])     #inf = infinite
print("Original array:")
x
print("Test a given array elemnt - wise for finiteness :")
print(np.isfinite(a))


#program to test element-wise for NaN of a given array.
import numpy as np
a=np.array([1,0, np.nan, np.inf])     #inf = infinite
print("Original array:")
x
print("Test a given array elemnt - wise for NaN :")
print(np.isnan(a))


'''
program to create an element-wise comparision (greater, greater_equal,
less and less_equal) of two given arrays.

'''

import numpy as np
x=np.array([3,5])
y=np.array([2,5])
print("Original arrays:")
x
y
print("Comparision - greater:")
np.greater(x,y)
print("Comparision - greater_equal:")
np.greater_equal(x,y)
print("Comparision - less:")
np.less(x,y)
print("Comparision - less_equal:")
np.less_equal(x,y)


##########################################################################

#program to create a 3*3 identity matrix
import numpy as np
array_2D=np.identity(3)
print('3x3 matrix:')
array_2D


#program to generate a random number between 0 and 1

import numpy as np
rand_num=np.random.normal(0,1,1) #as we have given step 1 so it will print only 
#one random number. if we give 2 it will print 2 random numbers.
print("random number between 0 and 1 :")
rand_num


import numpy as np
rand_num=np.random.normal(0,1,2) #as we have given step 2 it will print 2 random numbers.
print("random number between 0 and 1 :")
rand_num


###############################################################################################
#program to create 3x4 array and iterate over it.
import numpy as np
a=np.arange(10,22).reshape((3,4))
print("Original array:")
a
print("Each element of array is :")
for x in np.nditer(a):
    print(x,end=" ")
    print()


#write a program to create a vector of length 5 with values
#evenly distributed between 10 and 50.

import numpy as np
v = np.linspace(10,49,5)
print("Length 10 with values evenly distributed between 10 to 50")
v


#program to create a 3x3 matrix with values ranging from 2 to 10

x=np.arange(2,11).reshape(3,3)
x

#program to reverse an array (the first element becomes the last)
x=np.arange(12,38)
print("Original array:")
x
print("Reverse array:")
x=x[::-1]
x


#program to compute the multiplication of two matrices.
import numpy as np
p = [[1,0],[0,1]]
q= [[1,2],[3,4]]
print("Original matrix:")
p
q
result1=np.dot(p,q)            # gives a scalar result. 
print("Result of the said matrix multiplication:")
result1


#program to compute the cross product of two vectors.
import numpy as np
p = [[1,0],[0,1]]
q= [[1,2],[3,4]]
print("Original matrix:")
p
q
result1=np.cross(p,q)
result1=np.cross(q,p)       #gives a a new vector.
print("Result of the said matrix multiplication:")
result1


#program to compute the determinant of a given square array
import numpy as np
from numpy import linalg as LA
a=np.array([[1,0],[1,2]])
print("Original array:")
a
print("Determinant of the said 2-D array:")
print(np.linalg.det(a))

###########################################################################


#program to compute the inverse of a given matrix
m=np.array([[1,2],[3,4]])
print("Original matrix :")
m
result=np.linalg.inv(m)
print("Inverse of given matrix :")
result










