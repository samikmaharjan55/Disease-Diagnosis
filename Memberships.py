import numpy as np
import matplotlib.pyplot as plt

def gaus(x,mu,sig):
    r=np.exp(-(x-mu)**2/2/sig**2)
    return(r)

def inc(x,a,b):
    if(x<a):
        return(0)
    elif(x>b):
        return(1)
    else:
        return((x-a)/(b-a))

def flat(x,a):
    if(x<a):
        return(0)
    else:
        return(1)

def revflat(x,a):
    if(x<a):
        return(1)
    else:
        return(0)

def dec(x,a,b):
    if(x<a):
        return(1)
    elif(x>b):
        return(0)
    else:
        return((b-x)/(b-a))


def tri(x,a,b,c):
    if(x<a):
        return(0)
    elif(x>c):
        return(0)
    elif(x>=a and x<=b):
        return((x-a)/(b-a))
    else:
        return((c-x)/(c-b))


def trap(x,a,b,c,d):
    if(x<a or x>d):
        return(0)
    elif(x>b and x<c):
        return(1)
    elif(x>=a and x<=b):
        return((x-a)/(b-a))
    else:
        return((d-x)/(d-c))

#+ve a is increasing   -ve a is decreasing
#Large magnetude of a is sharp curve     Small magnetude of a is blunt curve
# b centers the curve at membership = 0.5
def sigmoid(x,a,b):
    r=1/(1+np.exp(-a*(x-b)))
    return(r)

def square(x,arr):
    return 1

def z_func_dec(x,a,b):
    if x<=a:
        return 1
    if (x>=a and x<=(a+b)/2):
        return 1-2*((x-a)/(b-a))**2
    if (x>=(a+b)/2 and x<=b):
        return 2*((x-b)/(a-b))**2
    elif x>=b:
        return 0

def z_func_inc(x,a,b):
    if x<=a:
        return 0
    if (x>=a and x<=(a+b)/2):
        return 2*((x-a)/(b-a))**2
    if (x>=(a+b)/2 and x<=b):
        return 1-2*((x-b)/(b-a))**2
    elif x>=b:
        return 1

def square(x,low_lim,up_lim):
    if x>=low_lim and x<up_lim:
        return 1
    else:
        return 0