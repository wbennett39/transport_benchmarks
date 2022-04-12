#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 11:20:05 2022

@author: bennett
"""
import numpy as np
import math
import numba 
from scipy.special import expi
from numba import njit
from numba import cfunc,carray
from numba.types import intc, CPointer, float64
from scipy import LowLevelCallable
import h5py
from pathlib import Path


###############################################################################
def xi(u, eta):
    q = (1+eta)/(1-eta)
    zz = np.tan(u/2)
    
    # return (np.log(q) + complex(0, u))/(eta + complex(0, zz))
    return (np.log(q) + u*1j)/(eta + zz*1j)

# @cfunc("float64(float64, float64, float64, float64, float64)")
# @jit
def pyF1(u, s, tau, x, t):
    xp = x-s
    tp = t-tau
    if abs(xp) <= tp:
        eta = xp/tp
        eval_xi = xi(u, eta)
        complex_term = np.exp(tp*((1 - eta**2)*eval_xi/2.))*eval_xi**2
        return (1/np.cos(u/2.0))**2*complex_term.real * (4/math.pi) * (1 - eta**2) * math.exp(-tp)/2 # this seems wrong
    else:
        return 0.0

def get_intervals(x, x0, t, t0):
    intervals = np.zeros(4)
    intervals[0] = 0.0 
    intervals[3] = min(t, t0, t - abs(x) + x0)
    intervals[2] = min(intervals[3], t + abs(x) - x0)
    intervals[1] = min(intervals[3], t - abs(x) - x0)
    for i in range(4):
        if intervals[i] < 0:
            intervals[i] = 0
    return intervals    
    
# @cfunc("float64(float64, float64, float64, float64)")
# @jit
def pyF(s, tau, t, x):
    xp = x-s
    tp = t - tau
    if tp != 0 and abs(xp) <= tp:     
        return math.exp(-tp)/2/tp
    else:
        return 0
    
def f1(t, tau, x0):
    return -x0 * expi(tau-t)
    
def f2(t, tau, x0, x):
    if tau != t:
        return 0.5*((-x0 + abs(x)) * expi(tau-t) + math.exp(tau - t))
    else:
        return 0.5 * math.exp(0.0)

def f3(t, tau, x0, x):
    return math.exp(tau-t)


def uncollided_square_source(x, t, x0, t0):
    tau_1 = 0.0
    end = min(t0, t - abs(x) + x0, t)
    if end <= 0.0:
        return 0.0
    tau_2 = min(end, t - x0 - abs(x))
    if tau_2 < 0.0:
        tau_2 = 0.0
    tau_3 = min(end, t - x0 + abs(x))
    if tau_3 < 0.0:
        tau_3 = 0.0
    tau_4 = end
    
    t1 = f1(t, tau_2, x0) - f1(t, tau_1, x0)
    t2 = f2(t, tau_3, x0, x) - f2(t, tau_2, x0, x)
    t3 = f3(t, tau_4, x0, x) - f3(t, tau_3, x0, x)
    
    return t1 + t2 + t3


def uncollided_square_IC(xx, t, x0):
    temp = 0.0
    
    if (t <= x0) and (xx >= -x0 + t) and (xx <= x0 - t):
        temp = math.exp(-t)
    elif t > x0  and (-t + x0 <=  xx) and (t - x0 >= xx):
        temp = math.exp(-t) * x0 / (t + 1e-12)
    elif (xx < t + x0) and (xx > -t - x0):
        if (x0 - xx >= t) and (x0 + xx <= t):
            temp = math.exp(-t)*(t + xx + x0)/(2.0 * t + 1e-12)
        elif (x0 - xx <= t) and (x0 + xx >= t):
            temp = math.exp(-t)*(t - xx + x0)/(2.0 * t + 1e-12)
            
    return temp

@njit
def gaussian_source_integrand(tau, t, x):
    abx = abs(x)
    temp = tau*0
    # for i in range(tau.size):
    tp = t - tau
        
    if tp != 0:
        erf1 = math.erf(2*(tp - abx)) 
        erf2 = math.erf(2*(tp + abx))
        temp = math.exp(-tp)* (erf1 + erf2) / tp
    else:
        temp = 0.0
    return temp

## I believe this is incorrect because it does not integrate over angle

# def uncollided_gauss_2D_integrand(s, rho, t, x0):
#     if rho**2 + s**2 -2*rho*s > 0:
#         eta = math.sqrt(rho**2 + s**2 - 2*rho*s)
    
#         if abs(eta) < 1 and eta > 0:
        
#             # res = s*0 * math.exp(-s**2/x0**2) / math.sqrt(1-eta**2) * math.exp(-t)/t/t
#             res = s *  math.exp(-s**2/x0**2) / math.sqrt(1-eta**2) * math.exp(-t)/t/t
#         else:
#             res = 0
#     else:
#         res = 0
#     return res
##################functions for integrating sources############################
def find_intervals_time(t, t0, x, s):
    a = 0 
    b = min(t, t0, t - abs(x-s))
    if b < 0:
        b = 0
    return [a,b]

################ solution checking functions ##################################
def check_gaussian_tail(phi, tol):
    index = 0
    for count, val in enumerate(phi):
        if phi[count] < tol:
            index = count
            break
        
    if index == 0 and phi[-1] > tol :
        print("solution is not sufficiently small. Extend evaluation interval The solution at the edge is ", phi[-1])
    return index
        
############# low level callable functions ####################################
def jit_F1(integrand_function):
    jitted_function = numba.jit(integrand_function, nopython=True)
    @cfunc(float64(intc, CPointer(float64)))
    def wrapped(n, xx):
        values = carray(xx,n)
        return jitted_function(values)
    return LowLevelCallable(wrapped.ctypes)



@njit
def source(s, source_type):
    if source_type == 0:     # square 
        return 1.0
    elif source_type == 1:   # gaussian 
        return np.exp(-4*s*s)
    
@njit 
def heaviside(arg):
    if arg >= 0.0:
        return 1.0
    else:
        return 0.0

@jit_F1
def F1(args):
    """ The integrand for the triple integral for the collided solution args = (u, s, tau, x, t)
    """
    u = args[0]
    s = args[1]
    tau = args[2]
    x = args[3]
    t = args[4]
    source_type = args[5]
    
    ## define new variables  ##
    xp = x-s
    tp = t-tau
    if abs(xp) < tp and tp > 0:
        
        eta = xp/tp
        
        ## find xi ##
        q = (1+eta)/(1-eta)
        zz = np.tan(u/2)
        xi = (np.log(q) + u*1j)/(eta + zz*1j)
        if abs(xi.real) < 1e-15:
            xi = 0.0 + xi.imag
        if abs(xi.imag) < 1e-15:
            xi = xi.real + 0.0*1j
        
        complex_term = np.exp(tp*((1 - eta**2)*xi/2.))*xi**2

        res = (1/np.cos(u/2.0))**2*complex_term.real * (1/math.pi/8) * (1 - eta**2) * math.exp(-tp) * source(s, source_type)
    
        return res
    
    else:
        return 0.0
    
@jit_F1
def F1_spacefirst(args):
    """ The integrand for the triple integral for the collided solution args = (u, s, tau, x, t)
    """
    u = args[0]
    s = args[2]
    tau = args[1]
    x = args[3]
    t = args[4]
    source_type = args[5]
    
    ## define new variables  ##
    xp = x-s
    tp = t-tau
    if abs(xp) < tp and tp > 0:
        
        eta = xp/tp
        
        ## find xi ##
        q = (1+eta)/(1-eta)
        zz = np.tan(u/2)
        xi = (np.log(q) + u*1j)/(eta + zz*1j)
        # if abs(xi.real) < 1e-15:
        #     xi = 0.0 + xi.imag
        # if abs(xi.imag) < 1e-15:
        #     xi = xi.real + 0.0*1j
        
        complex_term = np.exp(tp*((1 - eta**2)*xi/2.))*xi**2

        res = (1/np.cos(u/2.0))**2*complex_term.real * (1/math.pi/8) * (1 - eta**2) * math.exp(-tp) * source(s, source_type)
    
        return res
    
    else:
        return 0.0
    


@jit_F1
def F(args):
    """ integrand for the double integral for the uncollided solution. ags = s, tau, t, x
    the  sqrt(pi)/8 is left out 
    """
    s = args[0]
    tau = args[1]
    t = args[2]
    x = args[3]
    source_type = args[4]
    ## define new variables
    xp = x - s
    tp = t - tau
    ###
    if 1 - abs(xp/tp) > 0.0 :  
        return math.exp(-tp)/2/tp * source(s, source_type)
    else:
        return 0.0
    
@jit_F1
def F_gaussian_source(args):
    tau = args[0]
    t = args[1]
    x = args[2]
    
    abx = abs(x)
    tp = t - tau
    
    if tp != 0:
        erf1 = math.erf(2*(tp - abx)) 
        erf2 = math.erf(2*(tp + abx))
        return math.exp(-tp)* (erf1 + erf2) / tp
    else:
        return 0.0

@njit 
def point_collided_2(u, r, t):
    eta = r/t
    c = 1
    if eta >= 1:
        return 0.0
    else:
        # first = math.exp(-t)/4/math.pi/r/t * math.log((1 + eta)/(1-eta)) * c
        q = (1+eta)/(1-eta)
        zz = np.tan(u/2)
        xi = (np.log(q) + u*1j)/(eta + zz*1j)
        # exp_arg = c * t * (1 - eta**2) * xi / 2
        # complex_term = (eta + 1j * zz)*xi**3 * np.exp(exp_arg)
        complex_term = (eta + 1j * zz) * np.exp(c*t*((1 - eta**2)*xi/2.))*xi**3
        
        result =  (1/2/math.pi) * math.exp(-t)/4/math.pi/r * (c/2)**2 * (1 - eta**2) * (1/math.cos(u/2))**2 * complex_term.real
        
    return result 

@njit
def point_collided_1(r,t):
    
    eta = r/t
    c = 1
    if eta >= 1:
        return 0.0
    else:
        result = math.exp(-t)/4/math.pi/r/t * math.log((1 + eta)/(1-eta)) * c
    return result 

@jit_F1
def F_line_source_2(args):
    omega = args[0]
    u = args[1]
    r = args[2]
    t = args[3]
    
    eta = r/t
    
    if eta < 1:
        r_arg = t * math.sqrt(eta**2 + omega**2)
        
        return 2 * t * point_collided_2(u, r_arg, t)
    else:
        return 0.0
@jit_F1
def F_line_source_1(args):
    omega = args[0]
    r = args[1]
    t = args[2]
    
    eta = r/t
    
    if eta < 1:
        r_arg = t * math.sqrt(eta**2 + omega**2)
        
        return 2 * t * point_collided_1(r_arg, t)
    else:
        return 0.0
    

@jit_F1
def F2_2D_gaussian_pulse(args):
    u = args[0]
    omega = args[1]
    thetap = args[2]
    s = args[3]  # dummy radius 
    rho = args[4]
    theta = args[5]
    t = args[6]
    x0 = args[7]
    
    x = rho * math.cos(theta)
    y = rho * math.sin(theta)
    q = s * math.cos(thetap)
    v = s * math.sin(thetap)
    new_r = math.sqrt((x-q)**2 + (y-v)**2)
    eta = new_r/t
    
    if eta < 1:
        r_arg = t * math.sqrt(eta**2 + omega**2)
        return s * 2 * t * point_collided_2(u, r_arg, t) * math.exp(-s**2/x0**2)
    else: 
        return 0 
@jit_F1
def F1_2D_gaussian_pulse(args):
    omega = args[0]
    thetap = args[1]
    s = args[2]
    rho = args[3]
    theta = args[4]
    t = args[5]
    x0 = args[6]
    
    x = rho * math.cos(theta)
    y = rho * math.sin(theta)
    q = s * math.cos(thetap)
    v = s * math.sin(thetap)
    new_r = math.sqrt((x-q)**2 + (y-v)**2)
    eta = new_r/t
    
    if eta < 1:
        r_arg = t * math.sqrt(eta**2 + omega**2)
        return s * 2 * t * point_collided_1(r_arg, t) * math.exp(-s**2/x0**2)
    else: 
        return 0 
    
############################## 2D functions ###################################
@njit
def eta_func_2d_gauss_cartesian(x, s, y, v, t):
    res = (x-s)**2 + (y-v)**2
    return math.sqrt(res)/t

def uncollided_gauss_2D_integrand(s, v, x, y, t):
    res = 0.0
    eta = eta_func_2d_gauss_cartesian(x, s, y, v, t)
    if eta < 1:
        ft = math.exp(-t)/2/math.pi/t/t
        garg = -(s**2 + v**2)/0.5**2  # should make this variable x0
        gt = math.exp(garg)
        res = ft / math.sqrt(1-eta**2) * gt  
    return res

def find_intervals_2D_gaussian_s(r, t, theta, thetap):
    sqrt_term = -r**2 + 2*t**2 + r**2*math.cos(2*theta-2*thetap)
    a = 0.0
    b = 0.0
    if sqrt_term >=0:
        denominator = 2*(math.cos(thetap)**2 + math.sin(thetap)**2)
        t2 = 2 * r * (math.cos(theta) * math.cos(thetap) + math.sin(theta) * math.sin(thetap))
        a = (-math.sqrt(2) * math.sqrt(sqrt_term) + t2)/denominator
        b = (math.sqrt(2) * math.sqrt(sqrt_term) + t2)/denominator
        if a < 0:
            a = 0
        if b < 0:
            b = 0
    return [a,b]

######################saving solution##########################################
def make_benchmark_file_structure():
    data_folder = Path("moving_mesh_transport/benchmarks")
    bench_file_path = data_folder / 'benchmarks.hdf5'
    source_name_list = ['plane_IC', 'square_IC', 'square_source', 'gaussian_IC', 'gaussian_source', 'gaussian_IC_2D', 'line_source']
    
    f = h5py.File(bench_file_path, "a")
    
    for source_name in source_name_list:
        if f.__contains__(source_name):
            del f[source_name]
        f.create_group(source_name)
    
    f.close()

def write_to_file(xs, phi, uncol, tfinal, source_name, npnts):
    data_folder = Path("benchmarks")
    bench_file_path = data_folder / 'benchmarks.hdf5'
    
    with h5py.File(bench_file_path,'r+') as f:
        if f.__contains__(source_name + f'/t = {tfinal}'):
            del f[source_name + f'/t = {tfinal}'] 
        f.create_dataset(source_name + f'/t = {tfinal}', (3, npnts), dtype = "f", data=(xs, phi, uncol))
    f.close()
