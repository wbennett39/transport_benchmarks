#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 12:49:44 2022

@author: bennett
"""
from .benchmark_functions import F1, F1_spacefirst, find_intervals_time
from .benchmark_functions import F1_2D_gaussian_pulse,F2_2D_gaussian_pulse
from .benchmark_functions import find_intervals_2D_gaussian_s
from .benchmark_functions import F_line_source_1,  F_line_source_2
import scipy.integrate as integrate
import math
import numpy as np
###############################################################################

def opts0(*args, **kwargs):
       return {'limit':10000000}
def opts1(*args, **kwargs):
       return {'limit':10000000}
def opts2(*args, **kwargs):
       return {'limit':10000000}

class collided_class:
    def __init__(self, source_type, x0, t0, sigma):
        self.x0 = x0
        self.t0 = t0
        self.source_type = source_type
        self.sigma = sigma
    
    def plane_IC(self, xs,t):
        temp = xs*0
        for ix in range(xs.size):
            temp[ix] = integrate.nquad(F1, [[0, math.pi]], args =  (0.0, 0.0, xs[ix], t, 0, 0), opts = [opts0])[0]
        return temp
    
    def square_IC(self, xs, t):
        temp = xs*0
        for ix in range(xs.size):
            temp[ix] = integrate.nquad(F1, [[0, math.pi], [-self.x0, self.x0]], args =  (0.0, xs[ix], t, 0, 0, 0), opts = [opts0, opts1])[0]
        return temp
    
    def source_double_integral_time(self, s, x, t, t0, source):
        """ source function for the gaussian source and the square source (1-gaussian, 0-square)
        """
        ab = find_intervals_time(t, t0, x, s)
        solution = integrate.nquad(F1_spacefirst, [[0, math.pi], [ab[0],ab[1]]], args =  (s, x, t, source, self.sigma), opts = [opts0, opts1])[0]
        return solution
        
    def square_source(self, xs, t):
        temp = xs*0
        for ix in range(xs.size):
            temp[ix] = integrate.nquad(self.source_double_integral_time, [[-self.x0, self.x0]], args = (xs[ix], t, self.t0, 0), opts = [opts2])[0]
        return temp
    
    def gaussian_IC(self, xs, t):
        temp = xs*0
        # s_interval = [-np.inf, np.inf]
        
        for ix in range(xs.size):
            s_interval = [xs[ix]-t, xs[ix]+t]
            temp[ix] = integrate.nquad(F1, [[0, math.pi], s_interval], args =  (0.0, xs[ix], t, 1, self.sigma), opts = [opts0, opts1])[0]
        return temp

    def gaussian_source(self, xs, t):
        temp = xs*0
        for ix in range(xs.size):
            s_interval = [xs[ix]-t, xs[ix]+t]
            # s_interval = [-np.inf, np.inf]
            temp[ix] = integrate.nquad(self.source_double_integral_time, [s_interval], args = (xs[ix], t, self.t0, 1), opts = [opts2])[0]
        return temp
    
    
    ################## 2D #####################################################
    
    def gaussian_pulse_2D_double_integral(self, s, thetap, rho, t, theta, x0):
        """ integrates over u, omega
        """
        x = rho * math.cos(theta)
        y = rho * math.sin(theta)
        q = s * math.cos(thetap)
        v = s * math.sin(thetap)
        new_r = math.sqrt((x-q)**2 + (y-v)**2)
        eta = new_r/t
        omega_a = 0.0
        res = 0.0
        
        if eta < 1:
            omega_b = math.sqrt(1-eta**2)
            rest_collided = integrate.nquad(F2_2D_gaussian_pulse, [[0, math.pi], [omega_a, omega_b]], args = (thetap, s, rho, theta, t,  x0), opts = [opts0, opts0])[0]
            first_collided = integrate.nquad(F1_2D_gaussian_pulse, [[omega_a, omega_b]], args = (thetap, s, rho, theta, t,  x0), opts = [opts0])[0]
            res = rest_collided + first_collided
        return res

    def collided_gauss_2D_s(self, thetap, rho, t, x0):
        """ integrates over s
        """
        theta = 0
        # b = np.inf
        # b = rho + t
        # interval = [a, b]
        interval = find_intervals_2D_gaussian_s(rho, t, theta, thetap)
        res = integrate.nquad(self.gaussian_pulse_2D_double_integral, [interval], args = (thetap, rho, t, theta, x0), opts = [opts0])[0]
        
        return res

    def collided_gauss_2D_theta(self, rho, t, x0):
        """ integrates over thetap
        """

        res = integrate.nquad(self.collided_gauss_2D_s, [[0, math.pi*2]], args = (rho, t, x0), opts = [opts0])[0]
        
        return res
    
    def F_line_source_2_first_integral(self, u, rho, t):
        eta = rho/t
        res = 0.0
        if eta <1:
            omega_b = omega_b = math.sqrt(1-eta**2)
            res = integrate.nquad(F_line_source_2, [[0, omega_b]], args = (u, rho, t), opts = [opts0])[0]
        return res
    
    def collided_line_source(self, rho, t):
        
        res1  = integrate.nquad(self.F_line_source_2_first_integral, [[0, math.pi]], args = (rho, t), opts = [opts0])[0]
        res2 = integrate.nquad(F_line_source_1, [[0, math.pi]], args = (rho, t), opts = [opts0])[0]
        
        return res1 + res2
    
    def gaussian_IC_2D(self, rhos, t):
        # standard deviation is not set with this one, varies with x0
        # multiply integrand by s?
        temp = rhos*0
        for ix in range(rhos.size):
            rho = rhos[ix]
            temp[ix] = self.collided_gauss_2D_theta(rho, t, self.x0)
        return temp
    
    
    def line_source(self, rhos, t):
        temp = rhos*0
        for ix in range(rhos.size):
            rho = rhos[ix]
            temp[ix] = self.collided_line_source(rho, t)
        return temp
    
    def __call__(self, xs, t):
        if self.source_type == 'plane_IC':
            return self.plane_IC(xs, t)
        elif self.source_type == 'square_IC':
            return self.square_IC(xs, t)
        elif self.source_type == 'square_source':
            return self.square_source(xs, t)
        elif self.source_type == 'gaussian_IC':                
            return self.gaussian_IC(xs, t)
        elif self.source_type == 'gaussian_source':
            return self.gaussian_source(xs, t)
        elif self.source_type == 'gaussian_IC_2D':
            return self.gaussian_IC_2D(xs, t)
        elif self.source_type == "line_source":
            return self.line_source(xs, t)
        