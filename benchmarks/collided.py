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
from .benchmark_functions import  P1_su_olson_mat_integrand
from .benchmark_functions import  P1_gaussian_mat_integrand
from .benchmark_functions import  P1_su_olson_term1_integrand, P1_su_olson_term2_integrand
from .benchmark_functions import  P1_gaussian_term1_integrand, P1_gaussian_term2_integrand
from .benchmark_functions import  find_su_olson_interval
from .benchmark_functions import  point_collided_1, point_collided_2
import scipy.integrate as integrate
import math
import numpy as np
from numba import prange
###############################################################################

def opts0(*args, **kwargs):
       return {'limit':1000000000, 'epsabs':1.5e-12, 'epsrel':1.5e-12}
def opts1(*args, **kwargs):
       return {'limit':10000, 'epsabs':1.5e-8, 'epsrel':1.5e-8}
def opts2(*args, **kwargs):
       return {'limit':1000, 'epsabs':1.5e-8, 'epsrel':1.5e-8}  # used for sources 

class collided_class:
    def __init__(self, source_type, x0, t0, sigma):
        self.x0 = x0
        self.t0 = t0
        self.source_type = source_type
        self.sigma = sigma
        
    
    def plane_IC(self, xs,t, c):
        temp = xs*0
        for ix in range(xs.size):
            temp[ix] = integrate.nquad(F1, [[0, math.pi]], args =  (0.0, 0.0, xs[ix], t, 0, c), opts = [opts0])[0]
        return temp
    
    def square_IC(self, xs, t, c):
        temp = xs*0
        counter = 0 
        for ix in range(xs.size):
            if counter == 100:
                print("x=", xs[ix])
                counter = 0 
            left_space = xs[ix]-t
            right_space = xs[ix] + t
            left_int_bounds = max(-self.x0, left_space)
            right_int_bounds = min(self.x0, right_space)
            temp[ix] = integrate.nquad(F1, [[0, math.pi], [left_int_bounds, right_int_bounds]], args =  (0.0, xs[ix], t, 0, c), opts = [opts0, opts0])[0]
            counter += 1
        return temp
    
    def source_double_integral_time(self, s, x, t, t0, source, c):
        """ source function for the gaussian source and the square source (1-gaussian, 0-square)
        """
        ab = find_intervals_time(t, t0, x, s)
        solution = integrate.nquad(F1_spacefirst, [[0, math.pi], [ab[0],ab[1]]], args =  (s, x, t, source, c), opts = [opts1, opts1])[0]
        return solution
        
    def square_source(self, xs, t, c):
        temp = xs*0
        counter = 0 
        for ix in range(xs.size):
            if counter == 5:
                print('x=', xs[ix])
                counter = 0
            temp[ix] = integrate.nquad(self.source_double_integral_time, [[-self.x0, self.x0]], args = (xs[ix], t, self.t0, 0, c), opts = [opts2])[0]
            counter += 1
        return temp
    
    def gaussian_IC(self, xs, t, c):
        temp = xs*0
        # s_interval = [-np.inf, np.inf]
        
        for ix in range(xs.size):
            s_interval = [xs[ix]-t, xs[ix]+t]
            temp[ix] = integrate.nquad(F1, [[0, math.pi], s_interval], args =  (0.0, xs[ix], t, 1, c), opts = [opts0, opts0])[0]
        return temp

    def gaussian_source(self, xs, t, c):
        temp = xs*0
        for ix in range(xs.size):
            s_interval = [xs[ix]-t, xs[ix]+t]
            # s_interval = [-np.inf, np.inf]
            temp[ix] = integrate.nquad(self.source_double_integral_time, [s_interval], args = (xs[ix], t, self.t0, 1, c), opts = [opts2])[0]
        return temp
    
    
    ################## 2D #####################################################
    
    def gaussian_pulse_2D_double_integral(self, s, thetap, rho, t, theta, x0, c):
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
            rest_collided = integrate.nquad(F2_2D_gaussian_pulse, [[0, math.pi], [omega_a, omega_b]], args = (thetap, s, rho, theta, t,  x0, c), opts = [opts0, opts0])[0]
            first_collided = integrate.nquad(F1_2D_gaussian_pulse, [[omega_a, omega_b]], args = (thetap, s, rho, theta, t,  x0, c), opts = [opts0])[0]
            res = rest_collided + first_collided
        return res

    def collided_gauss_2D_s(self, thetap, rho, t, x0, c):
        """ integrates over s
        """
        theta = 0
        # b = np.inf
        # b = rho + t
        # interval = [a, b]
        interval = find_intervals_2D_gaussian_s(rho, t, theta, thetap)
        res = integrate.nquad(self.gaussian_pulse_2D_double_integral, [interval], args = (thetap, rho, t, theta, x0, c), opts = [opts1])[0]
        
        return res

    def collided_gauss_2D_theta(self, rho, t, x0, c):
        """ integrates over thetap
        """

        res = integrate.nquad(self.collided_gauss_2D_s, [[0, math.pi*2]], args = (rho, t, x0, c), opts = [opts1])[0]
        
        return res
    
    def F_line_source_2_first_integral(self, u, rho, t, c):
        eta = rho/t
        res = 0.0
        if eta <1:
            omega_b = math.sqrt(1-eta**2)
            res = integrate.nquad(F_line_source_2, [[0, omega_b]], args = (u, rho, t, c), opts = [opts1])[0]
        return res
    
    def collided_line_source(self, rho, t, c):
        
        res1  = integrate.nquad(self.F_line_source_2_first_integral, [[0, math.pi]], args = (rho, t, c), opts = [opts0])[0]
        res2 = integrate.nquad(F_line_source_1, [[0, math.pi]], args = (rho, t, c), opts = [opts1])[0]
        
        return res1 + res2
    
    def gaussian_IC_2D(self, rhos, t, c):
        # standard deviation is not set with this one, varies with x0
        # multiply integrand by s?
        temp = rhos*0
        for ix in range(rhos.size):
            rho = rhos[ix]
            temp[ix] = self.collided_gauss_2D_theta(rho, t, self.x0, c)
        return temp
    
    
    def line_source(self, rhos, t, c):
        temp = rhos*0
        for ix in range(rhos.size):
            rho = rhos[ix]
            temp[ix] = self.collided_line_source(rho, t, c)
        return temp
    
    def point_source(self, rhos, t, c):
        temp = rhos*0
        for ix in range(rhos.size):
            rho = rhos[ix]
            res1  = integrate.nquad(point_collided_2, [[0, math.pi]], args = (rho, t, c), opts = [opts0])[0]
            res2 = point_collided_1(rho, t, c)
            temp[ix] =  res1 + res2
        return temp

    def shell_source(self, rhos, t, c):
        R = self.x0
        # q0 = 4 * math.pi * R**3 / 3
        temp = rhos *0
        for ix in range(rhos.size):
            r = rhos[ix]
            # integrand = lambda omega: omega * (self.plane_IC(np.array([np.abs(R*omega-r)]), t, c) - self.plane_IC(np.array([np.abs(R*omega+r)]), t, c)) 
            integrand1 = lambda omega: integrate.nquad(F1, [[0, math.pi]], args =  (0.0, 0.0, abs(R*omega - r), t, 0, c), opts = [opts0])[0]
            integrand2 = lambda omega: integrate.nquad(F1, [[0, math.pi]], args =  (0.0, 0.0, abs(R*omega + r), t, 0, c), opts = [opts0])[0]
            integrand = lambda omega: omega * (integrand1(omega) - integrand2(omega))
            res = integrate.nquad(integrand, [[0, 1]], opts = [opts0])[0]
            temp[ix] = res * 3 / 4 /math.pi /R / (r + 1e-16)
        return temp 
 
    ########## su olson problem ########################################
    
    def P1_su_olson_rad_first_interval(self, tau, x, t):
        s_range = find_su_olson_interval(self.x0, tau, x)
        res = integrate.nquad(P1_su_olson_term2_integrand, [s_range], args = (tau, x, t), opts = [opts1, opts1])[0]
        return res
    
    def P1_su_olson_rad(self, xs, t):
        temp = xs * 0 
        if t <= self.t0: 
            trange = [0, t]
        else:
            trange = [t-self.t0, t]
            
        for ix in prange(xs.size):
            s_range = find_su_olson_interval(self.x0, t, xs[ix])
            term1 = integrate.nquad(P1_su_olson_term1_integrand, [s_range], args = (xs[ix], t), opts = [opts1])[0]
            term2 = integrate.nquad(self.P1_su_olson_rad_first_interval, [trange], args = (xs[ix], t), opts = [opts1])[0]
            
            temp[ix] = term1 + term2
        return temp 
    
    
    
    def P1_su_olson_mat_first_integral(self, tau, x, t):
        s_range = find_su_olson_interval(self.x0, tau, x)
        res = integrate.nquad(P1_su_olson_mat_integrand, [s_range], args = (tau, x, t), opts = [opts1])[0]
        return res
        
    def P1_su_olson_mat(self, xs, t):
        temp = xs * 0 
        if t <= self.t0: 
            trange = [0, t]
        else:
            trange = [t-self.t0, t]
            
        for ix in prange(xs.size):
                temp[ix] = integrate.nquad(self.P1_su_olson_mat_first_integral, [trange], args = (xs[ix], t), opts = [opts1])[0]
        return temp 
    
    def P1_gaussian_rad_first_interval(self, tau, x, t, sigma):
        s_range = [(-math.sqrt(3)*tau + 3*x)/3,(math.sqrt(3)*tau + 3*x)/3]
        res = integrate.nquad(P1_gaussian_term2_integrand, [s_range], args = (tau, x, t, sigma), opts = [opts1, opts1])[0]
        return res
    
    def P1_gaussian_rad(self, xs, t, sigma):
        temp = xs * 0 
        if t <= self.t0: 
            trange = [0, t]
        else:
            trange = [t-self.t0, t]
            
        for ix in prange(xs.size):
            s_range = [(-math.sqrt(3)*t + 3*xs[ix])/3,(math.sqrt(3)*t + 3*xs[ix])/3]
            term1 = integrate.nquad(P1_gaussian_term1_integrand, [s_range], args = (xs[ix], t, sigma), opts = [opts1])[0]
            term2 = integrate.nquad(self.P1_gaussian_rad_first_interval, [trange], args = (xs[ix], t, sigma), opts = [opts1])[0]
            # term1 = 0
            # term2 = 0
            temp[ix] = term1 + term2
        return temp 
    
    def P1_gaussian_mat_first_integral(self, tau, x, t, sigma):
        s_range = [(-math.sqrt(3)*tau + 3*x)/3,(math.sqrt(3)*tau + 3*x)/3]
        res = integrate.nquad(P1_gaussian_mat_integrand, [s_range], args = (tau, x, t, sigma), opts = [opts1])[0]
        return res

    def P1_gaussian_mat(self, xs, t, sigma):
        temp = xs * 0 
        if t <= self.t0: 
            trange = [0, t]
        else:
            trange = [t-self.t0, t]
            
        for ix in prange(xs.size):
                temp[ix] = integrate.nquad(self.P1_gaussian_mat_first_integral, [trange], args = (xs[ix], t, sigma), opts = [opts1])[0]
        return temp 
    
    
    def __call__(self, xs, t, c):
        if self.source_type == 'plane_IC':
            return self.plane_IC(xs, t, c)
        elif self.source_type == 'square_IC':
            return self.square_IC(xs, t, c)
        elif self.source_type == 'square_source':
            return self.square_source(xs, t, c)
        elif self.source_type == 'gaussian_IC':                
            return self.gaussian_IC(xs, t, c)
        elif self.source_type == 'gaussian_source':
            return self.gaussian_source(xs, t, c)
        elif self.source_type == 'gaussian_IC_2D':
            return self.gaussian_IC_2D(xs, t, c)
        elif self.source_type == "line_source":
            return self.line_source(xs, t, c)
        elif self.source_type == "P1_su_olson_rad":
            return self.P1_su_olson_rad(xs, t)
        elif self.source_type == "P1_su_olson_mat":
            return self.P1_su_olson_mat(xs, t)
        elif self.source_type == "P1_gaussian_rad":
            return self.P1_gaussian_rad(xs, t, self.sigma)
        elif self.source_type == "P1_gaussian_mat":
            return self.P1_gaussian_mat(xs, t, self.sigma)
        elif self.source_type == 'point_source':
            return self.point_source(xs, t, c)
        elif self.source_type == 'shell_source':
            return self.shell_source(xs, t, c)
            
        