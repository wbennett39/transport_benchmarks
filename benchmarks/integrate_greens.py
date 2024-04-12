#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 20:25:02 2022

@author: bennett
"""

from .benchmarks import make_benchmark
import numpy as np
    

def plane_IC(t, npnts, c = 1.0):
    fign = 1
    bench_class = make_benchmark('plane_IC', 1e-16, 1e-16, 0, c)
    bench_class.integrate(t, npnts)
    bench_class.save()
    bench_class.plot(fign)
    return bench_class.xs, bench_class.collided_sol, bench_class.uncollided_sol
    
def square_IC(t, npnts, x0 = 0.5, c = 1.0):
    fign = 2
    bench_class = make_benchmark('square_IC', x0, 1e-16, 0, c)
    bench_class.integrate(t, npnts)
    bench_class.save()
    bench_class.plot(fign)
    return bench_class.xs, bench_class.collided_sol, bench_class.uncollided_sol
    
def square_source(t, npnts, x0 = 0.5, t0 = 5, c = 1.0):
    fign = 3
    bench_class = make_benchmark('square_source', x0, t0, 0, c)
    bench_class.integrate(t, npnts)
    bench_class.save()
    bench_class.plot(fign)
    return bench_class.xs, bench_class.collided_sol, bench_class.uncollided_sol
    
def gaussian_IC(t, npnts, sigma = 0.5, c = 1.0):
    fign = 3
    bench_class = make_benchmark('gaussian_IC', 4.0, 1e-16, sigma, c)
    bench_class.integrate(t, npnts)
    bench_class.save()
    bench_class.plot(fign)
    return bench_class.xs, bench_class.collided_sol, bench_class.uncollided_sol


def gaussian_source(t, npnts, t0 = 5, sigma = 0.5, c = 1.0):
    fign = 4
    bench_class = make_benchmark('gaussian_source', 4.0, t0, sigma, c)
    bench_class.integrate(t, npnts)
    bench_class.save()
    bench_class.plot(fign)
    return bench_class.xs, bench_class.collided_sol, bench_class.uncollided_sol
    
def gaussian_IC_2D(t, npnts, sigma = 0.5, c = 1.0):
    fign = 5
    bench_class = make_benchmark('gaussian_IC_2D', 0.5, 1e-16, sigma, c)
    bench_class.integrate(t, npnts)
    bench_class.save()
    bench_class.plot(fign)
    return bench_class.xs, bench_class.collided_sol, bench_class.uncollided_sol

def line_source(t, npnts, c =1.0):
    fign = 6
    bench_class = make_benchmark("line_source", 0.5, 1e-16, 0, c)
    bench_class.integrate(t, npnts)
    bench_class.save()
    bench_class.plot(fign)
    return bench_class.xs, bench_class.collided_sol, bench_class.uncollided_sol
    
def point_source(t, npnts, c =1.0):
    fign = 7
    bench_class = make_benchmark("point_source", 1e-16, 1e-16, 0, c)
    bench_class.integrate(t, npnts)
    bench_class.save()
    bench_class.plot(fign)
    return bench_class.xs, bench_class.collided_sol, bench_class.uncollided_sol

def shell_source(t, npnts, c = 1.0, x0 = 0.5, choose_xs = False, xpnts = np.array([0.0])):
    fign = 8
    bench_class = make_benchmark("shell_source", x0, 1e-16, 0, c, choose_xs, xpnts)
    bench_class.integrate(t, npnts)
    bench_class.save()
    bench_class.plot(fign)
    return bench_class.xs, bench_class.collided_sol, bench_class.uncollided_sol
    
    
def do_all(npnts = [2500, 2500, 2500, 2500, 2500, 2500, 2500]):
    print("running plane IC")
    print("---    ---   ---   ---")
    plane_IC(1, npnts[0])
    plane_IC(5, npnts[0])
    plane_IC(10, npnts[0])
    print("plane finished")
    print("              ")
    print("running square IC")
    print("---    ---   ---   ---")
    square_IC(1, npnts[1])
    square_IC(5, npnts[1])
    square_IC(10, npnts[1])
    print("square IC finished")
    print("              ")
    print("---    ---   ---   ---")
    print("running Gaussian IC")
    print("---    ---   ---   ---")
    gaussian_IC(1, npnts[2])
    gaussian_IC(5, npnts[2])
    gaussian_IC(10, npnts[2])
    print("Gaussian IC finished")
    print("              ")
    print("running square source")
    print("---    ---   ---   ---")
    square_source(1, npnts[3])
    square_source(5, npnts[3])
    square_source(10, npnts[3])
    print("square source finished")
    print("              ")
    print("running Gaussian source")
    print("---    ---   ---   ---")
    gaussian_source(1, npnts[4])
    gaussian_source(5, npnts[4])
    gaussian_source(10, npnts[4])
    print("gaussian source finished")
    print("              ")
    print("running Gaussian line IC")
    print("---    ---   ---   ---")
    gaussian_IC_2D(1, npnts[5])
    gaussian_IC_2D(5, npnts[5])
    gaussian_IC_2D(10, npnts[5])
    print("Gaussian line IC finished")
    print("              ")
    print("running line IC")
    print("---    ---   ---   ---")
    line_source(1, npnts[6])
    line_source(5, npnts[6])
    line_source(10, npnts[6])
    print("line IC finished")
    
