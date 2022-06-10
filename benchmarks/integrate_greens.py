#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 20:25:02 2022

@author: bennett
"""

from .benchmarks import make_benchmark

def plane_IC(t, npnts):
    fign = 1
    bench_class = make_benchmark('plane_IC', 1e-16, 1e-16, 0)
    bench_class.integrate(t, npnts)
    bench_class.save()
    bench_class.plot(fign)
    
def square_IC(t, npnts, x0 = 0.5):
    fign = 2
    bench_class = make_benchmark('square_IC', x0, 1e-16, 0)
    bench_class.integrate(t, npnts)
    bench_class.save()
    bench_class.plot(fign)
    
def square_source(t, npnts, x0 = 0.5, t0 = 5):
    fign = 3
    bench_class = make_benchmark('square_source', x0, t0, 0)
    bench_class.integrate(t, npnts)
    bench_class.save()
    bench_class.plot(fign)
    
def gaussian_IC(t, npnts, sigma = 0.5):
    fign = 3
    bench_class = make_benchmark('gaussian_IC', 4.0, 1e-16, sigma)
    bench_class.integrate(t, npnts)
    bench_class.save()
    bench_class.plot(fign)


def gaussian_source(t, npnts, t0 = 5, sigma = 0.5):
    fign = 4
    bench_class = make_benchmark('gaussian_source', 4.0, t0, sigma)
    bench_class.integrate(t, npnts)
    bench_class.save()
    bench_class.plot(fign)
    
def gaussian_IC_2D(t, npnts):
    fign = 5
    bench_class = make_benchmark('gaussian_IC_2D', 0.5, 1e-16)
    bench_class.integrate(t, npnts)
    bench_class.save()
    bench_class.plot(fign)

def line_source(t, npnts):
    fign = 6
    bench_class = make_benchmark("line_source", 0.5, 1e-16)
    bench_class.integrate(t, npnts)
    bench_class.save()
    bench_class.plot(fign)
    
    
def do_all(npnts = [5000, 5000, 5000, 5000, 5000, 250, 500]):
    plane_IC(1, npnts[0])
    plane_IC(5, npnts[0])
    plane_IC(10, npnts[0])
    print("plane finished")
    square_IC(1, npnts[1])
    square_IC(5, npnts[1])
    square_IC(10, npnts[1])
    print("square IC finished")
    gaussian_IC(1, npnts[2])
    gaussian_IC(5, npnts[2])
    gaussian_IC(10, npnts[2])
    print("gaussian IC finished")
    square_source(1, npnts[3])
    square_source(5, npnts[3])
    square_source(10, npnts[3])
    print("square source finished")
    gaussian_source(1, npnts[4])
    gaussian_source(5, npnts[4])
    gaussian_source(10, npnts[4])
    print("gaussian source finished")
    gaussian_IC_2D(1, npnts[5])
    gaussian_IC_2D(5, npnts[5])
    gaussian_IC_2D(10, npnts[5])
    print("gaussian IC 2D finished")
    line_source(1, npnts[6])
    line_source(5, npnts[6])
    line_source(10, npnts[6])
    print("line source finished")
    