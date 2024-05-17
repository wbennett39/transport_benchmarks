#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 09:14:26 2022

@author: bennett
"""
import matplotlib.pyplot as plt
from .make_plots import rms_plotter

def plot_all_rms_cells(tfinal, M):
    source_type_list  = ["plane_IC", "square_IC", "square_s", "gaussian_IC", "gaussian_s"]
    case_list_1 = [True, True, False, False]
    case_list_2 = [True, False, True, False]
    
    for count1, source in enumerate(source_type_list):
        plotter = rms_plotter(tfinal, M, source)
        for count2, uncollided in enumerate(case_list_1):
            moving = case_list_2[count2]
            plotter.load_RMS_data(uncollided, moving)
            if count2 == 0:
                clear = False
            if source == "plane_IC" and uncollided == False and moving == True:
                print("skipping no-uncol moving case for plane IC")
            else:
                plotter.plot_RMS_vs_cells(count1+1, clear)
                
    plotter = rms_plotter(tfinal, 2, "MMS")
    plotter.load_RMS_data(uncollided = False, moving = True)
    plotter.plot_RMS_vs_cells(6, clear)
    
    plotter = rms_plotter(tfinal, 4, "MMS")
    plotter.load_RMS_data(uncollided = False, moving = True)
    plotter.plot_RMS_vs_cells(6, clear)
    
    plotter = rms_plotter(tfinal, 6, "MMS")
    plotter.load_RMS_data(uncollided = False, moving = True)
    plotter.plot_RMS_vs_cells(6, clear)
    
def plot_all_rms_times(tfinal, M):
    source_type_list  = ["plane_IC", "square_IC", "square_s", "gaussian_IC", "gaussian_s"]
    case_list_1 = [True, True, False, False]
    case_list_2 = [True, False, True, False]
    
    for count1, source in enumerate(source_type_list):
        plotter = rms_plotter(tfinal, M, source)
        for count2, uncollided in enumerate(case_list_1):
            moving = case_list_2[count2]
            plotter.load_RMS_data(uncollided, moving)
            if count2 == 0:
                clear = False
            if source == "plane_IC" and uncollided == False and moving == True:
                print("skipping no-uncol moving case for plane IC")
            else:
                plotter.plot_RMS_vs_times(count1+1, clear)
                
    plotter = rms_plotter(tfinal, 2, "MMS")
    plotter.load_RMS_data(uncollided = False, moving = True)
    plotter.plot_RMS_vs_times(6, clear)
    
    plotter = rms_plotter(tfinal, 4, "MMS")
    plotter.load_RMS_data(uncollided = False, moving = True)
    plotter.plot_RMS_vs_times(6, clear)
    
    plotter = rms_plotter(tfinal, 6, "MMS")
    plotter.load_RMS_data(uncollided = False, moving = True)
    plotter.plot_RMS_vs_times(6, clear)
    
def compare_rms(tfinal, M):
    source_type_list  = ["plane_IC", "square_IC", "square_s", "gaussian_IC", "gaussian_s"]
    case_list_1 = [True, True, False, False]
    case_list_2 = [True, False, True, False]
    
    for count1, source in enumerate(source_type_list):
        plotter = rms_plotter(tfinal, M, source)
        plotter.load_RMS_data(uncollided = True, moving = True)
        RMS_list_case_1 = plotter.RMS
        if source != 'plane_IC':
            plotter.load_RMS_data(uncollided = False, moving = True)
            RMS_list_case_2 = plotter.RMS
        plotter.load_RMS_data(uncollided = True, moving = False)
        RMS_list_case_3 = plotter.RMS
        plotter.load_RMS_data(uncollided = False, moving = False)
        RMS_list_case_4 = plotter.RMS
        diff2 = 0
        diff3 = 0
        diff4 = 0
        for i in range(RMS_list_case_1.size): ## calculate percent error
            if source != 'plane_IC':
                diff2 += (RMS_list_case_2[i] - RMS_list_case_1[i])/RMS_list_case_1[i]
            diff3 += (RMS_list_case_3[i] - RMS_list_case_1[i])/RMS_list_case_1[i]
            diff4 += (RMS_list_case_4[i] - RMS_list_case_1[i])/RMS_list_case_1[i]
        diff2 = diff2/RMS_list_case_1.size
        diff3 = diff3/RMS_list_case_1.size
        diff4 = diff4/RMS_list_case_1.size
        
        print("--------------")
        print(source)
        print("case 1 is moving with uncollided")
        print(diff2, "average percent difference between case 1 and moving w/o uncollided ")
        print(diff3, "average percent difference between case 1 and static w/ uncollided ")
        print(diff4, "average percent difference between case 1 and static w/o uncollided ")
        
def plot_all_benchmarks(tfinal):
    M = 3
    source_list = ["plane_IC", "square_IC", "square_source", "gaussian_IC", "gaussian_source", "MMS", "gaussian_IC_2D", "line_source", "shell_source"]
    for count, source in enumerate(source_list):
        print(source)
        plotter = rms_plotter(tfinal, M, source)
        plotter.plot_bench(tfinal, source, count)
        