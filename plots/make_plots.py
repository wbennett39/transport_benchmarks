#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 15:40:39 2022

@author: bennett
"""

import matplotlib.pyplot as plt
import h5py 
from pathlib import Path
from ..load_bench import load_bench
import numpy as np
from .show import show

class rms_plotter:
    
    def __init__(self, tfinal, M, source_name):
        data_folder = Path("moving_mesh_transport")
        self.data_file_path = data_folder / 'run_data_RMS.h5'
        self.plot_file_path = data_folder / "plots"
        # self.case_list = ["uncol_mov", "no_uncol_stat", "uncol_stat", "no_uncol_stat"]
        self.tfinal = tfinal
        self.M = M
        self.source_type_list  = ["plane_IC", "square_IC", "square_s", "gaussian_IC", "MMS", "gaussian_s"]
        self.source_name = source_name
        
    def load_RMS_data(self, uncollided = True, moving = True):

        self.uncollided = uncollided 
        self.moving = moving
        
        if self.M == 2:
            self.mkr = "o"
        elif self.M == 4:
            self.mkr = "^"
        elif self.M == 6:
            self.mkr = "s"
        else:
            self.mkr = "p"
            
        if self.moving == True:
            self.line_mkr = "-"
        elif self.moving == False:
            self.line_mkr = "--"
        if self.uncollided == True:
            self.clr = "b"
            self.mfc = "b"
        elif self.uncollided == False:
            self.clr = "r"
            self.mfc = "none"
            
        f = h5py.File(self.data_file_path, 'r')
        # f = h5py.File("run_data_RMS.h5", 'r')
        
        self.dest_str = str(self.source_name + "/" + "t="  + str(self.tfinal) + "/" + "RMS")
        data_str = self.uncollided * ("uncollided_")  + (not(self.uncollided))  * ("no_uncollided_")  + self.moving * ("moving_") + (not(self.moving)) * ("static_") + "M_" + str(self.M)
        
        data = f[self.dest_str + '/' + data_str]
        self.cells = data[0]
        self.RMS = data[1]
        self.angles = data[2]
        self.times = data[3]
        f.close()
    
    def plot_RMS_vs_cells(self, fign = 1, clear = False):
        plt.ion()
        plt.figure(fign)
        if clear == True:
            plt.clf()
        plt.xlabel("cells")
        plt.ylabel("RMSE")
        plt.title(f"{self.source_name} t = {self.tfinal}")
        plt.loglog(self.cells, self.RMS, self.line_mkr + self.mkr, c = self.clr, mfc = self.mfc)
        plt.savefig(self.plot_file_path / "RMS_plots" / f"{self.source_name}_t={self.tfinal}_RMSE_vs_cells.pdf")
        plt.show(block = False)
        
    def plot_RMS_vs_times(self, fign = 1, clear = False):
        plt.ion()
        plt.figure(fign)
        if clear == True:
            plt.clf()
        plt.xlabel("average run time")
        plt.ylabel("RMSE")
        plt.title(f"{self.source_name} t = {self.tfinal}")
        print(self.times)
        plt.loglog(self.times, self.RMS, self.line_mkr + self.mkr, c = self.clr, mfc = self.mfc)
        plt.savefig(self.plot_file_path / "RMS_plots" / f"{self.source_name}_t={self.tfinal}_times_vs_cells.pdf")
        plt.show(block = False)
        
    def plot_bench(self, tfinal, source_name, fign):
        plt.figure(fign)
        fntsize = 30
        plt.ion()
        plt.xlabel("x", fontsize = fntsize)
        plt.ylabel("scalar flux", fontsize = fntsize)
        file_path_string = str(self.plot_file_path) + "/" + "benchmark_plots"
        npnts = 10000
        if source_name == "plane_IC":
            source_type = np.array([1,0,0,0,0,0,0,0])
            x0 = 1e-11
            bench = load_bench(source_type, tfinal, x0)
            xs = np.linspace(0, tfinal, npnts)
            interp_bench = bench(xs)[0]
            uncol = bench(xs)[1]
            interp_bench2 = np.append(interp_bench, np.array([0.0]))
            uncol2 = np.append(uncol, np.array([0.0]))
            xs2 = np.append(xs, np.array([tfinal + .0000001]))
            plt.plot(xs2, interp_bench2, "-k")
            plt.plot(-xs2, interp_bench2, "-k")
            if tfinal == 1:
                plt.plot(xs2, uncol2, "--k")
                plt.plot(-xs2, uncol2, "--k")
            
            show(file_path_string + f"/plane_IC_t_{tfinal}_benchmark")
            plt.show(block = False)
        elif source_name == "square_IC":
            source_type = np.array([0,1,0,0,0,0,0,0])
            x0 = 0.5
            bench = load_bench(source_type, tfinal, x0)
            xs = np.linspace(0, tfinal + x0, npnts)
            interp_bench = bench(xs)[0]
            uncol = bench(xs)[1]
            plt.plot(xs, interp_bench, "-k")
            plt.plot(-xs, interp_bench, "-k")
            if tfinal == 1:
                plt.plot(xs, uncol, "--k")
                plt.plot(-xs, uncol, "--k")
            show(file_path_string + f"/square_IC_t_{tfinal}_benchmark")
            plt.show()
        elif source_name == "square_source":
            source_type = np.array([0,0,1,0,0,0,0,0])
            x0 = 0.5
            bench = load_bench(source_type, tfinal, x0)
            xs = np.linspace(0, tfinal + x0, npnts)
            interp_bench = bench(xs)[0]
            uncol = bench(xs)[1]
            plt.plot(xs, interp_bench, "-k")
            plt.plot(-xs, interp_bench, "-k")
            if tfinal == 1 or tfinal == 5:
                plt.plot(xs, uncol, "--k")
                plt.plot(-xs, uncol, "--k")
            show(file_path_string + f"/square_source_t_{tfinal}_benchmark")
            plt.show()
        elif source_name == "gaussian_IC":
            source_type = np.array([0,0,0,1,0,0,0,0])
            x0 = 4
            bench = load_bench(source_type, tfinal, x0)
            if tfinal == 1:
                xs = np.linspace(0, 3.0, npnts)
            elif tfinal == 5:
                xs= np.linspace(0, 5.0, npnts)
            elif tfinal == 10:
                xs = np.linspace(0, 10.0, npnts)
            interp_bench = bench(xs)[0]
            uncol = bench(xs)[1]
            if tfinal == 1:
                plt.plot(xs, uncol, "--k")
                plt.plot(-xs, uncol, "--k")
            plt.plot(xs, interp_bench, "-k")
            plt.plot(-xs, interp_bench, "-k")
            show(file_path_string + f"/gaussian_IC_t_{tfinal}_benchmark")
        elif source_name == "gaussian_source":
            source_type = np.array([0,0,0,0,0,1,0,0])
            x0 = 4
            bench = load_bench(source_type, tfinal, x0)
            if tfinal == 1:
                xs = np.linspace(0,3.0,npnts)
            elif tfinal == 5:
                xs = np.linspace(0, 5.0,npnts)
            elif tfinal == 10:
                xs = np.linspace(0, 10.0,npnts)
            interp_bench = bench(xs)[0]
            uncol = bench(xs)[1]
            if tfinal == 1 or tfinal == 5:
                plt.plot(xs, uncol, "--k")
                plt.plot(-xs, uncol, "--k")
            plt.plot(xs, interp_bench, "-k")
            plt.plot(-xs, interp_bench, "-k")
            show(file_path_string + f"/gaussian_source_t_{tfinal}_benchmark")
        elif source_name == "MMS":
            source_type = np.array([0,0,0,0,1,0,0,0])
            x0 = 4
            bench = load_bench(source_type, tfinal, x0)
            xs = np.linspace(0, tfinal + x0, npnts)
            interp_bench = bench(xs)
            plt.plot(xs, interp_bench, "-k")
            plt.plot(-xs, interp_bench, "-k")
            show(file_path_string + f"MMS_t_{tfinal}_benchmark")
        elif source_name == "gaussian_IC_2D":
            plt.xlabel("r", fontsize = fntsize)
            source_type = np.array([0,0,0,0,0,0,1,0])
            x0 = 0.5
            xs = np.linspace(0, tfinal + 1/x0, npnts)
            bench = load_bench(source_type, tfinal, x0)
            interp_bench = bench(xs)[0]
            uncol = bench(xs)[1]
            if tfinal == 1:
                plt.plot(xs, uncol, "--k")
            plt.plot(xs, interp_bench, "-k")
            show(file_path_string + f"/gaussian_IC_2D_t_{tfinal}_benchmark")
        elif source_name == "line_source":
            plt.xlabel("r", fontsize = fntsize)
            source_type = np.array([0,0,0,0,0,0,0,1])
            x0 = 0.5
            xs = np.linspace(0, tfinal, npnts)
            bench = load_bench(source_type, tfinal, x0)
            interp_bench = bench(xs)[0]
            uncol = bench(xs)[1]
            if tfinal == 1 or tfinal == 5:
                plt.plot(xs, uncol, "--k")
            plt.plot(xs, interp_bench, "-k")
            show(file_path_string + f"/line_source_t_{tfinal}_benchmark")



        

        
    
        
        
        
        
        
        
            

    
    
    
    
    
    
    


