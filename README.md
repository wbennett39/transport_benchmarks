# transport_benchmarks
Solves and loads semi-analytic benchmark equations for the isotropic particle transport equation 

### Dependencies 
requires `numpy`, `scipy`, and `pathlib`

### Installation guide 
Download the `transport_benchmarks` folder. 

### Quick start guide
To access precomputed solutions, navigate to `transport_benchmarks` in the terminal and run the command `python -i run.py` to import necessary modules. Declare the solution x vector `xs = np.linspace(0,xmax)` (only supports positive values since all of the solutions are symmetric). Call `load_func(source_name, tfinal, xs)` where `source_name` can be one of `["plane_IC", "square_IC", "square_source", "gaussian_IC", "MMS", "gaussian_source", "gaussian_IC_2D", "line_source"]` and `tfinal` is the evaluation time. This version of the package has benchmarks for `t=1,5,10` with `x0 = 0.5` and `t0=5`. At this point, it is difficult to change the values of `x0` and `t0` without directly editing the collided/uncollided solutions in the `benchmark_functions` script. `x0` is the source width for finite sources and `t0` is the time the source is turned off. For initial conditions, such as `gaussian_IC`, `t0` has no meaning. For the Gaussian solutions, the standard deviation is frozen at 0.5. This will be changed in future releases. 

To plot, run `plotter.plot_all_benchmarks(tfinal)`. Results will be saved in the `plots/benchmark_plots` folder.  
