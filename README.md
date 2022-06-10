# transport_benchmarks
Solves and loads semi-analytic benchmark equations for the isotropic particle transport equation 

### Dependencies 
requires `numpy`, `scipy`, `h5py`, and `pathlib`

### Installation guide 
Download the `transport_benchmarks` folder. 

### Quick start guide
To access precomputed solutions, navigate to `transport_benchmarks` in the terminal and run the command `python -i run.py` to import necessary modules. Declare the solution x vector `xs = np.linspace(0,xmax,npts)` (only supports positive values since all of the solutions are symmetric). Call `load_func(source_name, tfinal, xs)` where `source_name` can be one of `["plane_IC", "square_IC", "square_source", "gaussian_IC", "MMS", "gaussian_source", "gaussian_IC_2D", "line_source"]` and `tfinal` is the evaluation time. This version of the package has benchmarks for `t=1,5,10` with `x0 = 0.5`, `sigma=0.5`, and `t0=5`. This will return two arrays: the uncollided solution and the collided solution evaluated at the input `xs` points. `x0` is the source width for finite sources and `t0` is the time the source is turned off. For initial conditions, such as `gaussian_IC`, `t0` has no meaning. For the Gaussian solutions, the standard deviation is frozen at 0.5. This will be changed in future releases. 

To plot, run `plotter.plot_all_benchmarks(tfinal)`. Results will be saved in the `plots/benchmark_plots` folder in a hpf5 file. 

### Creating solutions
To run more solutions, run `greens.do_all()`. Solutions at different times can be found by running functions from the `integrate_greens` script. For example, to calculate another time point for the square source, run `greens.square_source(tfinal, npnts, x0 = 0.5, t0 = 5)` where `npnts` is the number of evaluation points. The parameters `x0` and `t0` do not have meaning for all source configurations. The arguments for each source are:

``plane_IC(t, npnts)``

``square_IC(t, npnts, x0 = 0.5)``

``square_source(t, npnts, x0 = 0.5, t0 = 5)``
<<<<<<< HEAD
``gaussian_IC(t, npnts, sigma = 0.5)``
``gaussian_source(t, npnts, t0 = 5, sigma = 0.5)``
=======

``gaussian_IC(t, npnts)``

``gaussian_source(t, npnts, t0 = 5)``

>>>>>>> 487d2e980567ebd6c7cc8888396204e352d0af44
``gaussian_IC_2D(t, npnts)``

``line_source(t, npnts)``

### Different evaluation times
At this release, the integrator easily computes solutions at any time that the user specifies. However, the loading module `load_bench` only returns solutions at `t=1`, `t=5` and `t=10`. To load solutions at different times, first compute them with the integrator then modify the ``__init__`` argument of the `load_bench` class  in the `load_bench.py` script ``(def __init__(self, source_type, tfinal, x0, t_eval_times = [1, 5, 10]))`` to include the desired evaluation time. 

### Notes on the Gaussian pulse and source
Since the Gaussian sources are infinite, it is necessary to choose a solution interval width where the solution is practically zero at the edge. To this end, the integrator checks at what x location the value of the solution goes to a user specified tolerance and and prints "solution goes to (tol) at (x val)". The script `benchmarks.py` takes this information into account and uses a different `x0` for each evaluation time. For example, since the solution for the Gaussian pulse is smaller than `1e-10` by `x=3.16`, the code sets,

``           if t == 1:
                self.xs = np.linspace(0.0, 3.5, npnts)
``
