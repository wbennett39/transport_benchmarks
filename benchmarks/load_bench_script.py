from .load_bench import load_bench
import numpy as np


def selector_func(source):
    return_array = np.zeros(8)
    if source == "plane_IC":
        return_array[0] = 1
    elif source == "square_IC":
        return_array[1] = 1
    elif source == "square_source":
        return_array[2] = 1
    elif source == "gaussian_IC":
        return_array[3] = 1
    elif source == "MMS":
        return_array[4] = 1
    elif source == "gaussian_source":
        return_array[5] = 1
    elif source == "gaussian_IC_2D":
        return_array[6] = 1
    elif source == "line_source":
        return_array[7] = 1
    return return_array
    
def load_func(source, tfinal, xs):
    x0 = 0.5
    t0 = 5
    source_selector_array = selector_func(source)
    load_object = load_bench(source_selector_array, tfinal, x0)
    res = load_object(xs)
    return res
