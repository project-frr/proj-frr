from math import sqrt, exp
import os, random
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline

np.random.seed(1)
random.seed(1)

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def map_data(data_list, available_item_set):
    item_to_int = {item: idx for idx, item in enumerate(sorted(available_item_set))}
    dummy_item_value = len(item_to_int)
    item_to_int['dummy'] = dummy_item_value
    
    int_to_item_dict = {}
    for key in item_to_int.keys():
        int_to_item_dict[item_to_int[key]] = key
    mapped_datalist = [item_to_int[item] if item in item_to_int else item_to_int['dummy'] for item in data_list]

    return np.array(mapped_datalist), int_to_item_dict

def get_cubic_hermite_spline_approximated_intervals(check_points, intervals, intervals_middle_idx_dict, first_derivatives\
    , histogram, first_derivatives_left, first_derivatives_right, epsilon, histogram_sw_woS=None):
    
    approximated_intervals = []
    approximated_segments = []
    segment_to_interval_map = {}
    for interval in intervals:
        interval_idx_l, interval_idx_r = interval[0], interval[1]
        first_derivative_l, first_derivative_r = first_derivatives_right[interval_idx_l], first_derivatives_left[interval_idx_r]
        middle_idx_list, middle_1st_derivatives = [], []
        
        if intervals_middle_idx_dict[(interval_idx_l, interval_idx_r)] != None:
            middle_idx_list = intervals_middle_idx_dict[(interval_idx_l, interval_idx_r)]
            for middle_idx in middle_idx_list:
                middle_1st_derivatives.append(first_derivatives[middle_idx])
        
        if len(middle_idx_list) > 0:
            x = [interval_idx_l] + middle_idx_list + [interval_idx_r]
            y = histogram[x].tolist()
            dy = [first_derivative_l] + middle_1st_derivatives + [first_derivative_r]
        else:
            x = [interval_idx_l] + [interval_idx_r]
            y = histogram[x].tolist()
            dy = [first_derivative_l] + [first_derivative_r]
        
        x_interp_new = []
        for idx, val in enumerate(x[:-1]):
            val_l, val_r = val, x[idx+1]
            approximated_segments.append((val_l, val_r))
            x_interp_new = x_interp_new + list(np.linspace(val_l, val_r, 200))

        err_min = 1e10
        for dy_idx, dy_val in enumerate(dy):
            for dy_val_try in np.arange(dy_val-0.5*exp(-epsilon), dy_val+0.5*exp(-epsilon), 100):
                err = 0
                dy_copy = dy
                dy_copy[dy_idx] = dy_val_try
                for x_idx, x_val in enumerate(x[:-1]):
                    err_i, _, _, _ = cal_error((x_val, x[x_idx+1]), (y[x_idx],y[x_idx+1]), (dy_copy[x_idx], dy_copy[x_idx+1]), histogram, histogram_sw_woS)
                    err += err_i
                if err < err_min:
                    err_min = err
                    dy[dy_idx] = dy_val_try
        cubic_hermite_spline = CubicHermiteSpline(x, y, dy)
        coefficients = cubic_hermite_spline.c
        y_interp_new = cubic_hermite_spline(x_interp_new)
        approximated_intervals.append((x_interp_new, y_interp_new))
        for idx, val in enumerate(x[:-1]):
            val_l, val_r = val, x[idx+1]
            segment_to_interval_map[(val_l, val_r)] = {
                'interval': (x_interp_new, y_interp_new),
                'coefficients': coefficients[:, idx] 
            }
        
    return approximated_segments, approximated_intervals, coefficients, segment_to_interval_map


def cal_error(x, y, dy, histogram_sw, histogram_sw_woS):
    cubic_hermite_spline = CubicHermiteSpline(x, y, dy)
    coefficients = cubic_hermite_spline.c
    x_interp_new = list(np.linspace(x[0], x[1], 200))
    y_interp_new = cubic_hermite_spline(x_interp_new)
    approximated_interval_integral = cubic_integral_(coefficients=np.array(coefficients).flatten(), x0=x[0], x1=x[1], calerr=True)
    noisy_accu_fre = sum(histogram_sw[x[0]:x[1]])
    noisy_accu_fre_woS = sum(histogram_sw_woS[x[0]:x[1]])
    err=0
    err_size = 1
    for step in range(1, x[1]-x[0]):
        for xx in range(x[0], x[1] - step + 1):
            approximated_interval_integral_xx = cubic_integral_(coefficients=np.array(coefficients).flatten(), x0=xx, x1=xx+step, calerr=True)
            if approximated_interval_integral>0 and noisy_accu_fre>0 and noisy_accu_fre_woS>0:
                errxx_S = sqrt(pow(sum(histogram_sw[xx:xx+step])/noisy_accu_fre - approximated_interval_integral_xx/approximated_interval_integral, 2))
                errxx_woS = sqrt(pow(sum(histogram_sw_woS[xx:xx+step])/noisy_accu_fre_woS - approximated_interval_integral_xx/approximated_interval_integral, 2))
                errxx = errxx_S * (1/2) + errxx_woS * (1/2)
                err += errxx
            err_size += 1
    if err_size > 1:
        err_size -= 1
    err = err/err_size
    
    return err, coefficients, x_interp_new, y_interp_new

def get_checkpoints(histogram, int_to_item_dict):
    first_derivatives, second_derivatives = compute_first_and_second_derivatives_central(histogram=histogram)
    first_derivatives_left, first_derivatives_right = compute_first_and_second_derivatives_left_and_right(histogram=histogram)
    zero_1st_derivatives_points_idx_set, zero_2nd_derivatives_points_idx_set = set(), set()
    for idx, derivative in enumerate(first_derivatives):
        previous_idx = idx - 1
        next_idx = idx + 1
        next_val = -1
        previous_val = -1
        if next_idx <= len(first_derivatives) - 1: next_val = histogram[next_idx]
        if previous_idx >= 0: previous_val = histogram[previous_idx]
        if derivative <= 0 + delta and derivative >= 0 - delta and previous_val >= histogram[idx] and next_val >= histogram[idx]:
            zero_1st_derivatives_points_idx_set.add(idx)
    for idx, derivative in enumerate(second_derivatives):
        previous_idx = idx - 1
        next_idx = idx + 1
        next_val = -1
        previous_val = -1
        if next_idx <= len(first_derivatives) - 1: next_val = second_derivatives[next_idx]
        if previous_idx >= 0: previous_val = second_derivatives[previous_idx]
        if derivative <= 0 + delta_2 and derivative >= 0 - delta_2 and previous_val * next_val < 0:
            zero_2nd_derivatives_points_idx_set.add(idx)
    
    zero_1st_derivatives_points_idx_list = sorted(zero_1st_derivatives_points_idx_set)
    checkpoints_idx_sorted = sorted(zero_1st_derivatives_points_idx_set.union(zero_2nd_derivatives_points_idx_set))
    if len(zero_1st_derivatives_points_idx_list) == 0:
        zero_1st_derivatives_points_idx_list.append(random.randint(0,len(first_derivatives)-1))
        zero_1st_derivatives_points_idx_list.append(random.randint(0,len(first_derivatives)-1))
        zero_1st_derivatives_points_idx_list = sorted(zero_1st_derivatives_points_idx_list)
    
    intervals = []
    intervals_middle_idx_dict = {}   
    zero_2nd_derivatives_points_idx_set_np = np.array(list(zero_2nd_derivatives_points_idx_set))
    if min(int_to_item_dict.keys()) < zero_1st_derivatives_points_idx_list[0]:
        intervals.append((min(int_to_item_dict.keys()), zero_1st_derivatives_points_idx_list[0]))
        intervals_middle_idx_dict[(min(int_to_item_dict.keys()), zero_1st_derivatives_points_idx_list[0])] \
            = get_intervals_middle_idx(MIN=min(int_to_item_dict.keys()), MAX=zero_1st_derivatives_points_idx_list[0], \
                zero_2nd_derivatives_points_idx_set_np=zero_2nd_derivatives_points_idx_set_np, \
                    first_derivatives=first_derivatives, histogram=histogram)
    for i, checkpoint_idx in enumerate(zero_1st_derivatives_points_idx_list[:-1]):
        intervals.append((checkpoint_idx, zero_1st_derivatives_points_idx_list[i+1]))
        intervals_middle_idx_dict[(checkpoint_idx, zero_1st_derivatives_points_idx_list[i+1])] \
            = get_intervals_middle_idx(MIN=checkpoint_idx, MAX=zero_1st_derivatives_points_idx_list[i+1], \
                zero_2nd_derivatives_points_idx_set_np=zero_2nd_derivatives_points_idx_set_np, \
                    first_derivatives=first_derivatives, histogram=histogram)
    if max(int_to_item_dict.keys()) - 1 > zero_1st_derivatives_points_idx_list[-1]:
        intervals.append((zero_1st_derivatives_points_idx_list[-1], max(int_to_item_dict.keys()) - 1))
        intervals_middle_idx_dict[(zero_1st_derivatives_points_idx_list[-1], max(int_to_item_dict.keys()) - 1)] \
            = get_intervals_middle_idx(MIN=zero_1st_derivatives_points_idx_list[-1], MAX=max(int_to_item_dict.keys()) - 1, \
                zero_2nd_derivatives_points_idx_set_np=zero_2nd_derivatives_points_idx_set_np, \
                    first_derivatives=first_derivatives, histogram=histogram)
    
    return zero_1st_derivatives_points_idx_list, intervals, intervals_middle_idx_dict, first_derivatives, first_derivatives_left, first_derivatives_right


def get_intervals_middle_idx(MIN, MAX, zero_2nd_derivatives_points_idx_set_np, first_derivatives, histogram):
    result = zero_2nd_derivatives_points_idx_set_np[(zero_2nd_derivatives_points_idx_set_np > MIN) \
            & (zero_2nd_derivatives_points_idx_set_np < MAX)]
    result_histogram = histogram[MIN:MAX]
    val = None
    if result.size != 0:
        val = list(sorted(result))
    
    return val


def compute_first_and_second_derivatives_central(histogram):
    histogram = np.array(histogram)
    n = len(histogram)
    first_derivative = np.zeros(n)
    second_derivative = np.zeros(n)
    for i in range(1, n-1):
        if (histogram[i] - histogram[i-1]) * (histogram[i+1] - histogram[i]) <= 0:
            first_derivative[i] = 0
        else:
            first_derivative[i] = (histogram[i+1] - histogram[i-1]) / 2

    first_derivative[0] = ((histogram[1] - histogram[0]) / 2 + histogram[1] - histogram[0]) / 2
    first_derivative[-1] = ((histogram[-1] - histogram[-2]) / 2 + histogram[-1] - histogram[-2]) / 2

    for i in range(1, n-1):
        second_derivative[i] = histogram[i+1] - 2*histogram[i] + histogram[i-1]

    second_derivative[0] = histogram[2] - 2*histogram[1] + histogram[0]
    second_derivative[-1] = histogram[-3] - 2*histogram[-2] + histogram[-1]

    return first_derivative, second_derivative


def compute_first_and_second_derivatives_left_and_right(histogram):
    histogram = np.array(histogram)

    histogram_new = np.copy(histogram)
    first_derivatives_left = np.diff(histogram_new)  
    first_derivatives_left = np.array([0] + list(first_derivatives_left))
    
    first_derivatives_right = np.diff(histogram)  
    first_derivatives_right = np.append(first_derivatives_right, 0)  


    return first_derivatives_left, first_derivatives_right


def cubic_integral_(coefficients, x0, x1, inter=False, calerr=False):

    coefficients = np.array(coefficients).flatten()
    a3, a2, a1, a0 = coefficients
    def cubic_function(x):
        return a3 * x**3 + a2 * x**2 + a1 * x + a0

    roots = np.roots([a3, a2, a1, a0])

    valid_roots = [r for r in roots if np.isreal(r) and x0 < r < x1]

    valid_roots = np.real(valid_roots)

    points = np.sort(np.concatenate(([x0, x1], valid_roots)))

    total_area = 0

    for i in range(len(points) - 1):
        xi, xi_plus_1 = points[i], points[i + 1]

        integral = (a3 / 4 * (xi_plus_1 - xi)**4 +
                    a2 / 3 * (xi_plus_1 - xi)**3 +
                    a1 / 2 * (xi_plus_1 - xi)**2 +
                    a0 * (xi_plus_1 - xi))

        if cubic_function((xi + xi_plus_1) / 2) >= 0:
            total_area += max(integral, 0)
        elif cubic_function(xi) >= 0 or cubic_function(xi_plus_1) >= 0:
            total_area += max(integral, 0)
    assert total_area >= 0

    return total_area
