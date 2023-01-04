#!/usr/bin/env python
# coding: utf-8

# In[31]:


from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
import numpy as np
import math, sys, re, statistics, os

from matplotlib.pyplot import figure

# 25 ns / gas
GAS_RATE = 25.0

evm_be_color = "#ff7878"
go_arith_be_color = "#E24C4C"
evm_le_color = "#ffff78"
go_arith_le_color = "#a0a011"
model_color = "black"
asm384_color = "#61CD61"
asm384_arith_color = "#1EA41E"
setmod_color = "purple"

def group_data(lines: [str]) -> [str]:
    result = {}

    for line in lines:
        parts = line.split(',')
        if parts[0] in result:
            result[parts[0]].append((int(parts[1]), int(parts[2])))
        else:
            result[parts[0]] = [(int(parts[1]), int(parts[2]))]

    return result

def insert_bench(benches_map, bench_preset, bench_op, bench_limbs, time):
    if not bench_op in benches_map:
        benches_map[bench_op] = {}  
    if not bench_preset in benches_map[bench_op]:
        benches_map[bench_op][bench_preset] = {}
    if not bench_limbs in benches_map[bench_op][bench_preset]:
        benches_map[bench_op][bench_preset][bench_limbs] = {'data':[]}

    benches_map[bench_op][bench_preset][bench_limbs]['data'].append(time)

def read_evm_arith_benchmarks(file_name):
    input_lines = []
    with open(file_name) as f:
        input_lines = f.readlines()

    benches_map = {}
    for line in input_lines[1:]:
        line_parts = line.strip('\n').split(',')
        bench_op = line_parts[0]
        if bench_op == "MULMONTX":
            bench_op = "mulmont"
        elif bench_op == "ADDMODX":
            bench_op = "addmod"
        elif bench_op == "SUBMODX":
            bench_op = "submod"
        else:
            raise Exception("unexpected")

        bench_limbs = int(line_parts[1])
        bench_time = float(line_parts[2])
        insert_bench(benches_map, "non-unrolled", bench_op, bench_limbs, bench_time)
        foo = 'bar'

    for bench_op in benches_map.keys():
        for bench_preset in benches_map[bench_op].keys():
            for bench_limbs in benches_map[bench_op][bench_preset].keys():
                item = benches_map[bench_op][bench_preset][bench_limbs]
                item['stddev'] = statistics.stdev(item['data'])
                item['mean'] = statistics.mean(item['data'])
                benches_map[bench_op][bench_preset][bench_limbs] = item # TODO is this necessary in python?
    return benches_map

#input_lines = sys.stdin.readlines()
def read_go_arith_benchmarks(file_name):
    input_lines = []
    with open(file_name) as f:
        input_lines = f.readlines()

    benches_map = {}

    for line in input_lines[4:-2]:
        parts = [elem for elem in line[13:].split(' ') if elem and elem != '\t']

        bench_full = parts[0][:-2]
        #bench_full = re.search(r'(.*)?(#.*-.*$)', parts[0]).groups()[0]
        if '#' in parts[0] and parts[0].index('#'):
            bench_full = parts[0].split('#')[0]

        bench_preset = bench_full.split('_')[0]
        bench_op = bench_full.split('_')[1]
        bench_limbs = int(bench_full.split('_')[2])
        assert bench_limbs % 64 == 0, "invalid bench limb bit width"
        bench_limbs //= 64
        time = float(parts[2])
        unit = re.search(r'(.*)\/', parts[3]).groups()[0]

        if unit != 'ns':
            raise Exception("expected ns got {}".format(unit))

        insert_bench(benches_map, bench_preset, bench_op, bench_limbs, time)

    for bench_op in benches_map.keys():
        for bench_preset in benches_map[bench_op].keys():
            for bench_limbs in benches_map[bench_op][bench_preset].keys():
                item = benches_map[bench_op][bench_preset][bench_limbs]
                item['stddev'] = statistics.stdev(item['data'])
                item['mean'] = statistics.mean(item['data'])
                benches_map[bench_op][bench_preset][bench_limbs] = item # TODO is this necessary in python?
            #print("{},{},{},{},{}".format(bench_preset, bench_op, bench_limbs, item['mean'], item['stddev']))
    return benches_map

evm_le_benchmarks = read_evm_arith_benchmarks("benchmarks-results/geth-evm-little-endian.csv")
evm_le_asm_384bit_benchmarks = read_evm_arith_benchmarks("benchmarks-results/geth-evm-asm384.csv")
evm_be_benchmarks = read_evm_arith_benchmarks("benchmarks-results/geth-evm-eip-5843.csv")

go_arith_benchmarks = read_go_arith_benchmarks("benchmarks-results/go-arith-benchmarks-le.txt")
go_arith_benchmarks_be = read_go_arith_benchmarks("benchmarks-results/go-arith-benchmarks-eip5843.txt")
go_arith_benchmarks_asm384 = read_go_arith_benchmarks("benchmarks-results/go-arith-benchmarks-asm384.txt")

def format_bench_data_for_graphing(x_range, bench_data, label, color):
    x_vals = []
    y_vals = []
    y_errs = []

    for i, limb_val in enumerate(bench_data.keys()):
        if i + 1 < x_range[0] or i + 1 > x_range[1]:
            continue

        x_val = limb_val
        y_val = bench_data[limb_val]['mean']
        y_err = bench_data[limb_val]['stddev']
        x_vals.append(x_val)
        y_vals.append(y_val)
        y_errs.append(y_err)
    
    return (x_range, x_vals, y_vals, y_errs, color, label, 'o')

def scatterplot_ns_data(fname: str, name: str, graph_range, annotates: [bool], markers: [str], args):
    assert len(annotates) == len(args)
    assert len(markers) == len(args)

    stripped_args = []
    for arg in args:
        stripped_xs = [x_val for x_val in arg[1] if x_val >= graph_range[0] and x_val <= graph_range[1]]
        stripped_ys = []
        stripped_yerrs = []
        for i, y_val in enumerate(arg[2]):
            if arg[1][i] in stripped_xs:
                stripped_ys.append(y_val)
        for i, y_err_val in enumerate(arg[3]):
            if arg[1][i] in stripped_xs:
                stripped_yerrs.append(y_err_val)
        stripped_args.append((arg[0], stripped_xs, stripped_ys, stripped_yerrs, arg[4], arg[5], arg[6]))

    len_x = len(stripped_args[0][1])

    x_min_all = min([min([v for v in d[1]]) for d in stripped_args])
    x_max_all = max([max([v for v in d[1]]) for d in stripped_args])
    y_min_all = min([min([v for v in d[2]]) for d in stripped_args])
    y_max_all = max([max([v for v in d[2]]) for d in stripped_args])
    span_x = x_max_all - x_min_all
    span_y = y_max_all - y_min_all

    plt.rcParams["figure.figsize"] = (20, 10)
    fig, ax = plt.subplots()
    plt.ylim(0, y_min_all + int(span_y * 1.2))
    plt.xlim(0, x_min_all + int(span_x * 1.2))
    plt.xticks(stripped_args[0][1])

    legend_lines = []
    legend_labels = []

    for i, (x_range, x_vals, y_vals, y_errs, color, label, marker) in enumerate(stripped_args):
        assert len(x_vals) == len(y_vals)

        plt.xlabel("input size (bits)")
        plt.ylabel("runtime (ns)")
        
        if len(y_errs) != 0:
            assert len(y_vals) == len(y_errs)
            for x, y, y_err in zip(x_vals, y_vals, y_errs):
                # TODO this should not be needed
                if x < x_range[0] or x > x_range[1]:
                    continue

                if annotates[i]:
                    ax.annotate(y, (float(x) + 0.2, float(y)))
                
                ax.errorbar(x=x, y=y, xerr=0.0, yerr=y_err, fmt=markers[i], color=color)
            ax.plot(x_vals, y_vals, markers[i], color=color, label=label)
        else:
           if annotates[i]:
               for x, y in zip(x_vals, y_vals):
                   # cheap hack because we are only executing this spot for
                   # annotating the model
                   ax.annotate(round(float(y) / GAS_RATE), (float(x) + 0.2, float(y)))
           ax.plot(x_vals, y_vals, markers[i], color=color, label=label)
        
        legend_lines.append(Line2D([0], [0], color=color, lw=4))
        legend_labels.append(label)

    x_axis_labels = [str(i*64) for i in range(1, len_x+1)]
    x_axis_pos = np.arange(1,len_x + 1)
    plt.xticks(x_axis_pos, x_axis_labels, color='black', fontsize='10')
    ax.legend(legend_lines, legend_labels, loc="upper left")
    #ax.set(title=name)
    plt.savefig(fname)

def fit_quadratic(input_range, dataset, intercept_min):
    xs = []
    ys = []

    for x in dataset.keys():
        if x < input_range[0] or x > input_range[1]:
            continue

        xs.append(x)
        ys.append(dataset[x]['mean'])

    eqn = np.polyfit(np.array(xs), np.array(ys), 2)
    eqn = [round(val, 2) for val in eqn]

    if intercept_min:
        # y = a * x ** 2  + b * x + c
        # want ys[0] = a * (xs[0]) ** 2 + b * xs[0] + c
        # c = ys[0] - (a * xs[0] ** 2 + b * xs[0]) if it is positive
        new_intercept = ys[0] - eqn[0] * xs[0] ** 2 + eqn[1] * xs[0]
        if new_intercept > 0:
            eqn[2] = new_intercept
        
    return eqn

def fit_linear(input_range, dataset, intercept_min):
    xs = []
    ys = []

    for x in dataset.keys():
        if x < input_range[0] or x > input_range[1]:
            continue

        xs.append(x)
        ys.append(dataset[x]['mean'])

    eqn = np.polyfit(np.array(xs), np.array(ys), 1)
    eqn = [round(val, 2) for val in eqn]
    
    # make sure the line of fit goes thru the first limb
    # TODO make this optional (I think addmod/submod needed it to get proper models)
    # new_y_intercept_addmod = graphing_dataset[0][1] - abs(abs(eqn[0]) - abs(eqn[1]))
    # eqn[1] = new_y_intercept_addmod

    if intercept_min:
        # y = mx + b
        # y = eqn[0] * x + eqn[1]
        # ys[0] = eqn[0] * xs[0] + b
        # b = ys[0] - eqn[0] * xs[0] 
        # if b > 0:
            # eqn = b
        intercept = ys[0] - eqn[0] * xs[0]
        if intercept > 0:
            eqn[1] = intercept

    return eqn

def stitch_model(model1, model2, cutoff: int):
    result = []
    for i, (x_val, y_val) in enumerate(model1):
        if i >= cutoff:
            break

        result.append((x_val, y_val))

    for i, (x_val, y_val) in enumerate(model2):
        result.append((x_val, y_val))

    return result

def stitch_data(data1, data2, cutoff: int, name, color):
    graph_range = (1,10000000) # TODO allow configurable?
    xs = []
    ys = []
    y_errs = []

    for key in data1.keys():
        if key > cutoff:
            break
        xs.append(key)
        ys.append(data1[key]['mean'])
        y_errs.append(data1[key]['stddev'])
    if data2 != None:
        for key in data2.keys():
            if key <= cutoff:
                continue
            xs.append(key)
            ys.append(data2[key]['mean'])
            y_errs.append(data2[key]['stddev'])

    return (graph_range, xs, ys, y_errs, color, name, 'o')

def prep_data_for_graphing(data, name, color):
    xs = []
    ys = []
    y_errs = []

    for key in data.keys():
        xs.append(key)
        ys.append(data[key]['mean'])
        y_errs.append(data[key]['stddev'])

    return ((1, 16), xs, ys, y_errs, color, name, 'o')

def prep_models_for_graphing(models, name, xs):
    ys = []
    x_idx = 0

    for model, range_max in models:
        while x_idx < len(xs) and xs[x_idx] <= range_max:
            y = round(eval_model(model, xs[x_idx]))
            if y == 0:
                y = 1
            ys.append(y * GAS_RATE)
            x_idx += 1
    
    return ((min(xs), max(xs)), xs, ys, [], 'black', name, 'o')

# def prep_model_data_for_graphing(model, name, plot_range):
#     return (plot_range, [item[0] for item in model if item[0] <= plot_range[1] and item[0]  >= plot_range[0]], [item[1] for item in model if item[0] <= plot_range[1] and item[0]  >= plot_range[0]], [], 'black', name, 'o')

def strip_graphing_data(x_range, data):
    val = list(data)
    x_vals = []
    y_vals = []
    y_errs = []
    if len(y_errs) == 0:
        for x, y in zip(data[1], data[2]):
            if x < x_range[0] or x > x_range[1]:
                continue

            x_vals.append(x)
            y_vals.append(y)
    else:
        for x, y, y_err in zip(data[1], data[2]):
            if x < x_range[0] or x > x_range[1]:
                continue

            x_vals.append(x)
            y_vals.append(y)
            y_errs.append(y_err)
        val[3] = y_errs

    val[0] = x_range
    val[1] = x_vals
    val[2] = y_vals
    return tuple(val)

def eval_model(model: [int], x: int) -> int:
    result = 0
    for i, coef in enumerate(model):
        if i != len(model) - 1:
            result += coef * (x ** (len(model) - 1 - i))
        else:
            result += coef

    return result


def format_model_eqn_for_graphing(model: [int], xs: [int]):
    result = []

    for x in xs:
        result.append(eval_model(model, x))

    return result

# originals (fit from the data)
# setmod_eqn = [62.71, 2679.49]
# mulmont_eqn_lo = [2.28, -0.72, 5.8782000000000005]
# mulmont_eqn_hi = [0.01, 246.97, 17683.52]
# addmod_eqn = [4.61, 3.6100000000000003]

# TODO remove this
fast_mulmont_cutoff = 49

setmod_eqn = [3.8, 75.0]
mulmont_eqn_low = [0.1, 0, 0.7]
# TODO remove this (mulmont/setmod will have one gas model)
mulmont_eqn_hi = [0.0004, 9.88, -268.0]
addmod_eqn = [0.2, 0.6]

mulmont_go_arith_be = prep_data_for_graphing(go_arith_benchmarks_be['mulmont']['non-unrolled'], "eip-5843 - arithmetic", go_arith_be_color)
addmod_go_arith_be = prep_data_for_graphing(go_arith_benchmarks_be['addmod']['non-unrolled'], "eip-5843 - arithmetic", go_arith_be_color)
submod_go_arith_be = prep_data_for_graphing(go_arith_benchmarks_be['submod']['non-unrolled'], "eip-5843 - arithmetic", go_arith_be_color)

mulmont_evm_le = prep_data_for_graphing(evm_le_benchmarks['mulmont']['non-unrolled'], "little-endian - evm", evm_le_color)
mulmont_evm_be = prep_data_for_graphing(evm_be_benchmarks['mulmont']['non-unrolled'], "eip-5843 - evm", evm_be_color)
mulmont_evm_le_asm_384bit = prep_data_for_graphing(evm_le_asm_384bit_benchmarks['mulmont']['non-unrolled'], "asm384 - evm", asm384_color)

addmod_evm_le = prep_data_for_graphing(evm_le_benchmarks['addmod']['non-unrolled'], "little-endian - evm", evm_le_color)
addmod_evm_be = prep_data_for_graphing(evm_be_benchmarks['addmod']['non-unrolled'], "eip-5843 - evm", evm_be_color)
addmod_evm_le_asm_384bit = prep_data_for_graphing(evm_le_asm_384bit_benchmarks['addmod']['non-unrolled'], "asm384 - evm", asm384_color)

submod_evm_le = prep_data_for_graphing(evm_le_benchmarks['submod']['non-unrolled'], "little-endian - evm", evm_le_color)
submod_evm_be = prep_data_for_graphing(evm_be_benchmarks['submod']['non-unrolled'], "eip-5843 - evm", evm_be_color)
submod_evm_le_asm_384bit = prep_data_for_graphing(evm_le_asm_384bit_benchmarks['submod']['non-unrolled'], "asm384 - evm", asm384_color)

# fast_mulmont_cutoff = 49
mulmont_benches = go_arith_benchmarks['mulmont']
mulmont_non_unrolled_data = format_bench_data_for_graphing((1, 64), go_arith_benchmarks['mulmont']['non-unrolled'], 'mulmont arithmetic', go_arith_le_color)

addmod_non_unrolled_data = format_bench_data_for_graphing((1, 64), go_arith_benchmarks['addmod']['non-unrolled'], 'little-endian - arithmetic', go_arith_le_color)
submod_non_unrolled_data = format_bench_data_for_graphing((1, 64), go_arith_benchmarks['submod']['non-unrolled'], 'little-endian - arithmetic', go_arith_le_color)

# setmod_non_unrolled_data = format_bench_data_for_graphing((1, 64), go_arith_benchmarks['setmod']['non-unrolled'], 'setmod-non-unrolled', 'red')
setmod_arith_be = prep_data_for_graphing(go_arith_benchmarks_be['setmod']['non-unrolled'], "setmod arithmetic - big-endian limbs", setmod_color)

mulmont_arith_asm384 = prep_data_for_graphing(go_arith_benchmarks_asm384['mulmont']['asm384'], 'asm384 - arithmetic', asm384_arith_color)
addmod_arith_asm384 = prep_data_for_graphing(go_arith_benchmarks_asm384['addmod']['asm384'], 'asm384 - arithmetic', asm384_arith_color)
submod_arith_asm384 = prep_data_for_graphing(go_arith_benchmarks_asm384['submod']['asm384'], 'asm384 - arithmetic', asm384_arith_color)

#scatterplot_ns_data("charts/mulmont_generic.png", "MULMONTMAX Generic Benchmarks", False, [mulmont_non_unrolled_data, mulmont_generic_data])
#scatterplot_ns_data('charts/setmod.png', 'SETMOD Benchmarks', False, [setmod_non_unrolled_data, setmod_generic_data])
#scatterplot_ns_data('charts/addmod.png', 'ADDMOD Benchmarks', False, [addmod_non_unrolled_data])
#scatterplot_ns_data('charts/submod.png', 'SUBMOD Benchmarks', False, [submod_non_unrolled_data])

#mulmont_non_unrolled_small_limbs_data = format_bench_data_for_graphing((1, 12), go_arith_benchmarks['mulmont']['non-unrolled'], 'mulmont', 'red')
#setmod_non_unrolled_small_limbs_data = format_bench_data_for_graphing((1, 12), go_arith_benchmarks['setmod']['non-unrolled'], 'setmod', 'green')

# linear cost for setmod up to cutoff
# setmod_low_eqn, setmod_low_cost = fit_linear((1, 32), (1, fast_mulmont_cutoff), setmod_non_unrolled_data, False)
# setmod_model = stitch_model(setmod_low_cost, setmod_high_cost, fast_mulmont_cutoff)

#mulmont_evmmax = stitch_data(go_arith_benchmarks['mulmont']['non-unrolled'], go_arith_benchmarks['mulmont']['generic'], fast_mulmont_cutoff)

# setmod_evmmax = stitch_data(go_arith_benchmarks['setmod']['non-unrolled'], go_arith_benchmarks['setmod']['generic'], fast_mulmont_cutoff)

mulmont_evmmax_low = go_arith_benchmarks['mulmont']['non-unrolled']

benches_xs = list(range(1, 16))

#mulmont_model = stitch_model(mulmont_cost_low, mulmont_cost_hi, fast_mulmont_cutoff)
import pdb; pdb.set_trace()
mulmont_model = prep_models_for_graphing([(mulmont_eqn_low, 100000)], "mulmontx model", benches_xs)

mulmont_go_arith_le = stitch_data(go_arith_benchmarks['mulmont']['non-unrolled'], None, fast_mulmont_cutoff, "mulmont arithmetic (little-endian)", go_arith_le_color)
#scatterplot_ns_data("charts/mulmontx_all.png", "MULMONTX Benchmarks", (1, 100000), [False, False], ["o", "-"], [mulmont_evmmax, mulmont_model])
# scatterplot_ns_data("charts/mulmontx_med.png", "MULMONTX Benchmarks", (1, 64), [False, False], ["o", "o"], [mulmont_evmmax, mulmont_model])
scatterplot_ns_data("charts/mulmontx_low.png", "MULMONTX Benchmarks with Gas Model Labeled", (1, 8), [False, False, True, False, False, False, False], ["o", "o", "o", "o", "o", "o", "o"], [mulmont_go_arith_be, mulmont_evm_be,  mulmont_model, mulmont_go_arith_le, mulmont_evm_le, mulmont_arith_asm384, mulmont_evm_le_asm_384bit])

scatterplot_ns_data("charts/mulmontx_all.png", "MULMONTX Benchmarks with Gas Model Labeled", (1, 16), [False, False, True, False, False, False, False], ["o", "o", "o", "o", "o", "o", "o"], [mulmont_go_arith_be, mulmont_evm_be,  mulmont_model, mulmont_go_arith_le, mulmont_evm_le, mulmont_arith_asm384, mulmont_evm_le_asm_384bit])

addmod_model = prep_models_for_graphing([(addmod_eqn, 100000)], 'addmod model', benches_xs)

# TODO use mix of impls
# TODO don't use stitch_data to put it in graphing format
addmod_evmmax = stitch_data(go_arith_benchmarks['addmod']['non-unrolled'], None, 100000, "little-endian - arithmetic", go_arith_le_color)
submod_evmmax = stitch_data(go_arith_benchmarks['submod']['non-unrolled'], None, 100000, "little-endian - arithmetic", go_arith_le_color)

scatterplot_ns_data("charts/addmodx_all.png", "ADDMODX Benchmarks with Gas Model Labelled", (1, 16), [False, False, True, False, False, False, False], ["o", "o", 'o', 'o', 'o', 'o', "o"], [mulmont_go_arith_be, mulmont_evm_be,  mulmont_model, mulmont_go_arith_le, mulmont_evm_le, mulmont_arith_asm384, mulmont_evm_le_asm_384bit])
# scatterplot_ns_data("charts/addmodx_med.png", "ADDMODX Benchmarks", (1, 64), [False, False], ["o", 'o'], [addmod_evmmax, addmod_model])
# scatterplot_ns_data("charts/addmodx_all.png", "ADDMODX Benchmarks", (1, 100000), [False, False], ["o", '-'], [addmod_evmmax, addmod_model])

scatterplot_ns_data("charts/submodx_all.png", "SUBMODX Benchmarks with Gas Model Labelled", (1, 16), [False, False, False, False, True, False, False], ["o", 'o', 'o', 'o', 'o', 'o', "o"], [mulmont_go_arith_be, mulmont_evm_be,  mulmont_model, mulmont_go_arith_le, mulmont_evm_le, mulmont_arith_asm384, mulmont_evm_le_asm_384bit])
# scatterplot_ns_data("charts/submodx_med.png", "SUBMODX Benchmarks", (1, 64), [False, False], ["o", 'o'], [submod_evmmax, addmod_model])
# scatterplot_ns_data("charts/submodx_all.png", "SUBMODX Benchmarks", (1, 100000), [False, False], ["o", '-'], [submod_evmmax, addmod_model])

setmod_evmmax = stitch_data(go_arith_benchmarks['setmod']['non-unrolled'], None, 100000, "eip-5843 - setmod", "purple")
setmod_model = prep_models_for_graphing([(setmod_eqn, 100000)], 'setmod model', benches_xs)

# scatterplot_ns_data("charts/setmodx_all.png", "SETMODMAX Benchmarks", (1, 100000), [False, False], ["-", "o"], [setmod_model, setmod_evmmax])
# scatterplot_ns_data("charts/setmodx_med.png", "SETMODMAX Benchmarks", (1, 64), [False, False], ["-", "o"], [setmod_model, setmod_evmmax])
scatterplot_ns_data("charts/setmodx_all.png", "SETMODX Benchmarks with Gas Model Labeled", (1, 16), [True, False], ["o", "o"], [setmod_model, setmod_arith_be])
