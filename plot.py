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

evm_op_color = "#ff7878"
go_arith_be_color = "#E24C4C"

arith_op_color = "#ffff78"
go_arith_le_color = "#a0a011"

asm384_op_color = "#61CD61"
asm384_arith_color = "#1EA41E"

model_color = "black"

setmod_color = "purple"

class ScatterPlotEntry():
    def __init__(self):
        self.x_range = None
        self.x_vals = []
        self.y_vals = []
        self.y_errs = []
        self.color = None
        self.label = None
        self.marker = None

    def add_datapoint(self, bit_width, time_ns):
        self.x_vals.append(int(bit_width))
        self.y_vals.append(int(time_ns))
        # TODO update xrange?

def parse_op_bench(data_file_name):
    with open(data_file_name) as f:
        lines = f.readlines()

    plot_entries = {} # map of preset x op -> ScatterPlotEntry
    for line in lines:
        parts = line.strip('\n').split(",")
        preset = parts[0]
        op = parts[1]
        bit_width = parts[2]
        time_ns = parts[3]

        if not preset+op in plot_entries:
            plot_entries[preset+op] = ScatterPlotEntry()

        plot_entries[preset+op].add_datapoint(bit_width, time_ns)

    return plot_entries

def parse_bls12381_bench():
    lines = []
    with open("benchmarks-results/geth-bls12381-benchmarks.csv") as f:
        lines = f.readlines()

    g1mul_benches = {}
    g2mul_benches = {}

    for line in lines:
        parts = line.strip('\n').strip(' ').split(',')
        preset = parts[0]
        bench = parts[1]
        time_ns = int(parts[2])
        gas_cost = int(parts[3])

        if bench == "g1mulgenorder":
            g1mul_benches[preset] = {"time_ns": time_ns, "gas_cost": gas_cost}
        elif bench == "g2mulgenorder":
            g2mul_benches[preset] = {"time_ns": time_ns, "gas_cost": gas_cost}
        else:
            continue

    return g1mul_benches, g2mul_benches


def plot_bls12381_time(g1mul_benches, g2mul_benches):
    times = {
            "native g1mul": (440000, 0, 490000),
            "evm g1mul": (g1mul_benches["arith384_asm"]["time_ns"], g1mul_benches["mulmont384_asm"]["time_ns"],g1mul_benches["fallback"]["time_ns"]),
            "native g2mul": (1296766, 0, 1515314),
            "evm g2mul":(g2mul_benches['arith384_asm']['time_ns'],g2mul_benches['mulmont384_asm']['time_ns'],g2mul_benches['fallback']['time_ns'])
            # old 
            #"asm384": (g1mul_benches["arith384_asm"]["time_ns"], g2mul_benches['arith384_asm']['time_ns']),
            #"mulmont384_asm": (g1mul_benches["mulmont384_asm"]["time_ns"], g2mul_benches['mulmont384_asm']['time_ns']),
            #"fallback": (g1mul_benches["fallback"]["time_ns"], g2mul_benches['fallback']['time_ns']),
    }
    categories = ("asm all ops", "asm mulmodx only", "fallback all ops")
    plot_bls12381(times, "charts/bls12381-time.png", categories, "Runtime: EVM and Native Implementations", "Runtime (μs)")

def plot_bls12381_gas_cost(g1mul_benches, g2mul_benches):
    gas_costs = {
            "g1mul precompile": (12000, 0, 0),
            "evm g1mul": (g1mul_benches["arith384_asm"]["gas_cost"], g1mul_benches["mulmont384_asm"]["gas_cost"],g1mul_benches["fallback"]["gas_cost"]),
            "g2mul precompile": (45000, 0, 0),
            "evm g2mul":(g2mul_benches['arith384_asm']['gas_cost'],g2mul_benches['mulmont384_asm']['gas_cost'],g2mul_benches['fallback']['gas_cost'])
            # old 
            #"asm384": (g1mul_benches["arith384_asm"]["time_ns"], g2mul_benches['arith384_asm']['time_ns']),
            #"mulmont384_asm": (g1mul_benches["mulmont384_asm"]["time_ns"], g2mul_benches['mulmont384_asm']['time_ns']),
            #"fallback": (g1mul_benches["fallback"]["time_ns"], g2mul_benches['fallback']['time_ns']),
    }
    categories = ("asm all ops", "asm mulmodx only", "fallback all ops")
    plot_bls12381(gas_costs, "charts/bls12381-gas.png", categories, "Gas Cost: EVM implementations and Precompile Prices", "Gas Cost")

def plot_bls12381(bench_data, output_file, categories, title, ylabel):
    x = np.arange(len(categories))  # the label locations
    width = 0.20  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in bench_data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    #ax.set_xticks(x + width, species)
    ax.legend(loc='upper left', ncols=3)
    ax.set_xticks(x + width, categories)

    plt.savefig(output_file)

g1mul_benches, g2mul_benches = parse_bls12381_bench()
plot_bls12381_time(g1mul_benches, g2mul_benches)
plot_bls12381_gas_cost(g1mul_benches, g2mul_benches)


arith_op_benchmarks = parse_op_bench("benchmarks-results/arith.csv")
evm_op_benchmarks = parse_op_bench("benchmarks-results/evm-op-benchmarks.csv")

bls12381_benchmarks = parse_bls12381_bench()



def scatterplot_ns_data(fname: str, name: str, args):
    #len_x = len(args[0][1])

    #x_min_all = min([min([v for v in d[1]]) for d in args])
    #x_max_all = max([max([v for v in d[1]]) for d in args])
    #y_min_all = min([min([v for v in d[2]]) for d in args])
    #y_max_all = max([max([v for v in d[2]]) for d in args])
    #span_x = x_max_all - x_min_all
    #span_y = y_max_all - y_min_all

    plt.rcParams["figure.figsize"] = (20, 10)
    fig, ax = plt.subplots()
    #plt.ylim(0, y_min_all + int(span_y * 1.2))
    #plt.xlim(0, x_min_all + int(span_x * 1.2))
    #plt.xticks(args[0][1])

    legend_lines = []
    legend_labels = []

    plt.xlabel("input size (bits)")
    plt.ylabel("runtime (ns)")

    for i, entry in enumerate(args):
        assert len(entry.x_vals) == len(entry.y_vals)
        
        if len(entry.y_errs) != 0:
            for x, y in zip(entry.x_vals, entry.y_vals):

                #if annotates[i]:
                #    ax.annotate(y, (float(x) + 0.2, float(y)))
                
                ax.errorbar(x=x, y=y, fmt=entry.marker, color=color)
            ax.plot(entry.x_vals, entry.y_vals, entry.marker, color=color, label=label)
        else:
           #if annotates[i]:
           #    for x, y in zip(entry.x_vals, entry.y_vals):
           #        # cheap hack because we are only executing this spot for
           #        # annotating the model
           #        ax.annotate(round(float(y) / GAS_RATE), (float(x) + 0.2, float(y)))
           ax.plot(entry.x_vals, entry.y_vals, entry.marker, color=entry.color, label=entry.label)
        
        legend_lines.append(Line2D([0], [0], color=entry.color, lw=4))
        legend_labels.append(entry.label)

    #x_axis_labels = [str(i*64) for i in range(1, len_x+1)]
    #x_axis_pos = np.arange(1,len_x + 1)
    #plt.xticks(x_axis_pos, x_axis_labels, color='black', fontsize='10')
    ax.legend(legend_lines, legend_labels, loc="upper left")
    plt.savefig(fname)

addmodx_fall_back_evm = evm_op_benchmarks['fallbackaddmodx']
addmodx_fall_back_evm.color = evm_op_color
addmodx_fall_back_evm.label = "1"
addmodx_fall_back_evm.marker = "o"

addmodx_fall_back_arith = arith_op_benchmarks['fallbackaddmodx']
addmodx_fall_back_arith.color = arith_op_color
addmodx_fall_back_arith.label = "2"
addmodx_fall_back_arith.marker = "o"

addmodx_asm384_evm = evm_op_benchmarks['arith384_asmaddmodx']
addmodx_asm384_evm.color = asm384_op_color
addmodx_asm384_evm.label = "1"
addmodx_asm384_evm.marker = "o"
addmodx_asm384_evm.x_range = 6

# TODO: make dict keys consistent named
addmodx_asm384_arith = arith_op_benchmarks['asm384addmodx']
addmodx_asm384_arith.color = asm384_arith_color
addmodx_asm384_arith.label = "2"
addmodx_asm384_arith.marker = "o"
addmodx_asm384_arith.x_range = 6
addmodx_asm384_evm.x_range = 6

submodx_fall_back_evm = evm_op_benchmarks['fallbacksubmodx']
submodx_fall_back_evm.color = asm384_op_color
submodx_fall_back_evm.label = "1"
submodx_fall_back_evm.marker = "o"

submodx_fall_back_arith = arith_op_benchmarks['fallbacksubmodx']
submodx_fall_back_arith.color = arith_op_color
submodx_fall_back_arith.label = "2"
submodx_fall_back_arith.marker = "o"

setmod_arith = arith_op_benchmarks['fallbacksetmod']
setmod_arith.color = arith_op_color
setmod_arith.label = "setmod arithmetic"
setmod_arith.marker = "o"


def plot_op_benchmarks():
    # graphs 1: scatterplot, addmodx.  entries: addsub-model, addmodx-fallback-evm, addmodx-fallback-arith, addmodx-asm384-arith, addmodx-asm384-evm
    scatterplot_ns_data("charts/addmodx.png", "addmodx", [addmodx_fall_back_evm, addmodx_fall_back_arith, addmodx_asm384_arith, addmodx_asm384_evm])
    # graphs 2: scatterplot submodx.  entries: addsub-model, submodx-fallback-evm, submodx-fallback-arith, submodx-asm384-arith, submodx-asm384-evm
    scatterplot_ns_data("charts/submodx.png", "submodx", [submodx_fall_back_evm, submodx_fall_back_arith])
    # graphs 3: scatterplot mulmodx.  entries: mulmodx-model, mulmodx-fallback-evm, mulmodx-fallback-arith, mulmodx-asm384-arith, mulmodx-asm384-evm
    # graphs 4: scatterplot setmod.  entries: setmod-model, setmod-arith
    scatterplot_ns_data("charts/setmod.png", "setmod", [setmod_arith])
    pass

plot_op_benchmarks()

def plot_bls_benchmarks():
    # graphs 1: g1/g2 mul execution time (group order).  group by (fallback evm + fallback go , mulmodx-asm evm, asm384 evm + asm go
    # graphs 2: g1/g2 mul cost (group order).  group by (precompile, asm384 evm, mulmodx-asm evm, fallback evm)
    pass

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

benches_xs = list(range(1, 16))
