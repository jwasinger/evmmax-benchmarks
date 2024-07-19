#!/usr/bin/env python
# coding: utf-8

# In[31]:


from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
import numpy as np
import math, sys, re, statistics, os
import copy

from matplotlib.pyplot import figure

# 25 ns / gas
GAS_RATE = 27.0

arith_op_color = "#36ba41"

evm_op_color = "#0f6316"

asm384_op_color = "#871215"
asm384_arith_color = "#c2363a"

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
        self.annotate = False
        self.is_model = False

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
    # bar order: gnark-crypto asm, evm asm, evm asm-modx, evm fallback
    g1_times = [round(361000 / GAS_RATE), round(g1mul_benches["arith384_asm"]["time_ns"] / GAS_RATE), round(g1mul_benches["mulmont384_asm"]["time_ns"] / GAS_RATE), round(g1mul_benches["fallback"]["time_ns"] / GAS_RATE)]
    preset_names = ["gnark-crypto asm", "evm asm", "evm asm-mulmodx", "evm fallback"]

    g2_times = [round(982000 / GAS_RATE), round(g2mul_benches['arith384_asm']['time_ns'] / GAS_RATE), round(g2mul_benches['mulmont384_asm']['time_ns'] / GAS_RATE), round(g2mul_benches['fallback']['time_ns'] / GAS_RATE)]

    #g1_times = {
    #        "native g1mul": (440, 0, 490),
    #        "evm g1mul": (round(g1mul_benches["arith384_asm"]["time_ns"] / 1000), round(g1mul_benches["mulmont384_asm"]["time_ns"] / 1000),round(g1mul_benches["fallback"]["time_ns"] / 1000)),
    #}
    #g2_times = {
    #        "native g2mul": (round(1296766 / 1000), 0, round(1515314 / 1000)),
    #        "evm g2mul":(round(g2mul_benches['arith384_asm']['time_ns'] / 1000),round(g2mul_benches['mulmont384_asm']['time_ns'] / 1000),round(g2mul_benches['fallback']['time_ns'] / 1000)),
    #        }
    categories = ("asm all ops", "asm mulmodx only", "fallback all ops")

    plot_bls12381(preset_names, g1_times, "charts/bls12381-time-g1.png", categories, "Gas Cost computed as Runtime / Target Gas Rate", "gas cost")
    plot_bls12381(preset_names, g2_times, "charts/bls12381-time-g2.png", categories, "Gas Cost computed as Runtime / Target Gas Rate", "gas cost")
    #plot_bls12381(g2_times, "charts/bls12381-time-g2.png", categories, "Runtime: EVM and Native Implementations", "Runtime (μs)")

def plot_bls12381_gas_cost(g1mul_benches, g2mul_benches):
    g1_gas_costs = {
            "g1mul precompile": (12000, 0, 0),
            "evm g1mul": (g1mul_benches["arith384_asm"]["gas_cost"], g1mul_benches["mulmont384_asm"]["gas_cost"],g1mul_benches["fallback"]["gas_cost"]),
    }
    g2_gas_costs = {
            "g2mul precompile": (45000, 0, 0),
            "evm g2mul":(g2mul_benches['arith384_asm']['gas_cost'],g2mul_benches['mulmont384_asm']['gas_cost'],g2mul_benches['fallback']['gas_cost']),
    }

    preset_names = ["precompile", "evm asm", "evm asm-mulmodx", "evm fallback"]
    g1_gas_costs = [12000, g1mul_benches["arith384_asm"]["gas_cost"], g1mul_benches["mulmont384_asm"]["gas_cost"], g1mul_benches["fallback"]["gas_cost"]]
    g2_gas_costs = [45000, g2mul_benches["arith384_asm"]["gas_cost"], g2mul_benches["mulmont384_asm"]["gas_cost"], g2mul_benches["fallback"]["gas_cost"]]

    categories = ("asm all ops", "asm mulmodx only", "fallback all ops")
    plot_bls12381(preset_names, g1_gas_costs, "charts/bls12381-gas-g1.png", categories, "EVM Gas Costs", "Gas Cost")
    plot_bls12381(preset_names, g2_gas_costs, "charts/bls12381-gas-g2.png", categories, "EVM Gas Costs", "Gas Cost")

def plot_bls12381(names, values, output_file, categories, title, ylabel):

    width = 0.3
    fig, ax = plt.subplots()
    rects = ax.bar(names, values, width)
    ax.bar_label(rects, padding=3)
    #ax.bar_label(rects, padding=3)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)

    ax.set_title(title)
    #ax.set_xticks(x + width, species)
    #ax.legend(loc='upper left', ncols=3)

    plt.savefig(output_file)




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

    plt.xlabel("modulus size (bits)")
    plt.ylabel("gas cost")

    for i, entry in enumerate(args):
       assert len(entry.x_vals) == len(entry.y_vals)
        
       for x, y in zip(entry.x_vals, entry.y_vals):
           if entry.annotate:
               ax.annotate(int(y), (float(x) + 0.2, float(y)))

       x_vals = entry.x_vals
       y_vals = entry.y_vals
       if not entry.is_model:
           y_vals = [val / GAS_RATE for val in y_vals]

       ax.plot(x_vals, y_vals, entry.marker, color=entry.color, label=entry.label)

       legend_lines.append(Line2D([0], [0], color=entry.color, lw=4))
       legend_labels.append(entry.label)

    x_ticks = ax.get_xticks()[:-1]
    ax.set_xticks(x_ticks, labels=[int(val * 64) for val in x_ticks])

    #x_axis_labels = [str(i*64) for i in range(1, len_x+1)]
    #x_axis_pos = np.arange(1,len_x + 1)
    #plt.xticks(x_axis_pos, x_axis_labels, color='black', fontsize='10')
    ax.legend(legend_lines, legend_labels, loc="upper left")
    plt.savefig(fname)

def fit_quadratic(xs, ys, intercept_min):
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

def fit_linear(xs, ys, intercept_min):

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

def __eval_model(model: [int], x: int) -> int:
    result = 0
    for i, coef in enumerate(model):
        if i != len(model) - 1:
            result += coef * (x ** (len(model) - 1 - i))
        else:
            result += coef

    return result


def eval_model(model: [int], xs: [int]):
    result = []

    for x in xs:
        result.append(__eval_model(model, x))

    return result

arith_op_benchmarks = parse_op_bench("benchmarks-results/arith.csv")
evm_op_benchmarks = parse_op_bench("benchmarks-results/evm-op-benchmarks.csv")

addmodx_fall_back_evm = evm_op_benchmarks['fallbackaddmodx']
addmodx_fall_back_evm.color = evm_op_color
addmodx_fall_back_evm.label = "go (evm)"
addmodx_fall_back_evm.marker = "o"

addmodx_fall_back_arith = arith_op_benchmarks['fallbackaddmodx']
addmodx_fall_back_arith.color = arith_op_color
addmodx_fall_back_arith.label = "go (arith)"
addmodx_fall_back_arith.marker = "o"

addmodx_asm384_evm = evm_op_benchmarks['arith384_asmaddmodx']
addmodx_asm384_evm.color = asm384_op_color
addmodx_asm384_evm.label = "asm (evm)"
addmodx_asm384_evm.marker = "o"
addmodx_asm384_evm.x_range = 6

# TODO: make dict keys consistent named
addmodx_asm384_arith = arith_op_benchmarks['asm384addmodx']
addmodx_asm384_arith.color = asm384_arith_color
addmodx_asm384_arith.label = "asm (arith)"
addmodx_asm384_arith.marker = "o"
addmodx_asm384_arith.x_range = 6
addmodx_asm384_evm.x_range = 6

# submodx

submodx_fall_back_evm = evm_op_benchmarks['fallbacksubmodx']
submodx_fall_back_evm.color = asm384_op_color
submodx_fall_back_evm.label = "go (evm)"
submodx_fall_back_evm.marker = "o"

submodx_fall_back_arith = arith_op_benchmarks['fallbacksubmodx']
submodx_fall_back_arith.color = arith_op_color
submodx_fall_back_arith.label = "go (arith)"
submodx_fall_back_arith.marker = "o"

# TODO: submod asm ops

# mulmodx

mulmodx_fall_back_evm = evm_op_benchmarks['fallbackmulmodx']
mulmodx_fall_back_evm.color = evm_op_color
mulmodx_fall_back_evm.label = "go (evm)"
mulmodx_fall_back_evm.marker = "o"

mulmodx_fall_back_arith = arith_op_benchmarks['fallbackmulmodx']
mulmodx_fall_back_arith.color = arith_op_color
mulmodx_fall_back_arith.label = "go (arith)"
mulmodx_fall_back_arith.marker = "o"

mulmodx_asm384_evm = evm_op_benchmarks['arith384_asmmulmodx']
mulmodx_asm384_evm.color = asm384_op_color
mulmodx_asm384_evm.label = "asm (evm)"
mulmodx_asm384_evm.marker = "o"
mulmodx_asm384_evm.x_range = 6

mulmodx_asm384_arith = arith_op_benchmarks['asm384mulmodx']
mulmodx_asm384_arith.color = asm384_arith_color
mulmodx_asm384_arith.label = "asm (arith)"
mulmodx_asm384_arith.marker = "o"
mulmodx_asm384_arith.x_range = 6
mulmodx_asm384_evm.x_range = 6

# setmod

setmod_arith = arith_op_benchmarks['fallbacksetmod']
setmod_arith.color = arith_op_color
setmod_arith.label = "go (arith)"
setmod_arith.marker = "o"


# gas models

# addmodx/submodx
#addmodx_model = fit_linear(addmodx_fall_back_evm.x_vals, addmodx_fall_back_evm.y_vals, 0)
addmodx_model = [np.float64(3), np.float64(30)]

addmodx_model_entry = ScatterPlotEntry()
addmodx_model_entry.x_vals = addmodx_fall_back_evm.x_vals
addmodx_model_entry.y_vals = [math.ceil(y_val / GAS_RATE) for y_val in eval_model(addmodx_model, addmodx_model_entry.x_vals)]
addmodx_model_entry.color = "black"
addmodx_model_entry.label = "addmodx gas model"
addmodx_model_entry.marker = "o"
addmodx_model_entry.annotate = True
addmodx_model_entry.is_model = True

# submodx
submodx_model_entry = copy.deepcopy(addmodx_model_entry)
submodx_model_entry.label = "submodx gas model"

# mulmodx
#mulmodx_model = fit_quadratic(mulmodx_fall_back_evm.x_vals, mulmodx_fall_back_evm.y_vals, 0)
mulmodx_model = [np.float64(2.30), np.float64(-2), np.float64(32)]

mulmodx_model_entry = ScatterPlotEntry()
mulmodx_model_entry.x_vals = mulmodx_fall_back_evm.x_vals
mulmodx_model_entry.y_vals = [math.ceil(y_val / GAS_RATE) for y_val in eval_model(mulmodx_model, mulmodx_model_entry.x_vals)]
mulmodx_model_entry.color = "black"
mulmodx_model_entry.label = "mulmodx gas model"
mulmodx_model_entry.marker = "o"
mulmodx_model_entry.annotate = True
mulmodx_model_entry.is_model = True


# setmod
# setmod_model = fit_linear(setmod_arith.x_vals, setmod_arith.y_vals, 0)

setmod_model = [np.float64(110), np.float64(730)]

setmod_model_entry = ScatterPlotEntry()
setmod_model_entry.x_vals = setmod_arith.x_vals
setmod_model_entry.y_vals = [math.ceil(y_val / GAS_RATE) for y_val in eval_model(setmod_model, setmod_model_entry.x_vals)]
setmod_model_entry.color = "black"
setmod_model_entry.label = "setmod gas model"
setmod_model_entry.marker = "o"
setmod_model_entry.annotate = True
setmod_model_entry.is_model = True

def plot_op_benchmarks():
    # graphs 1: scatterplot, addmodx.  entries: addsub-model, addmodx-fallback-evm, addmodx-fallback-arith, addmodx-asm384-arith, addmodx-asm384-evm
    scatterplot_ns_data("charts/addmodx-asm.png", "addmodx", [addmodx_fall_back_evm, addmodx_fall_back_arith, addmodx_asm384_arith, addmodx_asm384_evm, addmodx_model_entry])
    # graphs 2: scatterplot submodx.  entries: addsub-model, submodx-fallback-evm, submodx-fallback-arith, submodx-asm384-arith, submodx-asm384-evm
    scatterplot_ns_data("charts/submodx-asm.png", "submodx", [submodx_fall_back_evm, submodx_fall_back_arith, submodx_model_entry])
    # graphs 3: scatterplot mulmodx.  entries: mulmodx-model, mulmodx-fallback-evm, mulmodx-fallback-arith, mulmodx-asm384-arith, mulmodx-asm384-evm
    scatterplot_ns_data("charts/mulmodx-asm.png", "addmodx", [mulmodx_fall_back_evm, mulmodx_fall_back_arith, mulmodx_asm384_arith, mulmodx_asm384_evm, mulmodx_model_entry])
    # graphs 4: scatterplot setmod.  entries: setmod-model, setmod-arith
    scatterplot_ns_data("charts/setmod.png", "setmod", [setmod_arith, setmod_model_entry])
    #without asm
    # graphs 1: scatterplot, addmodx.  entries: addsub-model, addmodx-fallback-evm, addmodx-fallback-arith, addmodx-asm384-arith, addmodx-asm384-evm
    scatterplot_ns_data("charts/addmodx.png", "addmodx", [addmodx_fall_back_evm, addmodx_fall_back_arith, addmodx_model_entry])
    # graphs 2: scatterplot submodx.  entries: addsub-model, submodx-fallback-evm, submodx-fallback-arith, submodx-asm384-arith, submodx-asm384-evm
    scatterplot_ns_data("charts/submodx.png", "submodx", [submodx_fall_back_evm, submodx_fall_back_arith, submodx_model_entry])
    # graphs 3: scatterplot mulmodx.  entries: mulmodx-model, mulmodx-fallback-evm, mulmodx-fallback-arith, mulmodx-asm384-arith, mulmodx-asm384-evm
    scatterplot_ns_data("charts/mulmodx.png", "addmodx", [mulmodx_fall_back_evm, mulmodx_fall_back_arith, mulmodx_model_entry])

def print_cost_model_table():
    print("| Modulus Size | MULMODX cost | ADDMODX and SUBMODX cost | SETMOD cost |")
    print("| ---- | ---- |  ---- | ---- |")

    for limb_count in range(1, 13):
        cost_mulmodx = math.ceil(__eval_model(mulmodx_model, limb_count) / 27)
        cost_addmodx =  math.ceil(__eval_model(addmodx_model, limb_count) / 27)
        cost_setmod =  math.ceil(__eval_model(setmod_model, limb_count) / 27)
        print("|{} - {} bits|{}|{}|{}|".format((limb_count - 1) * 64 + 1, limb_count * 64, cost_mulmodx, cost_addmodx, cost_setmod))

if __name__ == "__main__":
    plot_op_benchmarks()

    g1mul_benches, g2mul_benches = parse_bls12381_bench()
    plot_bls12381_time(g1mul_benches, g2mul_benches)
    plot_bls12381_gas_cost(g1mul_benches, g2mul_benches)


    bls12381_benchmarks = parse_bls12381_bench()
