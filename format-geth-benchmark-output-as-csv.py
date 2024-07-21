import sys, math, re

# input_lines = sys.stdin.readlines()

with open('geth-benchmark-output-example.txt', 'r') as f:
    input_lines = f.readlines()
    for line in input_lines[4:-2]:
        parts = [s for s in line.split(' ') if s]

        target_parts = parts[0][13:].split('-')[:-1]
        op, width, preset = target_parts[0], target_parts[1], target_parts[2]

        if parts[-1] != 'ns/op\n':
            raise Exception("fuck")

        time = math.ceil(float(parts[-2]))

        print("{},{},{},{}".format(preset, op, width, time))
