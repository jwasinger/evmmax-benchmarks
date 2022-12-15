import sys, math, re

input_lines = None # sys.stdin.readlines()
with open("benchmarks-results/go-arith-benchmarks-be.txt", "r") as f:
    input_lines = f.readlines()

for line in input_lines[4:-2]:
    parts = [s for s in line.split(' ') if s]
    name = parts[0].split('/')[1].split('_')[1]
    limbs = re.sub(r'#.*$', '', parts[0].split('/')[1].split('_')[2])
    limbs = re.sub(r'-.*$', '', limbs)
    limbs = int(int(limbs) / 64)
    time = math.ceil(float(parts[-2]))
    print("{},{},{}".format(name,limbs,time))
