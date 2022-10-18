import sys, math

input_lines = sys.stdin.readlines()
for line in input_lines[4:-2]:
    parts = [s for s in line.split(' ') if s]
    name = parts[0].split('/')[0][9:]
    limbs = int(int(parts[0].split('/')[1][:-6]) / 64)
    time = math.ceil(float(parts[-2]))
    print("{},{},{}".format(name,limbs,time))
