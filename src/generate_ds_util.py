from itertools import product
import random

rng = random.Random(42)

window_size = [300, 900, 1800]
variables = ["rso2r rso2l abp", ""]
variance = [95, 80, 60]
frequency = [60, 10, 1]
smooth_frac = [0.2, 0.46]

lines = []

for w, v, x, f, s in product(window_size, variables, variance, frequency, smooth_frac):

    base = f"-m smooth -g 2 -w {w} -x {x} -f {f} -sf {s} -v {v}"
    base_chop = base + " -c 2"

    lines.append(base)
    lines.append(base_chop)

# randomize lines
rng.shuffle(lines)

with open("/home/mr2238/accelerate/scripts/grid/dataset_array.txt", "w") as f:
    f.write("\n".join(lines))

print(f"Generated {len(lines)} combinations")
