from itertools import product
import random

rng = random.Random(42)

window_size = [60, 300, 1200, 1800]
variables = ["rso2r rso2l abp", ""]
downsample = [1, 2]
variance = [95, 80, 60]
frequency = [60, 10, 1]
r2_threshold = [-1e2, 0.2, 0.7]
smooth_frac = [0.2, 0.46]

lines = []

for (
    w, v, g, x, f, r, s
) in product(window_size, variables, downsample, variance, frequency, r2_threshold, smooth_frac):

    base = f"-m smooth -g {g} -w {w} -x {x} -f {f} -r {r} -sf {s}"
    if v != "":
        base = base + f" -v {v}"

    lines.append(base)

# randomize lines
rng.shuffle(lines)

with open("/home/mr2238/accelerate/scripts/grid/dataset_array.txt", "w") as f:
    f.write("\n".join(lines))

print(f"Generated {len(lines)} combinations")