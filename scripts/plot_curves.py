import glob
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"sparse": "tab:blue", "native": "tab:orange", "pbrs": "tab:green"}
seen = set()
for path in sorted(glob.glob("runs_sac/*/eval.csv")):
    mode = os.path.basename(os.path.dirname(path)).rsplit("_s", 1)[0]
    d = np.loadtxt(path, delimiter=",", ndmin=2)
    plt.plot(d[:, 0], d[:, 1], color=COLORS[mode], alpha=0.6, label=None if mode in seen else mode)
    seen.add(mode)
plt.xlabel("env steps")
plt.ylabel("eval success (25 seeds)")
plt.legend()
plt.savefig("./results/curves.png", dpi=150)
