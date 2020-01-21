
# Black-Box Functions Experiment

## Requirement

- python==3.6
- nevergrad==0.3.0
- tensorflow==1.10.0

Note that self-contained baselines is just used for plotting figures.

## Usage

To Evaluate all algorithms on blackbox functions：

`` python main.py --func_name Sphere --max_iter 1e5 --dims 1000``

Then plot all figures:

``python plot.py``

Finally, compute final performance statistics:

``python compute_stats.py``