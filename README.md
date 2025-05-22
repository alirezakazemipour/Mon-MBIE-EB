Source code of the algorithm [Monitored MBIE-EB](https://arxiv.org/abs/2502.16772).

## Install

To install and use our environments, run
```
pip install -r requirements.txt
cd src/gym-monitor
pip install -e .
```

## Hydra Configs
We use [Hydra](https://hydra.cc/docs/intro/) to configure our experiments.  
Hyperparameters and other settings are defined in YAML files in the `configs/` folder.

## Sweeps
For a sweep over multiple jobs in parallel with Joblib, run
```
python main.py -m hydra/launcher=joblib hydra/sweeper=manual_sweeper
```
Custom sweeps are defined in `configs/hydra/sweeper/`.  
You can further customize a sweep via command line. For example,
```
python main.py -m hydra/launcher=joblib hydra/sweeper=manual_sweeper experiment.rng_seed="range(0, 10)" hydra.launcher.verbose=1000
```
Configs in `configs/hydra/sweeper/` hide the training progress bar of the agent, so we
suggest to pass `hydra.launcher.verbose=1000` to show the progress of the sweep.

## Plot Data From Sweeps
Experiments will save the expected discounted return of the ε-greedy (training)
and greedy (testing) policies in `npy` files (default dir is `data/`).  
If you want to zip and copy only the data needed for plotting, run
```
find data -type f -name "*test*.npy" -print0 | tar -czvf data.tar.gz --null -T -
```

To plot expected return curves, use `plot_curves.py`. This script takes two arguments:
- `-c` is the config file that defines where to save plots, axes limits, axes ticks,
  what algorithms to show, and so on. Default configs are located in `configs/plots/`.
- `-f` is the folder where data from the sweep is located.

For example, running
```
python plot_curves.py -c configs/plots/deterministic_appendix.py -f data/iGym-Monitor/
```
Will generate many plots like these two, and save them in `data/iGym-Monitor/deterministic_appendix`.

<p align="center">
  <img src="figures/Gridworld-Medium-3x3-v0_mes50_Easy.png" width=200 alt="Gridworld-Medium-3x3-v0_mes50_Easy"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="figures/Gridworld-Medium-3x3-v0_mes50_iLimitedTimeMonitor.png" width=200 alt="Gridworld-Medium-3x3-v0_mes50_iLimitedTimeMonitor">
</p>

Finally, `python plot_legend.py` will generate a separate pic with only the legend.

<p align="center">
  <img src="figures/legend.png" width=500 alt=Legend">
</p>
