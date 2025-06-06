<p align='left'>
  <a href="https://github.com/numba/numba"><img alt="Numba" src="https://img.shields.io/badge/Numba-000?logo=numba&style=for-the-badge" /></a>
</p>

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
`simple_plot.py` is the script that plots the results of sweeps collected in the `data` folder.