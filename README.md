# Gauge-informed PINNs for nonlinear non-reciprocal lattices

Code for the article *Gauge-informed neural networks for nonlinear non-reciprocal photonic lattices*,
by Ramsés C. S. Almeida (Universidade Federal de Alagoas).

Runs on CPU in float64 (JAX). Tested with Python 3.13.

## Draw the figures

```bash
pip install -r requirements.txt
python experiments/make_figs.py --all
```

The four figures are written to `figs/` from the data in `figdata/`.

## Quick check (about 10 minutes)

```bash
python experiments/exp_fig2.py --reference
python experiments/exp_fig2.py --seed 0 --time-budget 1e9
python experiments/exp_fig2.py --summarize
```

Trains the network of Fig. 2 (α = 1/4, h = 0.2, g = 2, W = 1) for one initialization and prints its
relative error at z = 10. The run in the article gave 3.88e-3 for this seed (mean over five seeds: 1.1e-2).

## Recompute the data

The training runs take several hours on a CPU. Each script writes its results to `results/`
and skips runs that already exist, so an interrupted run resumes when the command is repeated.

```bash
python experiments/exp_fig1c.py --all
python experiments/exp_fig2.py --all
python experiments/exp_fig3.py --all
python experiments/exp_fig4.py --map --h 0 && python experiments/exp_fig4.py --map --h 0.2
python experiments/exp_fig4.py --pick-regimes
python experiments/exp_fig4.py --all
python experiments/make_figs.py --export
python experiments/make_figs.py --all
```

## Layout

| path | content |
|---|---|
| `gipinn/` | lattice model, fractional gauge, reference solvers, network, losses, training, inversion, phase map |
| `experiments/` | one script per figure and `make_figs.py` |
| `figdata/` | the data plotted in the figures |

## License

MIT
