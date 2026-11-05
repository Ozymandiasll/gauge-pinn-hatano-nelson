# Gauge-informed PINNs for nonlinear non-reciprocal lattices

Code for the article *Gauge-informed neural networks for nonlinear non-reciprocal photonic lattices*,
by Ramsés C. S. Almeida (Universidade Federal de Alagoas).

Runs on CPU in float64 (JAX).

## Draw the figures

```bash
pip install -r requirements.txt
python experiments/make_figs.py --all
```

The four figures are written to `figs/` from the data in `figdata/`.

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
