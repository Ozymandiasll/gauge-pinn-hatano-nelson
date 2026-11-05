# Author: Ramsés C. S. Almeida (UFAL)

import jax

jax.config.update("jax_enable_x64", True)

__all__ = ["lattice", "integrate", "network", "losses", "train", "inverse", "phase_map"]
