# Author: Ramsés C. S. Almeida (UFAL)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "Lattice",
    "sites",
    "eps_array",
    "gauge_factor",
    "disorder",
    "initial_psi",
    "neighbors",
    "rhs_psi",
    "rhs_phi",
    "residual_psi",
    "residual_phi",
    "psi_to_phi",
    "phi_to_psi",
    "frac_gauge_factor",
    "rhs_frac",
    "residual_frac",
    "P_phi",
    "H_phi",
    "ipr",
    "p_trap",
    "com_shift",
]

@dataclass(frozen=True)
class Lattice:
    N: int = 41
    n0: int = 21
    h: float = 0.0
    g: float = 0.0
    eps: Any = ()

def sites(N: int, xp=np):
    return xp.arange(1, N + 1, dtype=float)

def _resolve(lat: Lattice, h=None, g=None, eps=None, xp=np):
    h = lat.h if h is None else h
    g = lat.g if g is None else g
    eps = eps_array(lat, xp) if eps is None else xp.asarray(eps, dtype=float)
    return h, g, eps

def eps_array(lat: Lattice, xp=np):
    if len(lat.eps) == 0:
        return xp.zeros(lat.N, dtype=float)
    return xp.asarray(lat.eps, dtype=float)

def gauge_factor(lat: Lattice, xp=np, h=None):
    h = lat.h if h is None else h
    return xp.exp(h * (sites(lat.N, xp) - lat.n0))

def disorder(N: int, W: float, seed: int) -> np.ndarray:
    if W == 0:
        return np.zeros(N)
    rng = np.random.default_rng(seed)
    return rng.uniform(-W / 2, W / 2, N)

def initial_psi(lat: Lattice, P: float = 1.0, xp=np):
    psi0 = np.zeros(lat.N, dtype=complex)
    psi0[lat.n0 - 1] = np.sqrt(P)
    return xp.asarray(psi0)

def neighbors(u, xp):
    zero = xp.zeros(1, dtype=u.dtype)
    u_minus = xp.concatenate([zero, u[:-1]])
    u_plus = xp.concatenate([u[1:], zero])
    return u_minus, u_plus

def _X_psi(lat, psi, xp, h, g, eps):
    um, up = neighbors(psi, xp)
    return xp.exp(h) * um + xp.exp(-h) * up + eps * psi + g * xp.abs(psi) ** 2 * psi

def _X_phi(lat, phi, xp, h, g, eps):
    um, up = neighbors(phi, xp)
    g_eff = g * gauge_factor(lat, xp, h) ** 2
    return um + up + eps * phi + g_eff * xp.abs(phi) ** 2 * phi

def rhs_psi(lat: Lattice, psi, xp=np, h=None, g=None, eps=None):
    h, g, eps = _resolve(lat, h, g, eps, xp)
    return 1j * _X_psi(lat, psi, xp, h, g, eps)

def rhs_phi(lat: Lattice, phi, xp=np, h=None, g=None, eps=None):
    h, g, eps = _resolve(lat, h, g, eps, xp)
    return 1j * _X_phi(lat, phi, xp, h, g, eps)

def residual_psi(lat: Lattice, psi, dpsi, xp=np, h=None, g=None, eps=None):
    h, g, eps = _resolve(lat, h, g, eps, xp)
    return 1j * dpsi + _X_psi(lat, psi, xp, h, g, eps)

def residual_phi(lat: Lattice, phi, dphi, xp=np, h=None, g=None, eps=None):
    h, g, eps = _resolve(lat, h, g, eps, xp)
    return 1j * dphi + _X_phi(lat, phi, xp, h, g, eps)

def frac_gauge_factor(lat: Lattice, alpha: float, xp=np, h=None):
    h = lat.h if h is None else h
    return xp.exp(alpha * h * (sites(lat.N, xp) - lat.n0))

def _X_frac(lat, u, alpha, xp, h, g, eps):
    um, up = neighbors(u, xp)
    g_eff = g * frac_gauge_factor(lat, alpha, xp, h) ** 2
    return xp.exp((1.0 - alpha) * h) * um + xp.exp(-(1.0 - alpha) * h) * up + eps * u + g_eff * xp.abs(u) ** 2 * u

def rhs_frac(lat: Lattice, u, alpha: float, xp=np, h=None, g=None, eps=None):
    h, g, eps = _resolve(lat, h, g, eps, xp)
    return 1j * _X_frac(lat, u, alpha, xp, h, g, eps)

def residual_frac(lat: Lattice, u, du, alpha: float, xp=np, h=None, g=None, eps=None):
    h, g, eps = _resolve(lat, h, g, eps, xp)
    return 1j * du + _X_frac(lat, u, alpha, xp, h, g, eps)

def psi_to_phi(lat: Lattice, psi, xp=np, h=None):
    return psi / gauge_factor(lat, xp, h)

def phi_to_psi(lat: Lattice, phi, xp=np, h=None):
    return phi * gauge_factor(lat, xp, h)

def P_phi(phi, xp=np):
    return xp.sum(xp.abs(phi) ** 2)

def H_phi(lat: Lattice, phi, xp=np, h=None, g=None, eps=None):
    h, g, eps = _resolve(lat, h, g, eps, xp)
    g_eff = g * gauge_factor(lat, xp, h) ** 2
    a2 = xp.abs(phi) ** 2
    hop = 2.0 * xp.sum(xp.real(phi[:-1] * xp.conj(phi[1:])))
    return -(hop + xp.sum(eps * a2) + 0.5 * xp.sum(g_eff * a2**2))

def ipr(psi, xp=np):
    a2 = xp.abs(psi) ** 2
    return xp.sum(a2**2) / xp.sum(a2) ** 2

def p_trap(lat: Lattice, psi, width: int = 2, xp=np):
    a2 = xp.abs(psi) ** 2
    mask = xp.abs(sites(lat.N, xp) - lat.n0) <= width
    return xp.sum(xp.where(mask, a2, 0.0)) / xp.sum(a2)

def com_shift(lat: Lattice, psi, xp=np):
    a2 = xp.abs(psi) ** 2
    return xp.sum(sites(lat.N, xp) * a2) / xp.sum(a2) - lat.n0
