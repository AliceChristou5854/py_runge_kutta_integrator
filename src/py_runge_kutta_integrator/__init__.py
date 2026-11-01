"""Fixed-step fourth-order Runge-Kutta integrator for systems of ODEs.

This package exposes a single public class, :class:`RK4Integrator`, which
integrates an arbitrary system of first-order ordinary differential equations
supplied as a plain Python callable.  The callable signature is
``f(t, y) -> dy/dt`` where ``y`` is a sequence of floats.
"""

from .core import RK4Integrator

__all__ = ["RK4Integrator"]
