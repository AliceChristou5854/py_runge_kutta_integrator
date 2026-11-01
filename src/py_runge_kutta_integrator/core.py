"""Core implementation of the fixed-step fourth-order Runge-Kutta integrator.

The design goal is deliberately narrow: take a plain Python callable that
describes a first-order system ``y' = f(t, y)`` and advance it with the
classical RK4 tableau.  No adaptive stepping, no dense output, no external
dependencies.  That keeps the code small enough to audit and makes the
behaviour predictable, which matters when the integrator is used as a
building block inside a larger simulation loop.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

Vector = Sequence[float]
Derivative = Callable[[float, Vector], Vector]


class RK4Integrator:
    """Fixed-step fourth-order Runge-Kutta integrator.

    Parameters
    ----------
    f:
        The right-hand side of the ODE system.  Called as ``f(t, y)`` where
        ``y`` is a sequence of floats; must return a sequence of the same
        length containing ``dy/dt``.
    y0:
        Initial state.  The integrator copies this into an internal list so
        later mutation of the caller's sequence cannot corrupt the state.
    t0:
        Initial time.
    dt:
        Fixed step size.  Must be strictly positive.

    Raises
    ------
    TypeError:
        If ``f`` is not callable or ``y0`` is not a sequence of numbers.
    ValueError:
        If ``dt`` is not strictly positive, or if the initial state is empty.
    """

    __slots__ = ("_f", "_y", "_t", "_dt", "_dim")

    def __init__(
        self,
        f: Derivative,
        y0: Sequence[float],
        t0: float = 0.0,
        dt: float = 1e-2,
    ) -> None:
        if not callable(f):
            raise TypeError("f must be callable")
        if dt <= 0.0:
            raise ValueError("dt must be strictly positive")
        try:
            y_list = [float(v) for v in y0]
        except TypeError as exc:  # y0 not iterable
            raise TypeError("y0 must be a sequence of numbers") from exc
        except ValueError as exc:  # y0 contains non-numeric values
            raise TypeError("y0 must be a sequence of numbers") from exc
        if not y_list:
            raise ValueError("y0 must contain at least one component")
        self._f = f
        self._y: List[float] = y_list
        self._t = float(t0)
        self._dt = float(dt)
        self._dim = len(y_list)

    @property
    def t(self) -> float:
        """Current integration time."""
        return self._t

    @property
    def y(self) -> Tuple[float, ...]:
        """Current state as an immutable tuple.

        Returning a tuple rather than the internal list prevents callers from
        mutating integrator state by accident; the cost is one allocation per
        call, which is irrelevant compared to the four derivative evaluations
        performed by :meth:`step`.
        """
        return tuple(self._y)

    @property
    def dt(self) -> float:
        """Fixed step size used by :meth:`step`."""
        return self._dt

    def step(self) -> Tuple[float, Tuple[float, ...]]:
        """Advance the state by one fixed step.

        Returns
        -------
        (t_new, y_new):
            The time and state after the step.  ``y_new`` is a tuple so the
            caller cannot alias the integrator's internal list.
        """
        f = self._f
        t = self._t
        dt = self._dt
        y = self._y

        k1 = self._evaluate(f, t, y)
        k2 = self._evaluate(f, t + 0.5 * dt, self._axpy(y, 0.5 * dt, k1))
        k3 = self._evaluate(f, t + 0.5 * dt, self._axpy(y, 0.5 * dt, k2))
        k4 = self._evaluate(f, t + dt, self._axpy(y, dt, k3))

        sixth = dt / 6.0
        y_new = [
            y[i] + sixth * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
            for i in range(self._dim)
        ]
        self._y = y_new
        self._t = t + dt
        return self._t, tuple(y_new)

    def integrate(self, n_steps: int) -> Tuple[float, Tuple[float, ...]]:
        """Advance the state by ``n_steps`` fixed steps.

        Parameters
        ----------
        n_steps:
            Number of steps to take.  Must be a non-negative integer.

        Returns
        -------
        (t, y):
            Time and state after the final step.  With ``n_steps == 0`` this
            is simply the current ``(t, y)``.
        """
        if not isinstance(n_steps, int) or isinstance(n_steps, bool):
            raise TypeError("n_steps must be an integer")
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        for _ in range(n_steps):
            self.step()
        return self._t, tuple(self._y)

    def _evaluate(self, f: Derivative, t: float, y: List[float]) -> List[float]:
        """Call ``f`` and validate the shape of the returned derivative.

        Shape checking is done on every evaluation because a mismatched
        derivative length would otherwise produce an ``IndexError`` deep
        inside the stage combination, which is much harder to diagnose.
        """
        out = f(t, list(y))
        try:
            values = [float(v) for v in out]
        except TypeError as exc:
            raise TypeError("f must return an iterable of numbers") from exc
        if len(values) != self._dim:
            raise ValueError(
                "f returned a derivative of length %d, expected %d"
                % (len(values), self._dim)
            )
        return values

    @staticmethod
    def _axpy(y: List[float], scale: float, k: List[float]) -> List[float]:
        """Return ``y + scale * k`` as a new list."""
        return [y[i] + scale * k[i] for i in range(len(y))]
