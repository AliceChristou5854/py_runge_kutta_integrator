# py_runge_kutta_integrator

A fixed-step fourth-order Runge-Kutta integrator for systems of first-order ODEs, driven by a plain Python callable and depending only on the standard library.

```python
from py_runge_kutta_integrator import RK4Integrator

# y' = -2 * t * y, y(0) = 1  =>  y(t) = exp(-t^2)
def rhs(t, y):
    return [-2.0 * t * y[0]]

integrator = RK4Integrator(rhs, [1.0], t0=0.0, dt=1e-3)
t, y = integrator.integrate(1000)
print(t, y[0])  # 1.0 0.36787944117144233
```

`RK4Integrator(f, y0, t0=0.0, dt=1e-2)` stores the state internally.  `step()` advances one step and returns `(t, y)`; `integrate(n)` advances `n` steps and returns the same pair.  The properties `t`, `y`, and `dt` expose the current time, the current state as a tuple, and the step size.  `y0` is copied at construction, and `y` is returned as a tuple, so the caller's data can never alias the integrator's state.

## Why this exists

The scientific Python world already has `scipy.integrate.solve_ivp`, but pulling SciPy into a container with no network access is not always possible, and the adaptive-step machinery is a lot of surface area when all you need is a predictable fixed step.  This library makes one trade-off: no adaptivity, no dense output, no event detection — just the classical RK4 tableau applied to a callable of the form `f(t, y) -> dy/dt`.  The payoff is a single class with no imports outside the standard library, which is easy to vendor and easy to reason about when it sits inside a larger simulation loop.

## The edge you will hit

`f` is called four times per step, and each call receives a fresh list.  If your derivative function mutates `y` in place, the mutation is discarded — the integrator does not read it back.  That is deliberate: it keeps the internal state safe from accidental aliasing, but it means in-place derivative functions will silently do nothing.  Return a new sequence from `f`.

`f` must return a derivative whose length matches `y0`.  A mismatch raises `ValueError` on the first step rather than producing a confusing `IndexError` later.  `dt` must be strictly positive; zero or negative values raise `ValueError` at construction.

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.

