import math
import unittest

from py_runge_kutta_integrator import RK4Integrator


class TestConstruction(unittest.TestCase):
    def test_rejects_non_callable_f(self):
        with self.assertRaises(TypeError):
            RK4Integrator(42, [1.0])

    def test_rejects_non_positive_dt(self):
        with self.assertRaises(ValueError):
            RK4Integrator(lambda t, y: [0.0], [1.0], dt=0.0)
        with self.assertRaises(ValueError):
            RK4Integrator(lambda t, y: [0.0], [1.0], dt=-0.5)

    def test_rejects_empty_state(self):
        with self.assertRaises(ValueError):
            RK4Integrator(lambda t, y: [], [])

    def test_copies_initial_state(self):
        y0 = [1.0, 2.0]
        integrator = RK4Integrator(lambda t, y: [0.0, 0.0], y0)
        y0[0] = 999.0
        self.assertEqual(integrator.y, (1.0, 2.0))

    def test_initial_properties(self):
        integrator = RK4Integrator(lambda t, y: [0.0], [3.5], t0=1.25, dt=0.1)
        self.assertEqual(integrator.t, 1.25)
        self.assertEqual(integrator.dt, 0.1)
        self.assertEqual(integrator.y, (3.5,))


class TestStepping(unittest.TestCase):
    def test_zero_derivative_keeps_state(self):
        integrator = RK4Integrator(lambda t, y: [0.0], [7.0], dt=0.5)
        t, y = integrator.step()
        self.assertAlmostEqual(t, 0.5)
        self.assertEqual(y, (7.0,))

    def test_exponential_growth_matches_analytic(self):
        # y' = y, y(0) = 1 => y(t) = exp(t).  RK4 error per step is O(dt^5),
        # so after 100 steps of 0.01 the accumulated error is tiny.
        integrator = RK4Integrator(lambda t, y: [y[0]], [1.0], dt=0.01)
        t, y = integrator.integrate(100)
        self.assertAlmostEqual(t, 1.0, places=12)
        self.assertAlmostEqual(y[0], math.exp(1.0), places=9)

    def test_harmonic_oscillator_preserves_amplitude(self):
        # y'' = -y  ->  (x, v)' = (v, -x).  RK4 is not symplectic, but over
        # one period with a fine step the amplitude drift is negligible.
        def rhs(t, y):
            return [y[1], -y[0]]

        integrator = RK4Integrator(rhs, [1.0, 0.0], dt=1e-3)
        steps = int(round(2.0 * math.pi / 1e-3))
        _, y = integrator.integrate(steps)
        self.assertAlmostEqual(y[0], 1.0, places=4)
        self.assertAlmostEqual(y[1], 0.0, places=3)

    def test_state_is_immutable_tuple(self):
        integrator = RK4Integrator(lambda t, y: [0.0], [1.0])
        result = integrator.y
        self.assertIsInstance(result, tuple)
        t, y = integrator.step()
        self.assertIsInstance(y, tuple)

    def test_integrate_zero_steps_is_noop(self):
        integrator = RK4Integrator(lambda t, y: [1.0], [0.0], t0=2.0, dt=0.25)
        t, y = integrator.integrate(0)
        self.assertEqual(t, 2.0)
        self.assertEqual(y, (0.0,))

    def test_integrate_rejects_negative_steps(self):
        integrator = RK4Integrator(lambda t, y: [0.0], [1.0])
        with self.assertRaises(ValueError):
            integrator.integrate(-1)

    def test_integrate_rejects_non_integer_steps(self):
        integrator = RK4Integrator(lambda t, y: [0.0], [1.0])
        with self.assertRaises(TypeError):
            integrator.integrate(1.5)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            integrator.integrate(True)  # type: ignore[arg-type]


class TestValidation(unittest.TestCase):
    def test_wrong_derivative_length_raises(self):
        integrator = RK4Integrator(lambda t, y: [0.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            integrator.step()

    def test_non_iterable_derivative_raises(self):
        integrator = RK4Integrator(lambda t, y: 1.0, [1.0])
        with self.assertRaises(TypeError):
            integrator.step()

    def test_non_numeric_initial_state_raises(self):
        with self.assertRaises(TypeError):
            RK4Integrator(lambda t, y: [0.0], ["a"])

    def test_derivative_receives_a_copy(self):
        seen = []

        def rhs(t, y):
            seen.append(y)
            y[0] = 123.0  # must not affect integrator state
            return [0.0]

        integrator = RK4Integrator(rhs, [1.0])
        integrator.step()
        self.assertEqual(integrator.y, (1.0,))
        # Every stage should have received its own list object.
        self.assertTrue(all(isinstance(s, list) for s in seen))


if __name__ == "__main__":
    unittest.main()
