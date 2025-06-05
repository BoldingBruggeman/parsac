import unittest

import parsac
import parsac.optimize
import parsac.core
import numpy as np


class TestInfer(unittest.TestCase):
    def test_type(self):
        exp = parsac.optimize.Optimization()
        dummy1 = exp.add_parameter("dummy1", 0.0, 1.0)
        parsac.core.Parameter("dummy2").infer(lambda x: np.exp(x) * 0.00024, dummy1)
        exp.add_parameter("dummy3", 0.0, 1.0)
        for k, v in exp.unpack_parameters([0.5, np.exp(1.0)]).items():
            with self.subTest(k=k):
                self.assertEqual(type(v), float)
            self.assertEqual(type(v), float)


if __name__ == "__main__":
    unittest.main()
