import unittest

import parsac
import parsac.optimize
import parsac.core
import numpy as np


class TestInfer(unittest.TestCase):
    def test_type(self):
        exp = parsac.optimize.Optimization()
        dummy1 = exp.add_parameter("test", 0.0, 1.0)
        parsac.core.Parameter("dummy2").infer(lambda x: np.exp(x) * 0.00024, dummy1)
        for k, v in exp.unpack_parameters([0.5]).items():
            self.assertEqual(type(v), float)


if __name__ == "__main__":
    unittest.main()
