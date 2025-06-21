import unittest
import tempfile

import parsac.util

FILE1 = """instances:
    name:
        parameters:
            dummy0: 0.0
            dummy1: 1.0
            dummy2: 2.0
"""


class TestInfer(unittest.TestCase):
    def test_dict(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            file = f.name
        yf = parsac.util.YAMLFile(file)
        self.assertIsNone(yf[""])
        with self.assertRaises(KeyError):
            _ = yf["dummy1"]
        with self.assertRaises(KeyError):
            yf["dummy1"] = 0.0

        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(FILE1)
            file = f.name
        yf = parsac.util.YAMLFile(file)
        self.assertIsInstance(yf[""], dict)
        self.assertEqual(yf["instances/name/parameters/dummy1"], 1.0)
        self.assertEqual(yf["/instances/name/parameters/dummy1"], 1.0)
        self.assertEqual(yf["//instances//name////parameters//dummy1"], 1.0)
        yf["instances/name/parameters/dummy1"] = 2.0
        self.assertEqual(yf["instances/name/parameters/dummy1"], 2.0)
        with self.assertRaises(KeyError):
            _ = yf["instances/name/parameters/dummy4"]
        yf["instances/name/parameters/dummy4"] = 4.0
        self.assertEqual(yf["instances/name/parameters/dummy4"], 4.0)
        with self.assertRaises(KeyError):
            yf["instances/name/parameters/dummy1/test"] = 0.0


if __name__ == "__main__":
    unittest.main()
