import random
import unittest

import numpy as np

from solution import MultinomialLogReg, OrdinalLogReg

class HW2Tests(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1],
                           [1, 1]])
        self.y = np.array([0, 0, 1, 1, 2])
        self.train = self.X[::2], self.y[::2]
        self.test = self.X[1::2], self.y[1::2]

    def test_multinomial(self):
        l = MultinomialLogReg()
        c = l.build(self.train[0], self.train[1])
        prob = c.predict(self.test[0])
        print(prob)
        self.assertEqual(prob.shape, (2, 3))
        self.assertTrue((prob <= 1).all())
        self.assertTrue((prob >= 0).all())
        np.testing.assert_almost_equal(prob.sum(axis=1), 1)

    def test_ordinal(self):
        l = OrdinalLogReg()
        c = l.build(self.X, self.y)
        prob = c.predict(self.test[0])
        print(prob)
        self.assertEqual(prob.shape, (2, 3))
        self.assertTrue((prob <= 1).all())
        self.assertTrue((prob >= 0).all())
        np.testing.assert_almost_equal(prob.sum(axis=1), 1)


if __name__ == "__main__":
    unittest.main()
