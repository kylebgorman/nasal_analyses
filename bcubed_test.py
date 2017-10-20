"""Tests of the bcubed module.

This example is stolen from the LingPipe documentation:

    http://alias-i.com/lingpipe/docs/api/com/aliasi/cluster/ClusterScore.html

Our scores correspond to their scores using the element equal weighting scheme.
"""

import bcubed

from sklearn.metrics.cluster import normalized_mutual_info_score
import unittest


class BCubedTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    gld = (0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2)
    hyp = ("A", "A", "A", "A", "A", "B", "B", "A", "A", "A", "A", "A")
    assert len(gld) == len(hyp)
    cls.cs = bcubed.ClusterScore(gld, hyp)

  def testPrecision(self):
    self.assertAlmostEqual(0.5833, self.cs.precision(), places=4)

  def testRecall(self):
    self.assertAlmostEqual(1., self.cs.recall(), places=4)

  def testF1(self):
    self.assertAlmostEqual(0.7368, self.cs.f1(), places=4)


if __name__ == "__main__":
  unittest.main()
