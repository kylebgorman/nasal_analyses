"""B-cubed statistics class.

This class implements b-cubed precision and recall statistics using the element
equal weighting scheme.

The basic idea of b-cubed is to apply precision, recall, etc. to evaluating
unsupervised clustering strategies with respect to some gold standard clusters.

For each element, we compute its gold equivalence class (the elements which
are assigned to the same gold cluster) and its hypothesis equivalence class 
(the elements which are assigned to the same hypothesis cluster), and then
use set intersection to compute true positives, and set differences to compute
false positives and false negatives. From these we compute a per-element
precision and/or recall, and then compute the arithmetic mean of these across
all elements.

This is designed for compatibility with the LingPipe ClusterScore class.
"""

from __future__ import division

import collections
import functools
import itertools
import numpy


class ClusterScore(object):
  """Scoring class for b-cubed clustering.

  Args:
    gld: An iterable of cluster labels for the gold clusters.
    hyp: An iterable of cluster labels for the hypothesized clusters, of equal
        length to, and corresponding in order to, the first argument.
  """

  @staticmethod
  def _partitions(iterable):
    partitions = collections.defaultdict(list)
    for (index, cluster) in enumerate(iterable):
      partitions[cluster].append(index)
    return frozenset(tuple(indices) for indices in partitions.itervalues())

  def __init__(self, gld, hyp):
    # The key is simply an integer index; the value is a [gld equivalence
    # frozenset, hyp equivalence frozenset] tuple.
    equivalences_table = {}
    gld_partitions = ClusterScore._partitions(gld)
    for indices in gld_partitions:
      for index in indices:
        equivalences_table[index] = [frozenset(indices), None]
    hyp_partitions = ClusterScore._partitions(hyp)
    for indices in hyp_partitions:
      for index in indices:
        equivalences_table[index][1] = frozenset(indices)
    # The final form is a tuple of (gld equivalence frozenset, hyp equivalence
    # frozenset) tuples.
    self._equivalences = tuple(tuple(pair) for pair in
                               equivalences_table.itervalues())

  def _true_positives(self):
    return numpy.array([len(gld_equivalence & hyp_equivalence) for
            (gld_equivalence, hyp_equivalence) in self._equivalences])

  def _false_positives(self):
    return numpy.array([len(hyp_equivalence - gld_equivalence) for
            (gld_equivalence, hyp_equivalence) in self._equivalences])

  def _false_negatives(self):
    return numpy.array([len(gld_equivalence - hyp_equivalence) for
            (gld_equivalence, hyp_equivalence) in self._equivalences])

  def precision(self):
    """Computes B-cubed precision.
    """
    tp = self._true_positives()
    fp = self._false_positives()
    return numpy.mean(tp / (tp + fp))

  def recall(self):
    """Computes B-cubed recall."""
    tp = self._true_positives()
    fn = self._false_negatives()
    return numpy.mean(tp / (tp + fn))

  def f1(self):
    """Computes B-cubed F1 score."""
    precision = self.precision()
    recall = self.recall()
    return (2. * precision * recall) / (precision + recall)

  # TODO(kbg): Implement a bunch of other metrics.
