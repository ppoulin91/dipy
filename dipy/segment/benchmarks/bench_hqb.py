""" Benchmarks for QuickBundles

Run all benchmarks with::

    import dipy.segment as dipysegment
    dipysegment.bench()

If you have doctests enabled by default in nose (with a noserc file or
environment variable), and you have a numpy version <= 1.6.1, this will also run
the doctests, let's hope they pass.

Run this benchmark with:

    nosetests -s --match '(?:^|[\\b_\\.//-])[Bb]ench' /path/to/bench_quickbundles.py
"""
import numpy as np
import nibabel as nib

from dipy.data import get_data

import dipy.tracking.streamline as streamline_utils
from dipy.segment.metric import Metric
from dipy.segment.clustering import QuickBundles, HierarchicalQuickBundles
from nose.tools import assert_equal

from dipy.testing import assert_arrays_equal
from numpy.testing import assert_array_equal, measure


def bench_hqb():
    dtype = "float32"
    repeat = 10
    nb_points = 18

    streams, hdr = nib.trackvis.read(get_data('fornix'))
    fornix = [s[0].astype(dtype) for s in streams]
    fornix = streamline_utils.set_number_of_points(fornix, nb_points)

    #Create eight copies of the fornix to be clustered (one in each octant).
    streamlines = []
    streamlines += [s + np.array([100, 100, 100], dtype) for s in fornix]
    streamlines += [s + np.array([100, -100, 100], dtype) for s in fornix]
    streamlines += [s + np.array([100, 100, -100], dtype) for s in fornix]
    streamlines += [s + np.array([100, -100, -100], dtype) for s in fornix]
    streamlines += [s + np.array([-100, 100, 100], dtype) for s in fornix]
    streamlines += [s + np.array([-100, -100, 100], dtype) for s in fornix]
    streamlines += [s + np.array([-100, 100, -100], dtype) for s in fornix]
    streamlines += [s + np.array([-100, -100, -100], dtype) for s in fornix]

    # The expected number of clusters of the fornix using threshold=10 is 4.
    threshold = 10.
    #expected_nb_clusters = 4*8

    print("Timing Hierarhical QuickBundles")

    qb = QuickBundles(threshold)
    qb_time = measure("clusters = qb.cluster(streamlines)", repeat)
    print("QuickBundles time: {0:.4}sec".format(qb_time))

    hqb = HierarchicalQuickBundles(min_threshold=threshold)
    hqb_time = measure("clusters = hqb.cluster(streamlines)", repeat)
    print("Hierarchical QuickBundles time: {0:.4}sec".format(hqb_time))
    print("Speed up of {0}x".format(qb_time/hqb_time))

    clusters = qb.cluster(streamlines)
    hclusters = hqb.cluster(streamlines)
    print len(clusters), len(hclusters)

    #assert_equal(len(clusters), expected_nb_clusters)
    #assert_array_equal(sizes3, sizes1)
    #assert_arrays_equal(indices3, indices1)
