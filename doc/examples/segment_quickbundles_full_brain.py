"""
============================================================================
Tractography Clustering with QuickBundles for Immense Full Brain Datasets
============================================================================

This example explains how we can use QuickBundles [Garyfallidis12]_ to
simplify/cluster streamlines.

First import the necessary modules.
"""

import numpy as np
from nibabel import trackvis as tv
from dipy.tracking.streamline import length, set_number_of_points, center_streamlines
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import Metric, MidpointFeature
from dipy.segment.metricspeed import ArcLengthMetric
from dipy.io.pickles import save_pickle
from dipy.viz import fvtk
from dipy.viz.colormap import line_colors
from time import time



def show_streamlines(streamlines):
    ren = fvtk.ren()
    fvtk.add(ren, fvtk.axes((100, 100, 100)))
    ren.SetBackground(1, 1, 1)
    #fvtk.add(ren, fvtk.line(streamlines, fvtk.colors.white))
    fvtk.add(ren, fvtk.line(streamlines, line_colors(streamlines)))
    fvtk.show(ren)
    #fvtk.record(ren, n_frames=1, out_path='full_brain_initial.png',
    #            size=(600, 600))


def show_centroids(centroids, colormap, clusters=None):
    ren = fvtk.ren()
    ren.SetBackground(1, 1, 1)
    if clusters is None:
        fvtk.add(ren, fvtk.line(centroids, colormap, linewidth=3.))
    else:
        cluster_sizes = np.array(map(len, clusters))
        max_cz = np.max(cluster_sizes)
        # min_cz = np.min(cluster_sizes)
        for cluster, color, cz in zip(clusters, colormap, cluster_sizes):
            fvtk.add(ren, fvtk.line(cluster.centroid,
                                    color, linewidth=cz*10./float(max_cz)))

    fvtk.show(ren)
    #fvtk.record(ren, n_frames=1, out_path='full_brain_centroids.png',
    #            size=(600, 600))


def show_clusters(streamlines, clusters, colormap):
    ren = fvtk.ren()
    colormap_full = np.ones((len(streamlines), 3))
    for i, cluster in enumerate(clusters):
        inds = cluster.indices
        for j in inds:
            colormap_full[j] = colormap[i]
    fvtk.clear(ren)
    ren.SetBackground(1, 1, 1)
    fvtk.add(ren, fvtk.line(streamlines, colormap_full))
    fvtk.show(ren)
    fvtk.record(ren, n_frames=1, out_path='full_brain_clust.png',
                size=(600, 600))


def remove_clusters(cluster_map, size):

    indices =[]
    for cluster in cluster_map:
        if len(cluster) >= size:
            indices += cluster.indices.tolist()
    return indices


def remove_clusters_by_length(cluster_map, length_range = (50, 300)):

    indices =[]
    for cluster in cluster_map:
        if cluster.centroid >= length_range[0] and cluster.centroid <= length_range[1]:
            indices += cluster.indices.tolist()
    return indices



def streamlines_from_indices(streamlines, indices):
    return [streamlines[i] for i in indices]


dname = '/home/eleftherios/Data/fancy_data/2013_02_26_Patrick_Delattre/'
fname =  dname + 'streamlines_500K.trk'

"""
Load full brain streamlines.
"""

streams, hdr = tv.read(fname, points_space='rasmm')

streamlines = [i[0] for i in streams]
streamlines = streamlines[:10000]

pts = 20

rstreamlines = set_number_of_points(streamlines, pts)

rstreamline, shift_ = center_streamlines(rstreamlines)

"""
Length
"""

qb = QuickBundles(metric=ArcLengthMetric(), threshold=7.5)

t0 = time()

cluster_map = qb.cluster(rstreamlines)

print('Duration %f sec' % (time()-t0, ))

#remove_clusters(cluster_map, size=100)
indices = remove_clusters_by_length(cluster_map, length_range = (50, 200))

streamlines = streamlines_from_indices(rstreamlines, indices)

"""
L-R-M
"""

class MidpointXFeature(MidpointFeature):


class LeftRightMiddleMetric(Metric):

    pass


"""
MDF
"""

qb = QuickBundles(threshold=20.)

t0 = time()

cluster_map = qb.cluster(streamlines)

print('Duration %f sec' % (time()-t0, ))

clusters = cluster_map.clusters
centroids = cluster_map.centroids

colormap = np.random.rand(len(clusters), 3)
#colormap = line_colors(centroids)


show_streamlines(streamlines)
show_centroids(centroids, colormap , clusters)

#show_centroids(centroids, colormap, clusters)
show_clusters(streamlines, clusters, colormap)


"""
.. include:: ../links_names.inc

.. [MarcCote14]

.. [Garyfallidis12] Garyfallidis E. et al., QuickBundles a method for
                    tractography simplification, Frontiers in Neuroscience, vol
                    6, no 175, 2012.

"""
