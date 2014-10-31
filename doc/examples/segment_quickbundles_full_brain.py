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
from dipy.tracking.streamline import set_number_of_points, center_streamlines
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import Metric, Feature
from dipy.segment.metricspeed import ArcLengthMetric
from dipy.viz import fvtk
from dipy.viz.colormap import line_colors
from time import time
from itertools import chain
from dipy.segment.metric import SumPointwiseEuclideanMetric
from dipy.tracking.metrics import winding
from dipy.segment.metric import MidpointFeature

class EndpointsXFeature(Feature):

    def infer_shape(self, streamline):
        return (1, 1)

    def extract(self, streamline):
        x1 = streamline[0, 0]
        x2 = streamline[-1, 0]

        if x1 < 0 and x2 < 0:
            return np.array([[-1]])

        if x1 > 0 and x2 > 0:
            return np.array([[1]])

        return np.array([[0]])


class LeftRightMiddleMetric(Metric):

    def __init__(self):
        super(LeftRightMiddleMetric, self).__init__(EndpointsXFeature())

    def dist(self, feature1, feature2):

        return 1 - np.float32(feature1 == feature2)


def identify_left_right_middle(streamlines, cluster_map):
    feature = EndpointsXFeature()

    left_streamlines = []
    right_streamlines = []
    middle_streamlines = []

    for cluster in cluster_map:
        side = feature.extract(streamlines[cluster[0]])[0, 0]
        if side == 0:
            middle_streamlines.extend([streamlines[i] for i in cluster.indices])
        elif side == 1:
            right_streamlines.extend([streamlines[i] for i in cluster.indices])
        else:
            left_streamlines.extend([streamlines[i] for i in cluster.indices])

    return left_streamlines, right_streamlines, middle_streamlines


def show_streamlines(streamlines, cam_pos=None, cam_focal=None, cam_view=None,
                     magnification=1, fname=None, size=(900, 900), axes=False):
    ren = fvtk.ren()
    if axes :
        fvtk.add(ren, fvtk.axes((100, 100, 100)))
    ren.SetBackground(1, 1, 1)
    #fvtk.add(ren, fvtk.line(streamlines, fvtk.colors.white))
    fvtk.add(ren, fvtk.line(streamlines, line_colors(streamlines)))
    fvtk.show(ren, size=size)

    fvtk.record(ren, cam_pos=cam_pos, cam_focal=cam_focal, cam_view=cam_view,
                out_path=fname, path_numbering=False, n_frames=1, az_ang=10,
                magnification=magnification, size=size, verbose=True)


def show_centroids(centroids, colormap, clusters=None, cam_pos=None,
                   cam_focal=None, cam_view=None,
                   magnification=1, fname=None, size=(900, 900)):

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

    fvtk.show(ren, size=size)

    fvtk.record(ren, cam_pos=cam_pos, cam_focal=cam_focal, cam_view=cam_view,
                out_path=fname, path_numbering=False, n_frames=1, az_ang=10,
                magnification=magnification, size=size, verbose=True)


def show_clusters(streamlines, clusters, colormap, cam_pos=None,
                  cam_focal=None, cam_view=None,
                  magnification=1, fname=None, size=(900, 900)):

    ren = fvtk.ren()
    colormap_full = np.ones((len(streamlines), 3))
    for i, cluster in enumerate(clusters):
        inds = cluster.indices
        for j in inds:
            colormap_full[j] = colormap[i]
    fvtk.clear(ren)
    ren.SetBackground(1, 1, 1)
    fvtk.add(ren, fvtk.line(streamlines, colormap_full))
    fvtk.show(ren, size=size)

    fvtk.record(ren, cam_pos=cam_pos, cam_focal=cam_focal, cam_view=cam_view,
                out_path=fname, path_numbering=False, n_frames=1, az_ang=10,
                magnification=magnification, size=size, verbose=True)


def remove_clusters(cluster_map, size=None, alpha=1):

    indices =[]
    szs = np.array(map(len, cluster_map))
    mean_sz = szs.mean()
    std_sz = szs.std()

    for cluster in cluster_map:
        if size is None:
            if len(cluster) >= mean_sz - alpha * std_sz:
                indices += cluster.indices.tolist()
        else:
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


def qb_mdf(streamlines, threshold, disp=False):

    qb = QuickBundles(threshold=threshold)

    t0 = time()

    cluster_map = qb.cluster(streamlines)

    print('Duration %f sec' % (time()-t0, ))

    if disp:

        clusters = cluster_map.clusters
        centroids = cluster_map.centroids

        colormap = np.random.rand(len(clusters), 3)
        #colormap = line_colors(centroids)

        show_streamlines(streamlines)
        show_centroids(centroids, colormap , clusters)

        #show_centroids(centroids, colormap, clusters)
        show_clusters(streamlines, clusters, colormap)

    return cluster_map


def qb_pts(streamlines, threshold=10., cam_pos=None,
           cam_focal=None, cam_view=None,
           magnification=1, fname=None, size=(900, 900)):

    pts = [np.array([p]) for p in chain(*streamlines)]

    qb = QuickBundles(metric=SumPointwiseEuclideanMetric(),
                      threshold=threshold)
    cluster_map = qb.cluster(pts)

    ren=fvtk.ren()
    pts = np.squeeze(np.array(pts))
    cluster_sizes = map(len, cluster_map)
    maxc = np.max(cluster_sizes)

    colormap = np.random.rand(len(cluster_map), 3)

    for cluster, c in zip(cluster_map, colormap):
        fvtk.add(ren, fvtk.dots(pts[cluster.indices], c, opacity=0.2))

        #fvtk.add(ren, fvtk.point(centroid, fvtk.colors.green,
        #                         point_radius=5 * cluster_sizes[i]/float(maxc),
        #                         theta=3, phi=3))

        if len(cluster) == maxc:
            fvtk.add(ren, fvtk.point(cluster.centroid, c, opacity=0.5,
                                     point_radius=threshold))

    fvtk.show(ren, size=size)
    fvtk.record(ren, cam_pos=cam_pos, cam_focal=cam_focal, cam_view=cam_view,
                out_path=fname, path_numbering=False, n_frames=1, az_ang=10,
                magnification=magnification, size=size, verbose=True)


def full_brain_pipeline(streamlines):

    show_streamlines(streamlines, fname='initial_full_brain.png')

    pts = 20

    rstreamlines = set_number_of_points(streamlines, pts)

    rstreamline, shift_ = center_streamlines(rstreamlines)

    """
    Length
    """

    qb = QuickBundles(metric=ArcLengthMetric(), threshold=7.5)

    t0 = time()

    cluster_map = qb.cluster(rstreamlines)

    print('QB-Length duration %f sec' % (time()-t0, ))

    #remove_clusters(cluster_map, size=100)
    indices = remove_clusters_by_length(cluster_map, length_range = (50, 250))

    streamlines = streamlines_from_indices(rstreamlines, indices)

    show_streamlines(streamlines, fname='after_length_full_brain.png')

    #colormap = np.random.rand(len(cluster_map.clusters), 3)

    #show_clusters(streamlines, cluster_map.clusters, colormap,
    #              fname='after_length_clusters.png')

    """
    L-R-M
    """

    qb = QuickBundles(metric=LeftRightMiddleMetric(), threshold=0.5)

    t0 = time()

    cluster_map = qb.cluster(streamlines)

    print('QB-LRM duration %f sec' % (time()-t0, ))

    print(len(cluster_map))

    colormap = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.]])


    show_clusters(streamlines, cluster_map.clusters, colormap,
                  fname='LRM_clusters.png')

    L, R, M = identify_left_right_middle(streamlines, cluster_map)

    """
    MDF
    """

    streamlines = L

    qb = QuickBundles(threshold=20.)

    t0 = time()

    cluster_map = qb.cluster(streamlines)

    print('MDF duration %f sec' % (time()-t0, ))

    clusters = cluster_map.clusters
    centroids = cluster_map.centroids

    colormap = np.random.rand(len(clusters), 3)
    #colormap = line_colors(centroids)

    show_centroids(centroids, colormap , clusters, fname='mdf_centroids.png')

    show_clusters(streamlines, clusters, colormap, fname='mdf_clusters.png')

    indices = remove_clusters(cluster_map, alpha=-1)
    streamlines = streamlines_from_indices(streamlines, indices)

    show_streamlines(streamlines, fname='remove_clusters.png')

    print('Number of bundles %d' % (len(cluster_map),))


def bundle_specific_pruning(streamlines, bundle_name='af'):

#    show_streamlines(streamlines, fname=bundle_name + '_initial.png')
#
#    """
#    Length
#    """
#
#    qb = QuickBundles(metric=ArcLengthMetric(), threshold=20)
#
#    t0 = time()
#
#    cluster_map = qb.cluster(streamlines)
#
#    print('QB-Length duration %f sec' % (time()-t0, ))
#
#    colormap = np.random.rand(len(cluster_map.clusters), 3)
#
#    show_clusters(streamlines, cluster_map.clusters, colormap, fname=bundle_name + '_length.png')
#
#    print('Number of clusters %d' % (len(cluster_map),) )
#
#    # TODO Historgram of lengths of streamlines
#
#    """
#    Midpoint
#    """
#
#    metric = SumPointwiseEuclideanMetric(MidpointFeature())
#
#    qb = QuickBundles(metric=metric, threshold=15.)
#
#    t0 = time()
#
#    cluster_map = qb.cluster(streamlines)
#
#    print('QB-midpoint duration %f sec' % (time()-t0, ))
#
#    colormap = np.random.rand(len(cluster_map.clusters), 3)
#
#    show_clusters(streamlines, cluster_map.clusters, colormap, fname=bundle_name + '_midpoint.png')
#
#    print('Number of clusters %d' % (len(cluster_map),) )
#
#    # TODO separate visualization clusters (IronMan stype - explode view)
#
#    """
#    Stem Detection
#    """
#    qb_pts(streamlines, threshold=10., fname=bundle_name + '_stem.png')


    """
    Winding angle - projected sum curvature - total turning angle projected
    """

    class WindingAngleFeature(Feature):
        def infer_shape(self, streamline):
            return (1, 1)

        def extract(self, streamline):
            return np.array([[winding(streamline)]])

    class WindingAngleMetric(Metric):

        def __init__(self):
            super(WindingAngleMetric, self).__init__(WindingAngleFeature())

        def dist(self, w1, w2):
            return np.abs(w1 - w2)

    qb = QuickBundles(metric=WindingAngleMetric(), threshold=15)

    t0 = time()

    cluster_map = qb.cluster(streamlines)

    print('Duration %f sec' % (time()-t0, ))

    colormap = np.random.rand(len(cluster_map), 3)

    show_clusters(streamlines, cluster_map.clusters, colormap,
                  fname=bundle_name + '_winding_angle.png')

    print('Number of clusters %d' % (len(cluster_map),) )
    print(map(len, cluster_map))
    print(np.squeeze(cluster_map.centroids))

if __name__ == '__main__':


    np.random.seed(43)

    dname = '/home/eleftherios/Data/fancy_data/2013_02_26_Patrick_Delattre/'
    fname =  dname + 'streamlines_500K.trk'

    #
    #
    #dname = '/home/eleftherios/Data/fancy_data/2013_02_28_Zhara_Owji/TRK_files/'
    #fname = dname + 'bundles_ifof.right.trk'
    ##fname = dname + 'bundles_af.right.trk'
    #
    dname = '/home/eleftherios/Data/fancy_data/2013_03_26_Emmanuelle_Renauld/TRK_files/'
    fname = dname + 'bundles_af.right.trk'

    streams, hdr = tv.read(fname, points_space='rasmm')

    streamlines = [i[0] for i in streams]
    streamlines = streamlines[:100000]

    bundle_specific_pruning(streamlines)

    """
    Load full brain streamlines.




    full_brain_pipeline(streamlines)
    """

