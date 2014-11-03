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
from dipy.segment.metric import HausdorffMetric, ArcLengthMetric
from dipy.viz import fvtk
from dipy.viz.colormap import line_colors
from time import time
from itertools import izip, chain
from dipy.segment.metric import SumPointwiseEuclideanMetric
from dipy.tracking.metrics import winding
from dipy.segment.metric import MidpointFeature
from dipy.viz.axycolor import distinguishable_colormap


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


def identify_left_right_middle_clusters(clusters):
    feature = EndpointsXFeature()

    left_clusters = []
    right_clusters = []
    middle_clusters = []

    for cluster in clusters:
        side = feature.extract(cluster[0])[0, 0]
        if side == 0:
            middle_clusters.append(cluster)
        elif side == 1:
            right_clusters.append(cluster)
        else:
            left_clusters.append(cluster)

    return left_clusters[0], right_clusters[0], middle_clusters[0]


def show_streamlines(streamlines, cam_pos=None, cam_focal=None, cam_view=None,
                     magnification=1, fname=None, size=(900, 900), axes=False):
    ren = fvtk.ren()
    if axes :
        fvtk.add(ren, fvtk.axes((100, 100, 100)))

    ren.SetBackground(1, 1, 1)
    fvtk.add(ren, fvtk.line(streamlines, line_colors(streamlines)))
    fvtk.show(ren, size=size)

    fvtk.record(ren, cam_pos=cam_pos, cam_focal=cam_focal, cam_view=cam_view,
                out_path=fname, path_numbering=False, n_frames=1, az_ang=10,
                magnification=magnification, size=size, verbose=True)


def show_centroids(clusters, colormap=None, cam_pos=None,
                   cam_focal=None, cam_view=None,
                   magnification=1, fname=None, size=(900, 900)):

    bg = (1, 1, 1)
    if colormap is None:
        colormap = distinguishable_colormap(bg=bg)

    ren = fvtk.ren()
    ren.SetBackground(*bg)

    max_cz = np.max(map(len, clusters))
    for cluster, color in izip(clusters, colormap):
            fvtk.add(ren, fvtk.line(cluster.centroid,
                                    color, linewidth=len(cluster)*10./float(max_cz)))

    fvtk.show(ren, size=size)
    if fname is not None:
        fvtk.record(ren, cam_pos=cam_pos, cam_focal=cam_focal, cam_view=cam_view,
                    out_path=fname, path_numbering=False, n_frames=1, az_ang=10,
                    magnification=magnification, size=size, verbose=True)


def show_clusters(clusters, colormap=None, cam_pos=None,
                  cam_focal=None, cam_view=None,
                  magnification=1, fname=None, size=(900, 900)):

    bg = (1, 1, 1)
    if colormap is None:
        colormap = distinguishable_colormap(bg=bg)

    ren = fvtk.ren()
    fvtk.clear(ren)
    ren.SetBackground(*bg)
    for cluster, color in izip(clusters, colormap):
        fvtk.add(ren, fvtk.line(list(cluster), [color]*len(cluster)))

    fvtk.show(ren, size=size)
    if fname is not None:
        fvtk.record(ren, cam_pos=cam_pos, cam_focal=cam_focal, cam_view=cam_view,
                    out_path=fname, path_numbering=False, n_frames=1, az_ang=10,
                    magnification=magnification, size=size, verbose=True)


def show_clusters_exploded_view(clusters, offsets=None, scale=500, colormap=None, cam_pos=None,
                                cam_focal=None, cam_view=None,
                                magnification=1, fname=None, size=(900, 900)):

    def uniform_spherical_distribution(N):
        import math
        pts = []
        inc = math.pi * (3 - math.sqrt(5))
        off = 2 / float(N)
        for k in range(0, int(N)):
            y = k * off - 1 + (off / 2)
            r = math.sqrt(1 - y*y)
            phi = k * inc
            pts.append([math.cos(phi)*r, y, math.sin(phi)*r])
        return np.array(pts)

    bg = (1, 1, 1)
    if colormap is None:
        colormap = distinguishable_colormap(bg=bg)

    if offsets is None:
        offsets = uniform_spherical_distribution(len(clusters))

    offsets = [scale * offset for offset in offsets]

    ren = fvtk.ren()
    fvtk.clear(ren)
    ren.SetBackground(*bg)
    for cluster, color, offset in izip(clusters, colormap, offsets):
        fvtk.add(ren, fvtk.line([s + offset for s in cluster], [color]*len(cluster)))

    fvtk.show(ren, size=size)

    if fname is not None:
        fvtk.record(ren, cam_pos=cam_pos, cam_focal=cam_focal, cam_view=cam_view,
                    out_path=fname, path_numbering=False, n_frames=1, az_ang=10,
                    magnification=magnification, size=size, verbose=True)


def remove_clusters_by_size(clusters, min_size=0, alpha=1):
    sizes = np.array(map(len, clusters))
    mean_size = sizes.mean()
    std_size = sizes.std()

    by_size = lambda c: len(c) >= min_size and len(c) >= mean_size - alpha * std_size
    return filter(by_size, clusters)

    # for cluster in cluster_map:
    #     if min_size is None:
    #         if len(cluster) >= mean_sz - alpha * std_sz:
    #             indices += cluster.indices.tolist()
    #     else:
    #         if len(cluster) >= size:
    #             indices += cluster.indices.tolist()

    # return indices


def remove_clusters_by_length(clusters, low, high):
    by_length = lambda c: low <= c.centroid and c.centroid <= high
    return filter(by_length, clusters)


def streamlines_from_indices_deprecated(streamlines, indices):
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


def remove_outlier_clusters(clusters, alpha=2):
    last_nb_clusters = None
    while last_nb_clusters != len(clusters):
        last_nb_clusters = len(clusters)
        sizes = np.array(map(len, clusters))
        mean_size = sizes.mean()
        std_size = sizes.std()

        by_size = lambda c: len(c) >= mean_size - alpha*std_size
        clusters = filter(by_size, clusters)

    return clusters


def mdf(streamlines, threshold, pts=None):
    if pts is not None:
        streamlines = set_number_of_points(streamlines, pts)

    qb = QuickBundles(threshold=threshold)

    t0 = time()
    cluster_map = qb.cluster(streamlines)
    cluster_map.refdata = streamlines
    print("QB-MDF duration: {:.2f} sec".format(time()-t0))

    return cluster_map.clusters


# class HausdorffMetric(Metric):
#     def dist(self, streamline1, streamline2):
#         max_d = 0.0
#         for a in streamline1:
#             min_d = np.inf
#             for b in streamline2:
#                 min_d = min(min_d, np.sum((a-b)**2))

#             max_d = max(max_d, min_d)

#         for b in streamline2:
#             min_d = np.inf
#             for a in streamline1:
#                 min_d = min(min_d, np.sum((a-b)**2))

#             max_d = max(max_d, min_d)

#         return max_d


def hausdorff(streamlines, threshold):
    qb = QuickBundles(metric=HausdorffMetric(), threshold=threshold)

    t0 = time()
    cluster_map = qb.cluster(streamlines)
    cluster_map.refdata = streamlines
    print("QB-Hausdorff duration: {:.2f} sec".format(time()-t0))

    return cluster_map.clusters


def full_brain_pipeline(streamlines):
    #show_streamlines(streamlines, fname='initial_full_brain.png')

    """
    Length
    """
    qb = QuickBundles(metric=ArcLengthMetric(), threshold=7.5)

    t0 = time()
    cluster_map = qb.cluster(streamlines)
    cluster_map.refdata = streamlines
    print("QB-Length duration: {:.2f} sec".format(time()-t0))

    show_clusters_exploded_view(cluster_map, fname='length_full_brain_clusters_exploded.png')
    clusters = remove_clusters_by_length(cluster_map, low=50, high=250)

    show_clusters(clusters, fname='length_full_brain_clusters.png')
    show_clusters_exploded_view(clusters, fname='length_full_brain_clusters_exploded.png')

    """
    L-R-M
    """
    streamlines = list(chain(*clusters))
    streamlines, shift_ = center_streamlines(streamlines)

    qb = QuickBundles(metric=LeftRightMiddleMetric(), threshold=0.5)

    t0 = time()
    cluster_map = qb.cluster(streamlines)
    cluster_map.refdata = streamlines
    print("QB-LRM duration: {:.2f} sec".format(time()-t0))

    clusters = cluster_map.clusters
    show_clusters(clusters, fname='LRM_full_brain_clusters.png')

    L, R, M = identify_left_right_middle_clusters(clusters)
    show_clusters_exploded_view([L, R, M], offsets=np.array([[-1, 0, 0], [1, 0, 0], [0, 0, 0]]), scale=200,
                           fname='LRM_full_brain_clusters_exploded.png')

    """
    MDF
    """
    clusters_L = mdf(L, threshold=20., pts=20)
    clusters_R = mdf(R, threshold=20., pts=20)
    clusters_M = mdf(M, threshold=20., pts=20)
    clusters = clusters_L + clusters_R + clusters_M

    print("Number of bundles before removing outliers: {0}".format(len(clusters)))
    clusters = remove_clusters_by_size(clusters, alpha=-1)
    print("Number of bundles: {0}".format(len(clusters)))

    show_centroids(clusters, fname='mdf_centroids.png')
    show_clusters(clusters, fname='MDF_full_brain_clusters.png')
    show_clusters_exploded_view(clusters, fname='MDF_full_brain_clusters_exploded.png')


def bundle_specific_pruning(streamlines, bundle_name='af'):
    show_streamlines(streamlines, fname=bundle_name + '_initial.png')

    """
    MDF
    """
    clusters = mdf(streamlines, threshold=20., pts=12)
    show_clusters(clusters)

    """
    Hausdorff
    """
    clusters = hausdorff(streamlines, threshold=20.)
    show_clusters(clusters)

    return

    """
    Length
    """
    qb = QuickBundles(metric=ArcLengthMetric(), threshold=20)

    t0 = time()

    cluster_map = qb.cluster(streamlines)

    print('QB-Length duration %f sec' % (time()-t0, ))

    colormap = np.random.rand(len(cluster_map.clusters), 3)

    show_clusters(streamlines, cluster_map.clusters, colormap, fname=bundle_name + '_length.png')

    print('Number of clusters %d' % (len(cluster_map),) )

    # TODO Historgram of lengths of streamlines

    """
    Midpoint
    """

    metric = SumPointwiseEuclideanMetric(MidpointFeature())

    qb = QuickBundles(metric=metric, threshold=15.)

    t0 = time()

    cluster_map = qb.cluster(streamlines)

    print('QB-midpoint duration %f sec' % (time()-t0, ))

    colormap = np.random.rand(len(cluster_map.clusters), 3)

    show_clusters(streamlines, cluster_map.clusters, colormap, fname=bundle_name + '_midpoint.png')

    print('Number of clusters %d' % (len(cluster_map),) )

    # TODO separate visualization clusters (IronMan stype - explode view)

    """
    Stem Detection
    """
    qb_pts(streamlines, threshold=10., fname=bundle_name + '_stem.png')


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

    qb = QuickBundles(metric=WindingAngleMetric(), threshold=50)

    t0 = time()

    cluster_map = qb.cluster(streamlines)

    print('Duration %f sec' % (time()-t0, ))

    colormap = np.random.rand(len(cluster_map), 3)

    show_clusters(streamlines, cluster_map.clusters, colormap,
                  fname=bundle_name + '_winding_angle.png')

    print('Number of clusters %d' % (len(cluster_map),) )
    print(map(len, cluster_map))
    print(np.squeeze(cluster_map.centroids))


def run_bundle_specific_pruning():
    #dname = '/home/eleftherios/Data/fancy_data/2013_03_26_Emmanuelle_Renauld/TRK_files/'
    dname = '/home/marc/research/data/streamlines/ismrm/'
    fname = dname + 'bundles_af.right.trk'
    #fname = dname + 'bundles_ifof.right.trk'
    #fname = dname + 'bundles_af.right.trk'

    streams, hdr = tv.read(fname, points_space='rasmm')

    streamlines = [i[0] for i in streams]
    streamlines = streamlines[:1000]

    bundle_specific_pruning(streamlines)


def run_full_brain_pipeline():
    #dname = '/home/eleftherios/Data/fancy_data/2013_02_26_Patrick_Delattre/'
    dname = '/home/marc/research/data/streamlines/ismrm/'
    fname =  dname + 'streamlines_500K.trk'

    # Load streamlines
    import os
    if not os.path.isfile('data.npy'):
        streams, hdr = tv.read(fname, points_space='rasmm')
        streamlines = [i[0] for i in streams]
        streamlines = streamlines[:1000]
        np.save('data.npy', streamlines)
    else:
        streamlines = np.load('data.npy')

    full_brain_pipeline(streamlines)

if __name__ == '__main__':
    np.random.seed(43)
    #run_full_brain_pipeline()
    run_bundle_specific_pruning()


