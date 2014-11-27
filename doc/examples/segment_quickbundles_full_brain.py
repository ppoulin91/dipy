# -*- coding: utf-8 -*-
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
from dipy.segment.metric import Metric, Feature, distance_matrix
from dipy.segment.clustering import Cluster
from dipy.segment.metric import ArcLengthMetric, MinimumAverageDirectFlipMetric
from dipy.viz import fvtk
from dipy.viz.colormap import line_colors
from time import time
from itertools import izip, chain
from dipy.segment.metric import SumPointwiseEuclideanMetric
from dipy.tracking.metrics import winding
from dipy.segment.metric import MidpointFeature
from dipy.viz.axycolor import distinguishable_colormap
from dipy.segment.quickbundles import bundles_distances_mdf
from scipy.special import ndtri


def quickbundles_with_merging(streamlines, qb, ordering=None):
    cluster_map = qb.cluster(streamlines, ordering=ordering)
    if len(streamlines) == len(cluster_map):
        return cluster_map.clusters

    clusters = quickbundles_with_merging(cluster_map.centroids, qb, None)

    merged_clusters = []
    for cluster in clusters:
        merged_cluster = Cluster()
        merged_cluster.refdata = cluster_map.refdata
        for i in cluster.indices:
            merged_cluster.indices.extend(cluster_map[i].indices)
        merged_clusters.append(merged_cluster)

    return merged_clusters


def automatic_outliers_removal(streamlines, qb, nb_samplings=100):
    start_time = time()
    cluster_maps = []
    cluster_sizes_per_ordering = []
    ordering = range(len(streamlines))
    for i in range(nb_samplings):
        np.random.shuffle(ordering)
        #clusters = quickbundles_with_merging(streamlines, qb, ordering)
        clusters = qb.cluster(streamlines, ordering=ordering)
        cluster_maps.append(clusters)
        cluster_sizes_per_ordering.append(map(len, clusters))

    #nb_clusters_per_ordering = np.array(map(len, cluster_maps))
    clusters_size_per_streamline = np.zeros((len(streamlines), nb_samplings))
    for i, cluster_map in enumerate(cluster_maps):
        for cluster in cluster_map:
            clusters_size_per_streamline[cluster.indices, i] += len(cluster)

    print "{} qb done in {:.2f} sec on {} streamlines".format(nb_samplings, time()-start_time, len(streamlines))

    # Compute confidence interval on mean cluster's size for each streamlines
    alpha = 0.05
    confidence_level = 1 - alpha
    sterror_factor = ndtri(confidence_level)

    mean_size_per_streamline = np.mean(clusters_size_per_streamline, axis=1)
    sterror = np.std(clusters_size_per_streamline, axis=1, ddof=1) / np.sqrt(nb_samplings)
    print "max sterror", sterror.min(), sterror.max()
    mean_size_bounds_per_streamline = np.array((mean_size_per_streamline - sterror_factor*sterror,
                                                mean_size_per_streamline + sterror_factor*sterror)).T

    mean_size_per_ordering = np.array([np.mean(cluster_sizes) for cluster_sizes in cluster_sizes_per_ordering])
    sterror = np.std(mean_size_per_ordering, ddof=1) / np.sqrt(nb_samplings)
    mean_size_bound = (np.mean(mean_size_per_ordering) - sterror_factor*sterror,
                       np.mean(mean_size_per_ordering) + sterror_factor*sterror)

    threshold = mean_size_bound[0]
    indices = np.arange(len(streamlines))

    cluster_outliers = Cluster()
    cluster_outliers.indices = indices[mean_size_bounds_per_streamline[:, 0] <= threshold]
    cluster_outliers.refdata = streamlines

    cluster_rest = Cluster()
    cluster_rest.indices = indices[mean_size_bounds_per_streamline[:, 0] > threshold]
    cluster_rest.refdata = streamlines

    show_clusters_grid_view([cluster_outliers, cluster_rest])

    #Redo a QB and show clusters
    rest_streamlines = list(cluster_rest)
    cmap = quickbundles_with_merging(rest_streamlines, qb)
    #cmap.refdata = rest_streamlines
    makelabel = lambda c: "{}".format(len(c))
    show_clusters_grid_view(cmap, makelabel=makelabel)

    outlier_streamlines = list(cluster_outliers)
    cmap = quickbundles_with_merging(outlier_streamlines, qb)
    #cmap.refdata = outlier_streamlines
    makelabel = lambda c: "{}".format(len(c))
    show_clusters_grid_view(cmap, makelabel=makelabel)

    from ipdb import set_trace as dbg
    dbg()

    return


    import pylab as plt
    plt.hist(np.mean(clusters_size_per_streamline, axis=1), bins=clusters_size_per_streamline.max())

    t = np.sqrt(nb_samplings) * np.mean(clusters_size_per_streamline, axis=1) / np.var(clusters_size_per_streamline, axis=1)
    plt.plot(t, 'o')
    #plt.plot(np.mean(clusters_size_per_streamline, axis=1), 'o')
    plt.show()

    threshold = 5
    indices = np.arange(len(streamlines))
    A = np.mean(clusters_size_per_streamline, axis=1)
    cluster_outliers = Cluster()
    cluster_outliers.indices = indices[A <= threshold]
    cluster_outliers.refdata = streamlines

    cluster_rest = Cluster()
    cluster_rest.indices = indices[A > threshold]
    cluster_rest.refdata = streamlines

    plt.hist(nb_clusters_per_ordering, bins=nb_clusters_per_ordering.max());plt.show()

    size_max = int(clusters_size_per_streamline.max())
    for sizes in clusters_size_per_streamline: plt.hist(sizes, bins=range(1, size_max+1));plt.show()
    for sizes in clusters_size_per_streamline.T: plt.hist(sizes, bins=range(1, size_max+1));plt.show()

    mean_size_per_streamlines = np.mean(clusters_size_per_streamline, axis=1)
    std_size_per_streamlines = np.std(clusters_size_per_streamline, axis=1)
    indices = np.argsort(mean_size_per_streamlines)

    plt.figure()
    plt.gca().set_xmargin(0.1)
    plt.errorbar(range(len(mean_size_per_streamlines)), mean_size_per_streamlines[indices], yerr=std_size_per_streamlines[indices], fmt='o')
    plt.ticklabel_format(useOffset=False, axis='y')
    plt.show()

    plt.plot(sort(np.mean(clusters_size_per_streamline, axis=1)), 'o'); plt.show()

    plt.hist(np.mean(clusters_size_per_streamline, axis=1), bins=range(1, size_max+1)); plt.show()
    plt.hist(np.std(clusters_size_per_streamline, axis=1), bins=range(1, size_max+1)); plt.show()
    plt.hist(np.mean(clusters_size_per_streamline, axis=0), bins=range(1, size_max+1)); plt.show()

    #hists = [np.histogram(sizes, bins=size_max) for sizes in cluster_size_per_ordering]
    #plt.hist(list(chain(*cluster_size_per_ordering)), bins=size_max); plt.show()

    from ipdb import set_trace as dbg
    dbg()

    rest_streamlines = list(cluster_rest)
    cmap = qb.cluster(rest_streamlines)
    cmap.refdata = rest_streamlines
    makelabel = lambda c: "{}".format(len(c))
    show_clusters_grid_view(cmap, makelabel=makelabel)

    show_clusters_grid_view([cluster_outliers, cluster_rest])



def get_bounding_box(streamlines):
    box_min = np.array([np.inf, np.inf, np.inf])
    box_max = -np.array([np.inf, np.inf, np.inf])

    for s in streamlines:
        box_min = np.minimum(box_min, np.min(s, axis=0))
        box_max = np.maximum(box_max, np.max(s, axis=0))

    return box_min, box_max


class EndpointsXFeature(Feature):

    def infer_shape(self, streamline):
        return 1

    def extract(self, streamline):
        x1 = streamline[0, 0]
        x2 = streamline[-1, 0]

        if x1 < 0 and x2 < 0:
            return -1

        if x1 > 0 and x2 > 0:
            return 1

        return 0


class LeftRightMiddleMetric(Metric):
    def __init__(self):
        super(LeftRightMiddleMetric, self).__init__(EndpointsXFeature())

    def compatible(self, shape1, shape2):
        return shape1 == shape2

    def dist(self, feature1, feature2):
        return 1 - np.float32(feature1 == feature2)


def bundle_adjacency(dtracks0, dtracks1, threshold):
    d01 = distance_matrix(MinimumAverageDirectFlipMetric(), dtracks0, dtracks1)
    #d01=bundles_distances_mdf(dtracks0,dtracks1)

    pair12=[]
    solo1=[]

    for i in range(len(dtracks0)):
        if np.min(d01[i,:]) < threshold:
            j=np.argmin(d01[i,:])
            pair12.append((i,j))
        else:
            solo1.append(dtracks0[i])

    pair12=np.array(pair12)
    pair21=[]

    solo2=[]
    for i in range(len(dtracks1)):
        if np.min(d01[:,i]) < threshold:
            j=np.argmin(d01[:,i])
            pair21.append((i,j))
        else:
            solo2.append(dtracks1[i])

    pair21=np.array(pair21)
    return 0.5*(len(pair12)/np.float(len(dtracks0))+len(pair21)/np.float(len(dtracks1)))


def identify_left_right_middle_clusters(clusters):
    feature = EndpointsXFeature()

    left_clusters = []
    right_clusters = []
    middle_clusters = []

    for cluster in clusters:
        side = feature.extract(cluster[0])
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


def show_stem(clusters, stem_radius, colormap=None, cam_pos=None,
              cam_focal=None, cam_view=None,
              magnification=1, fname=None, size=(900, 900)):

    bg = (1, 1, 1)
    if colormap is None:
        colormap = distinguishable_colormap(bg=bg)

    ren=fvtk.ren()
    fvtk.clear(ren)
    ren.SetBackground(*bg)
    cluster_sizes = map(len, clusters)
    maxc = np.max(cluster_sizes)

    for cluster, color in izip(clusters, colormap):
        fvtk.add(ren, fvtk.dots(np.squeeze(list(cluster)), color, opacity=0.2))

        #fvtk.add(ren, fvtk.point(centroid, fvtk.colors.green,
        #                         point_radius=5 * cluster_sizes[i]/float(maxc),
        #                         theta=3, phi=3))

        if len(cluster) == maxc:
            fvtk.add(ren, fvtk.point(cluster.centroid, color, opacity=0.5,
                                     point_radius=stem_radius))

    fvtk.show(ren, size=size)
    if fname is not None:
        fvtk.record(ren, cam_pos=cam_pos, cam_focal=cam_focal, cam_view=cam_view,
                    out_path=fname, path_numbering=False, n_frames=1, az_ang=10,
                    magnification=magnification, size=size, verbose=True)


def show_clusters_grid_view(clusters, colormap=None, makelabel=None,
                            cam_pos=None, cam_focal=None, cam_view=None,
                            magnification=1, fname=None, size=(900, 900)):

    def grid_distribution(N):
        def middle_divisors(n):
            for i in range(int(n ** (0.5)), 2, -1):
                if n % i == 0:
                    return i, n // i

            return middle_divisors(n+1)  # If prime number take next one

        height, width = middle_divisors(N)
        X, Y, Z = np.meshgrid(np.arange(width), np.arange(height), [0])
        return np.array([X.flatten(), Y.flatten(), Z.flatten()]).T

    bg = (1, 1, 1)
    if colormap is None:
        colormap = distinguishable_colormap(bg=bg)

    positions = grid_distribution(len(clusters))

    box_min, box_max = get_bounding_box(chain(*clusters))

    ren = fvtk.ren()
    fvtk.clear(ren)
    ren.SetBackground(*bg)

    width, height, depth = box_max - box_min
    text_scale = [height*0.1] * 3
    for cluster, color, pos in izip(clusters, colormap, positions):
        offset = pos * (box_max - box_min)
        offset[0] += pos[0] * 4*text_scale[0]
        offset[1] += pos[1] * 4*text_scale[1]
        fvtk.add(ren, fvtk.line([s + offset for s in cluster], [color]*len(cluster)))

        if makelabel is not None:
            label = makelabel(cluster)
            #text_scale = tuple([scale / 50.] * 3)
            text_pos = offset + np.array([0, height+4*text_scale[1], depth])/2.
            text_pos[0] -= len(label) / 2. * text_scale[0]

            fvtk.label(ren, text=label, pos=text_pos, scale=text_scale, color=(0, 0, 0))

    fvtk.show(ren, size=size)

    if fname is not None:
        fvtk.record(ren, cam_pos=cam_pos, cam_focal=cam_focal, cam_view=cam_view,
                    out_path=fname, path_numbering=False, n_frames=1, az_ang=10,
                    magnification=magnification, size=size, verbose=True)


def show_exploded_elef(list_of_clusters, offsets=None, scale=500, colormap=None, cam_pos=None,
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
        offsets = uniform_spherical_distribution(len(list_of_clusters))

    offsets = [scale * offset for offset in offsets]

    ren = fvtk.ren()
    fvtk.clear(ren)
    ren.SetBackground(*bg)
    for clusters, offset in izip(list_of_clusters, offsets):
        max_cz = np.max(map(len, clusters))

        for cluster, color in izip(clusters, colormap):
            fvtk.add(ren, fvtk.line(cluster.centroid + offset, color, linewidth=len(cluster)*10./float(max_cz)))

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


def recursive_quickbundles(streamlines, qb, alpha=None):
    cluster_map = qb.cluster(streamlines)
    if len(streamlines) == len(cluster_map):
        return cluster_map.clusters

    clusters = recursive_quickbundles(cluster_map.centroids, qb)

    merged_clusters = []
    for cluster in clusters:
        merged_cluster = Cluster()
        for i in cluster.indices:
            merged_cluster.indices.extend(cluster_map[i].indices)
        merged_clusters.append(merged_cluster)

    if alpha is not None:
        merged_clusters = remove_clusters_by_size(merged_clusters, alpha=alpha)

    return merged_clusters


def mdf(streamlines, threshold, pts=None):
    if pts is not None:
        streamlines = set_number_of_points(streamlines, pts)

    qb = QuickBundles(threshold=threshold)

    t0 = time()
    cluster_map = qb.cluster(streamlines)
    print("QB-MDF duration: {:.2f} sec".format(time()-t0))

    return cluster_map


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

    makelabel = lambda c: "{:.2f}mm".format(c.centroid[0, 0])
    #show_clusters(cluster_map)
    show_clusters_grid_view(cluster_map, makelabel=makelabel, fname='length_full_brain_clusters_grid.png')

    print "low: ",
    low = float(raw_input())
    print "high: ",
    high = float(raw_input())
    clusters = remove_clusters_by_length(cluster_map, low=low, high=high)

    show_clusters(clusters)
    show_clusters_grid_view(clusters, makelabel=makelabel, fname='length_full_brain_clusters_grid.png')

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
    clusters_L.refdata = L
    clusters_R = mdf(R, threshold=20., pts=20)
    clusters_R.refdata = R
    clusters_M = mdf(M, threshold=20., pts=20)
    clusters_M.refdata = M

    clusters = list(clusters_L) + list(clusters_R) + list(clusters_M)

    print("Number of bundles: {0}".format(len(clusters)))
    sizes = map(len, clusters)
    print "Sizes: {:.2f} ± {:.2f}".format(np.mean(sizes), np.std(sizes))

    show_centroids(clusters, fname='mdf_centroids.png')
    show_exploded_elef([clusters_L, clusters_R, clusters_M],
                       offsets=np.array([[-1, 0, 0], [1, 0, 0], [0, 0, 0]]),
                       scale=200, fname='mdf_centroids_exploded.png')

    show_clusters(clusters, fname='MDF_full_brain_clusters.png')
    show_clusters_exploded_view(clusters, fname='MDF_full_brain_clusters_exploded.png')

    makelabel = lambda c: "{}".format(len(c))
    show_clusters_grid_view(clusters, makelabel=makelabel, fname='MDF_full_brain_clusters_grid.png')

    print "Alpha: ",
    alpha = float(raw_input())

    clusters_L = remove_clusters_by_size(clusters_L, alpha=alpha)
    clusters_R = remove_clusters_by_size(clusters_R, alpha=alpha)
    clusters_M = remove_clusters_by_size(clusters_M, alpha=alpha)
    clusters = clusters_L + clusters_R + clusters_M


    print("Number of bundles: {0}".format(len(clusters)))
    sizes = map(len, clusters)
    print "Sizes: {:.2f} ± {:.2f}".format(np.mean(sizes), np.std(sizes))

    show_clusters_grid_view(clusters, makelabel=makelabel, fname='MDF_full_brain_clusters_grid.png')

    show_exploded_elef([clusters_L, clusters_R, clusters_M],
                       offsets=np.array([[-1, 0, 0], [1, 0, 0], [0, 0, 0]]),
                       scale=200, fname='mdf_centroids_exploded.png')

    show_clusters_exploded_view(clusters, fname='MDF_full_brain_clusters_exploded.png')
    show_clusters(clusters, fname='MDF_full_brain_clusters.png')


def bundle_specific_stats(streamlines, bundle_name='af'):

    rstreamlines = set_number_of_points(streamlines, 12)

    def plot_nb_clusters_vs_threshold(rstreamlines):
        nb_clusters_per_threshold = []
        clusters_size_per_threshold = []
        thresholds = np.linspace(0, 30, 100)
        for t in thresholds:
            cluster_map = mdf(rstreamlines, threshold=t)

            nb_clusters_per_threshold.append(len(cluster_map))
            clusters_size_per_threshold.append(map(len, cluster_map))

        import pylab as plt
        plt.plot(thresholds, nb_clusters_per_threshold, 'o')
        plt.show(False)

        means_clusters_size_per_threshold = map(np.mean, clusters_size_per_threshold)
        stds_clusters_size_per_threshold = map(np.std, clusters_size_per_threshold)

        plt.figure()
        plt.gca().set_xmargin(0.1)
        plt.errorbar(thresholds, means_clusters_size_per_threshold, yerr=stds_clusters_size_per_threshold, fmt='o')
        plt.ticklabel_format(useOffset=False, axis='y')
        plt.show()


    plot_nb_clusters_vs_threshold(rstreamlines)

    qb = QuickBundles(threshold=5.)
    merged_clusters = recursive_quickbundles(rstreamlines, qb)
    for c in merged_clusters:
        c.refdata = streamlines

    show_clusters_grid_view(merged_clusters)
    show_clusters(merged_clusters)


    merged_clusters = recursive_quickbundles(rstreamlines, qb, alpha=0.)
    for c in merged_clusters:
        c.refdata = streamlines

    show_clusters_grid_view(merged_clusters)
    show_clusters(merged_clusters)

    # clusters = remove_clusters_by_size(merged_clusters, alpha=0)
    # show_clusters(clusters)

    from ipdb import set_trace as dbg
    dbg()

    print len(cluster_map)
    cluster_map.refdata = streamlines
    show_clusters(cluster_map)
    #show_clusters_exploded_view(cluster_map, scale=200)
    show_clusters_grid_view(cluster_map)
    clusters = remove_clusters_by_size(cluster_map, alpha=0)
    print len(clusters)
    show_clusters_grid_view(clusters)


def visualize_impact_of_metric(streamlines, bundle_name='af'):
    show_streamlines(streamlines, fname=bundle_name + '_initial.png')

    """
    Length
    """
    qb = QuickBundles(metric=ArcLengthMetric(), threshold=7.5)

    t0 = time()
    cluster_map = qb.cluster(streamlines)
    cluster_map.refdata = streamlines
    print("QB-Length duration: {:.2f} sec".format(time()-t0))

    makelabel = lambda c: "{:.2f}mm".format(c.centroid[0, 0])
    show_clusters(cluster_map, fname=bundle_name +'_length_clusters.png')
    show_clusters_grid_view(cluster_map, makelabel=makelabel, fname=bundle_name +'_length_clusters_grid.png')


    """
    Stem
    """
    pts = [np.array([p]) for p in chain(*streamlines)]

    threshold = 10.
    qb = QuickBundles(metric=SumPointwiseEuclideanMetric(), threshold=threshold)
    t0 = time()
    cluster_map = qb.cluster(pts)
    cluster_map.refdata = pts
    print("QB-Stem duration: {:.2f} sec".format(time()-t0))

    show_stem(cluster_map, stem_radius=threshold, fname=bundle_name +'_stem.png')

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
    cluster_map.refdata = streamlines
    print("QB-Curvature duration: {:.2f} sec".format(time()-t0))

    makelabel = lambda c: "{:.2f}o".format(c.centroid[0, 0])
    show_clusters(cluster_map, fname=bundle_name +'_curvature_clusters.png')
    show_clusters_grid_view(cluster_map, makelabel=makelabel, fname=bundle_name +'_curvature_clusters_grid.png')


def bundle_pruning(streamlines, bundle_name='af'):
    show_streamlines(streamlines, fname=bundle_name + '_initial.png')

    """
    MDF
    """
    rstreamlines = set_number_of_points(streamlines, 12)

    threshold = 5.
    cluster_map = mdf(rstreamlines, threshold=threshold)
    cluster_map.refdata = streamlines

    print("Number of bundles: {0}".format(len(cluster_map)))
    sizes = map(len, cluster_map)
    print "Sizes: {:.2f} ± {:.2f}".format(np.mean(sizes), np.std(sizes))

    makelabel = lambda c: "{}".format(len(c))
    show_centroids(cluster_map, fname=bundle_name + '_mdf_centroids.png')
    show_clusters(cluster_map, fname=bundle_name + '_mdf_clusters.png')
    show_clusters_grid_view(cluster_map, makelabel=makelabel, fname=bundle_name + '_mdf_clusters_grid.png')

    qb = QuickBundles(threshold=threshold)
    clusters = recursive_quickbundles(rstreamlines, qb, alpha=0.)
    for c in clusters:
        c.refdata = streamlines

    print("Number of bundles: {0}".format(len(clusters)))
    sizes = map(len, clusters)
    print "Sizes: {:.2f} ± {:.2f}".format(np.mean(sizes), np.std(sizes))

    makelabel = lambda c: "{}".format(len(c))
    #show_centroids(clusters, fname=bundle_name + '_mdf_centroids_pruned.png')
    show_clusters(clusters, fname=bundle_name + '_mdf_clusters_pruned.png')
    show_clusters_grid_view(clusters, makelabel=makelabel, fname=bundle_name + '_mdf_clusters_grid_pruned.png')

    ordering = list(chain(*[c.indices for c in clusters]))
    cluster_map1 = qb.cluster(rstreamlines, ordering=ordering, refdata=streamlines)
    np.random.shuffle(ordering)
    cluster_map2 = qb.clusters(rstreamlines, ordering=ordering, refdata=streamlines)

    D = bundle_adjacency(cluster_map1.centroids, cluster_map2.centroids)

    from ipdb import set_trace as dbg
    dbg()
    return

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


def run_bundle_pruning(dname, bundle_name):
    fname = dname + 'bundles_' + bundle_name + '.trk'
    streams, hdr = tv.read(fname, points_space='rasmm')
    streamlines = [i[0] for i in streams]

    bundle_pruning(streamlines)


def run_visualize_impact_of_metric(dname, bundle_name):
    fname = dname + 'bundles_' + bundle_name + '.trk'
    streams, hdr = tv.read(fname, points_space='rasmm')
    streamlines = [i[0] for i in streams]

    visualize_impact_of_metric(streamlines, bundle_name)


def run_bundle_specific_stats():
    #dname = '/home/eleftherios/Data/fancy_data/2013_03_26_Emmanuelle_Renauld/TRK_files/'
    dname = '/home/marc/research/data/streamlines/ismrm/'
    fname = dname + 'bundles_af.right.trk'
    #fname = dname + 'bundles_cc_1.trk'
    #fname = dname + 'bundles_ifof.right.trk'

    streams, hdr = tv.read(fname, points_space='rasmm')
    streamlines = [i[0] for i in streams]

    #bundle_specific_stats(streamlines)

    rstreamlines = set_number_of_points(streamlines, 12)
    qb = QuickBundles(threshold=5.)
    cluster_map = qb.cluster(rstreamlines)
    print len(cluster_map)

    #automatic_outliers_removal(rstreamlines, qb, nb_samplings=100)


def run_full_brain_pipeline(dname):
    fname =  dname + 'streamlines_500K.trk'

    # Load streamlines
    import os
    if not os.path.isfile('data.npy'):
        streams, hdr = tv.read(fname, points_space='rasmm')
        streamlines = [i[0] for i in streams]
        streamlines = streamlines[:10000]
        np.save('data.npy', streamlines)
    else:
       streamlines = np.load('data.npy').tolist()

    full_brain_pipeline(streamlines)


if __name__ == '__main__':
    np.random.seed(43)
    #dname = '/home/eleftherios/Data/fancy_data/2013_03_26_Emmanuelle_Renauld/TRK_files/'
    dname = '/home/marc/research/data/streamlines/ismrm/'

    #run_visualize_impact_of_metric(dname, 'af.right')
    #run_bundle_pruning(dname, 'af.right')
    run_full_brain_pipeline(dname)

    #run_bundle_specific_stats()
