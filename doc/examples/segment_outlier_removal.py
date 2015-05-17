# -*- coding: utf-8 -*-
from __future__ import division

import sys
import readline
import numpy as np
from time import time

from scipy.special import ndtri

import dipy.segment.metric as dipymetric
from dipy.tracking.streamline import set_number_of_points
from dipy.segment.clustering import QuickBundles, Cluster
from itertools import takewhile, count, izip, chain


def get_streamlines_bounding_box(streamlines):
    box_min = np.array([np.inf, np.inf, np.inf])
    box_max = -np.array([np.inf, np.inf, np.inf])

    for s in streamlines:
        box_min = np.minimum(box_min, np.min(s, axis=0))
        box_max = np.maximum(box_max, np.max(s, axis=0))

    return box_min, box_max


def show_clusters(clusters, colormap=None, cam_pos=None,
                  cam_focal=None, cam_view=None,
                  magnification=1, fname=None, size=(900, 900),
                  opacities=None):

    from dipy.viz import fvtk
    from dipy.viz.axycolor import distinguishable_colormap
    bg = (1, 1, 1)
    if colormap is None:
        colormap = distinguishable_colormap(bg=bg)

    if opacities is None:
        opacities = [1.]*len(clusters)

    ren = fvtk.ren()
    fvtk.clear(ren)
    ren.SetBackground(*bg)
    for cluster, color, opacity in izip(clusters, colormap, opacities):
        fvtk.add(ren, fvtk.line(list(cluster), [color]*len(cluster), linewidth=2, opacity=opacity))

    fvtk.show(ren, size=size, cam_pos=cam_pos, cam_focal=cam_focal, cam_view=cam_view)
    cam = ren.GetActiveCamera()
    return cam.GetPosition(), cam.GetFocalPoint(), cam.GetViewUp()


def show_clusters_grid_view(clusters, colormap=None, makelabel=None, grid_of_clusters=False,
                            cam_pos=None, cam_focal=None, cam_view=None,
                            magnification=1, fname=None, size=(900, 900)):

    from dipy.viz import fvtk
    from dipy.viz.axycolor import distinguishable_colormap

    def grid_distribution(N):
        height, width = (int(np.ceil(np.sqrt(N))), ) * 2  # Square
        X, Y, Z = np.meshgrid(np.arange(width), np.arange(height), [0])
        return np.array([X.flatten(), -Y.flatten(), Z.flatten()]).T

    bg = (1, 1, 1)
    if colormap is None:
        colormap = distinguishable_colormap(bg=bg)

    positions = grid_distribution(len(clusters))

    if grid_of_clusters:
        box_min, box_max = get_streamlines_bounding_box(chain(*chain(*clusters)))
    else:
        box_min, box_max = get_streamlines_bounding_box(chain(*clusters))

    ren = fvtk.ren()
    fvtk.clear(ren)
    ren.SetBackground(*bg)

    width, height, depth = box_max - box_min
    text_scale = [height*0.2]*3
    for cluster, color, pos in izip(clusters, colormap, positions):
        offset = pos * (box_max - box_min)
        offset[0] += pos[0] * 4*text_scale[0]
        offset[1] += pos[1] * 4*text_scale[1]

        if grid_of_clusters:
            for cc, color in izip(cluster, distinguishable_colormap(bg=bg)):
                fvtk.add(ren, fvtk.line([s + offset for s in cc], [color]*len(cc), linewidth=2))
        else:
            fvtk.add(ren, fvtk.line([s + offset for s in cluster], [color]*len(cluster), linewidth=2))

        if makelabel is not None:
            label = makelabel(cluster)
            text_pos = offset + np.array([-len(label)*text_scale[0]/2., height/2.+text_scale[1], 0])

            fvtk.label(ren, text=label, pos=text_pos, scale=text_scale, color=(0, 0, 0))

    fvtk.show(ren, size=size)


def prune_split(streamlines, thresholds, features):
    indices = np.arange(len(streamlines))
    clusters_indices = []

    last_threshold = 0
    for threshold in sorted(thresholds):
        idx = indices[np.bitwise_and(last_threshold <= features, features < threshold)]
        clusters_indices.append(idx)
        last_threshold = threshold

    idx = indices[last_threshold <= features]
    clusters_indices.append(idx)
    return clusters_indices


class Node(object):
    def __init__(self, cluster, threshold, children=None, parent=None):
        self.cluster = cluster
        self.threshold = threshold
        self.children = []
        self.parent = parent

        if children is not None:
            for child in children:
                self.add(child)

    @property
    def is_leaf(self):
        return len(self.children) == 0

    def add(self, child):
        self.children.append(child)


# TODO: make an iterative version instead of recurvise one
def traverse_postorder(node, visit):
    for child in node.children:
        traverse_postorder(child, visit)

    visit(node)


def show_clusters_tree_view(tree, theta_range=(0, np.pi), radius_scale=10, show_circles=False, cam_pos=None, cam_focal=None, cam_view=None, magnification=1, fname=None, size=(900, 900)):
    from dipy.viz import fvtk
    from dipy.viz.axycolor import distinguishable_colormap

    bg = (1, 1, 1)
    ren = fvtk.ren()
    fvtk.clear(ren)
    ren.SetBackground(*bg)

    box_min, box_max = get_streamlines_bounding_box(tree.cluster)
    width, height, depth = box_max - box_min
    radius_scale = 2*max(width, height, depth) / 1.5

    # Count nb_leaves per node
    def set_nb_leaves(node):
        if node.is_leaf:
            node.nb_leaves = 1
        else:
            node.nb_leaves = sum((child.nb_leaves for child in node.children))

    traverse_postorder(tree, set_nb_leaves)
    thresholds = set()

    # Suppose `root` is a list of lists (tree)
    def _draw_subtree(root, color=fvtk.colors.orange_red, theta_range=theta_range, parent_pos=(0, 0, 0)):
        print np.array(theta_range) / np.pi * 360

        # Draw root
        offset = np.zeros(3)
        theta = theta_range[0] + (theta_range[1] - theta_range[0]) / 2.
        radius = 0
        if root.parent is not None:
            radius = tree.threshold - root.threshold

        thresholds.add(root.threshold)

        offset[0] += radius*radius_scale * np.cos(theta)
        offset[1] -= radius*radius_scale * np.sin(theta)
        mean = np.mean([s.mean(0) for s in root.cluster], axis=0)
        fvtk.add(ren, fvtk.line([s + offset - mean for s in root.cluster], [color]*len(root.cluster), linewidth=2))
        fvtk.add(ren, fvtk.line(np.array([parent_pos, offset]), fvtk.colors.black, linewidth=1))

        if len(root.children) == 0:
            return

        children = sorted(root.children, key=lambda c: c.nb_leaves)
        ratios = np.maximum([c.nb_leaves / root.nb_leaves for c in children], 0.05)
        ratios = ratios / np.sum(ratios)
        sections = theta_range[0] + np.cumsum([0] + ratios.tolist()) * (theta_range[1] - theta_range[0])

        colormap = distinguishable_colormap(bg=bg)
        for i, (node, color) in enumerate(izip(children, colormap)):
            _draw_subtree(node, color, (sections[i], sections[i+1]), offset)

    _draw_subtree(tree)

    # Draw circles for the different radius
    if show_circles:
        for threshold in thresholds:
            theta = -np.linspace(*theta_range, num=200)
            radius = tree.threshold - threshold
            if radius > 0:
                X = radius*radius_scale * np.cos(theta)
                Y = radius*radius_scale * np.sin(theta)
                Z = np.zeros_like(X)
                dashed_line = zip(np.array([X, Y, Z]).T[::4], np.array([X, Y, Z]).T[1::4])
                fvtk.add(ren, fvtk.line(dashed_line, fvtk.colors.black, linewidth=1))

                scale = radius_scale/2.2
                text = "{:.1f}mm".format(threshold)
                pos = np.array([X[0], Y[0], Z[0]]) + np.array([-len(text)/2.*scale, scale/2., 0])
                fvtk.label(ren, text=text, pos=pos, scale=scale, color=(0, 0, 0))

                pos = np.array([X[-1], Y[-1], Z[-1]]) + np.array([-len(text)/2.*scale, scale/2., 0])
                fvtk.label(ren, text=text, pos=pos, scale=scale, color=(0, 0, 0))

    fvtk.show(ren, size=size)


def prune(streamlines, threshold, features, refdata=None):
    indices = np.arange(len(streamlines))

    outlier_indices = indices[features < threshold]
    rest_indices = indices[features >= threshold]

    if refdata is not None:
        # Redo QB for vizu on outliers and streamlines we keep
        qb_vizu = QuickBundles(threshold=5)
        clusters_outliers = qb_vizu.cluster(streamlines, ordering=outlier_indices, refdata=refdata)
        clusters_rest = qb_vizu.cluster(streamlines, ordering=rest_indices, refdata=refdata)

        outliers_cluster = Cluster(indices=outlier_indices, refdata=refdata)
        rest_cluster = Cluster(indices=rest_indices, refdata=refdata)

        # Show outliers streamlines vs. the ones we kept
        if len(outlier_indices) > 0 and len(rest_indices) > 0:
            show_clusters_grid_view([clusters_outliers, [outliers_cluster, rest_cluster], clusters_rest], grid_of_clusters=True)
        else:
            show_clusters_grid_view([clusters_outliers, clusters_rest], grid_of_clusters=True)

    return outlier_indices, rest_indices


def outliers_removal_using_hierarchical_quickbundles(streamlines, confidence=0.95, min_threshold=1, nb_samplings_max=30, seed=None, nb_thresholds=10):
    rng = np.random.RandomState(seed)
    sterror_factor = ndtri(confidence)
    #feature = ResampleFeature(nb_points=18)
    # We suppose streamlines have already been resampled
    metric = dipymetric.AveragePointwiseEuclideanMetric()

    box_min, box_max = get_streamlines_bounding_box(streamlines)
    #initial_threshold = np.sqrt(np.sum((box_max - box_min)**2)) / 4.  # Half of the bounding box's halved diagonal length.
    initial_threshold = np.min(np.abs(box_max - box_min)) / 2.

    # Quickbundle's threshold is halved between hierarchical level.
    #thresholds = list(takewhile(lambda t: t >= min_threshold, (initial_threshold / 1.2**i for i in count())))
    thresholds = np.logspace(np.log10(min_threshold), np.log10(initial_threshold), nb_thresholds)[::-1]
    #thresholds = np.linspace(min_threshold, initial_threshold, nb_thresholds)[::-1]

    start_time = time()
    ordering = np.arange(len(streamlines))
    nb_clusterings = 0

    streamlines_path = np.ones((len(streamlines), len(thresholds), nb_samplings_max), dtype=int) * -1
    for i in range(nb_samplings_max):
        rng.shuffle(ordering)
        #tree = Node(streamlines)

        cluster_orderings = [ordering]
        #nodes = [tree]
        for j, threshold in enumerate(thresholds):
            id_cluster = 0
            print "Ordering #{0}, QB/{2}mm, {1} clusters to process".format(i+1, len(cluster_orderings), threshold)

            next_cluster_orderings = []
            #next_nodes = []
            qb = QuickBundles(metric=metric, threshold=threshold)
            #for cluster_ordering, node in zip(cluster_orderings, nodes):
            for cluster_ordering in cluster_orderings:
                clusters = qb.cluster(streamlines, ordering=cluster_ordering)
                nb_clusterings += 1

                for k, cluster in enumerate(clusters):
                    #new_node = Node(cluster)
                    #node.add(new_node)

                    streamlines_path[cluster.indices, j, i] = id_cluster
                    id_cluster += 1
                    if len(cluster) > 10:
                        next_cluster_orderings.append(cluster.indices)
                        #next_nodes.append(new_node)

            cluster_orderings = next_cluster_orderings
            #nodes = next_nodes

        print "{} qb done in {:.2f} sec on {} streamlines".format(nb_clusterings, time()-start_time, len(streamlines))

        #path_lengths_per_streamline = np.sum(T[:, None]*(streamlines_path == -1), axis=1)
        path_lengths_per_streamline = np.sum((streamlines_path != -1), axis=1)[:, :i]

        # Compute confidence interval on mean cluster's size for each streamlines
        sterror_path_length_per_streamline = np.std(path_lengths_per_streamline, axis=1, ddof=1) / np.sqrt(i+1)
        print "Avg. sterror:", sterror_factor*sterror_path_length_per_streamline.mean()
        print "Max. sterror:", sterror_factor*sterror_path_length_per_streamline.max()

        if sterror_factor*sterror_path_length_per_streamline.mean() < 0.5:
            break

    #summary = np.mean(path_lengths_per_streamline, axis=1) / np.max(path_lengths_per_streamline)
    summary = np.mean(path_lengths_per_streamline, axis=1)
    summary /= summary.max()
    #return summary, tree, thresholds
    return summary


def outliers_removal_using_hierarchical_quickbundles_improved(streamlines, confidence=0.95, min_threshold=0.5, nb_samplings_max=30):
    sterror_factor = ndtri(confidence)
    #feature = ResampleFeature(nb_points=18)
    # We suppose streamlines have already been resampled
    metric = dipymetric.AveragePointwiseEuclideanMetric()
    streamlines = np.array(streamlines)

    start_time = time()
    ordering = np.arange(len(streamlines))
    nb_clusterings = 0
    divisor = 1.2

    streamlines_path = np.ones((len(streamlines), 100, nb_samplings_max), dtype=int) * -1
    for i in range(nb_samplings_max):
        np.random.shuffle(ordering)

        cluster_orderings = [ordering]

        box_min, box_max = get_streamlines_bounding_box(streamlines)
        threshold = np.min(np.abs(box_max - box_min))
        nodes = [Node(ordering, threshold=threshold)]
        j = 0
        while len(cluster_orderings) > 0:
        #for j, threshold in enumerate(thresholds):
            id_cluster = 0
            #print "Ordering #{0}, QB/{2}mm, {1} clusters to process".format(i+1, len(cluster_orderings), threshold)
            print "Ordering #{0}, {1} clusters to process".format(i+1, len(cluster_orderings))

            next_cluster_orderings = []
            next_nodes = []
            for cluster_ordering, node in zip(cluster_orderings, nodes):
                threshold = max(node.threshold, min_threshold)
                while threshold >= min_threshold:
                    threshold /= divisor
                    qb = QuickBundles(metric=metric, threshold=threshold)
                    clusters = qb.cluster(streamlines, ordering=cluster_ordering)
                    if len(clusters) > 1:
                        break

                if len(clusters) == 1:
                    continue

                nb_clusterings += 1

                for k, cluster in enumerate(clusters):
                    new_node = Node(cluster.indices, threshold=threshold)
                    node.add(new_node)

                    streamlines_path[cluster.indices, j, i] = id_cluster
                    id_cluster += 1
                    if len(cluster) > 1:
                        next_cluster_orderings.append(cluster.indices)
                        next_nodes.append(new_node)

            cluster_orderings = next_cluster_orderings
            nodes = next_nodes
            j += 1

        print "{} qb done in {:.2f} sec on {} streamlines".format(nb_clusterings, time()-start_time, len(streamlines))

        #path_lengths_per_streamline = np.sum(T[:, None]*(streamlines_path == -1), axis=1)
        path_lengths_per_streamline = np.sum((streamlines_path != -1), axis=1)[:, :i+1]

        # Compute confidence interval on mean cluster's size for each streamlines
        sterror_path_length_per_streamline = np.std(path_lengths_per_streamline, axis=1, ddof=1) / np.sqrt(i+1)
        print "Avg. sterror:", sterror_factor*sterror_path_length_per_streamline.mean()
        print "Max. sterror:", sterror_factor*sterror_path_length_per_streamline.max()

        if sterror_factor*sterror_path_length_per_streamline.mean() < 0.5:
            break

    #summary = np.mean(path_lengths_per_streamline, axis=1) / np.max(path_lengths_per_streamline)
    summary = np.mean(path_lengths_per_streamline, axis=1)
    summary /= summary.max()
    return summary


def outliers_removal_using_hierarchical_quickbundles_improved_proba(streamlines, confidence=0.95, min_threshold=0., nb_samplings_max=30):
    sterror_factor = ndtri(confidence)
    #feature = ResampleFeature(nb_points=18)
    # We suppose streamlines have already been resampled
    metric = dipymetric.AveragePointwiseEuclideanMetric()
    streamlines = np.array(streamlines)

    start_time = time()
    ordering = np.arange(len(streamlines))
    nb_clusterings = 0

    # QuickBundles threshold decreases as we go down in the hierarchy.
    reduction_factor = 3.1#0.5

    # Simple heuristic to determine the initial threshold, we take
    # half of the bounding box diagonal length.
    box_min, box_max = get_streamlines_bounding_box(streamlines)
    initial_threshold = 20.6#np.sqrt(np.sum((box_max - box_min)**2)) / 2.
    min_threshold = 2.  #TODO delete

    # Find root of the hierarchical quickbundles.
    while True:
        qb = QuickBundles(metric=metric, threshold=initial_threshold-reduction_factor)
        clusters = qb.cluster(streamlines, ordering=ordering)
        if len(clusters) > 1:
            break

        initial_threshold -= reduction_factor  # Linear reduction

    streamlines_prob = np.ones((len(streamlines), nb_samplings_max), dtype=float) * 0
    streamlines_path = np.ones((len(streamlines), 100, nb_samplings_max), dtype=int) * -1
    for i in range(nb_samplings_max):
        np.random.shuffle(ordering)

        cluster_orderings = [ordering]

        tree = Node(streamlines, threshold=initial_threshold)
        nodes = [tree]
        j = 0
        while len(cluster_orderings) > 0:
        #for j, threshold in enumerate(thresholds):
            id_cluster = 0
            #print "Ordering #{0}, QB/{2}mm, {1} clusters to process".format(i+1, len(cluster_orderings), threshold)
            print "Ordering #{0}, {1} clusters to process".format(i+1, len(cluster_orderings))

            next_cluster_orderings = []
            next_nodes = []
            for cluster_ordering, node in zip(cluster_orderings, nodes):

                threshold = max(node.threshold-reduction_factor, min_threshold)
                acc = 0
                previous_threshold = node.threshold
                while threshold >= min_threshold:
                    qb = QuickBundles(metric=metric, threshold=threshold)
                    clusters = qb.cluster(streamlines, ordering=cluster_ordering)
                    if len(clusters) > 1:
                        break

                    acc += len(cluster_ordering) * (previous_threshold - threshold)
                    previous_threshold = threshold
                    threshold -= reduction_factor  # Linear reduction

                if len(clusters) == 1:
                    continue

                nb_clusterings += 1

                for k, cluster in enumerate(clusters):
                    new_node = Node(cluster, threshold=threshold, parent=node)
                    node.add(new_node)

                    streamlines_prob[cluster.indices, i] += len(cluster) * (previous_threshold - threshold) + acc
                    streamlines_path[cluster.indices, j, i] = id_cluster
                    id_cluster += 1
                    if len(cluster) > 1:
                        next_cluster_orderings.append(cluster.indices)
                        next_nodes.append(new_node)

            cluster_orderings = next_cluster_orderings
            nodes = next_nodes
            j += 1

        print "{} qb done in {:.2f} sec on {} streamlines".format(nb_clusterings, time()-start_time, len(streamlines))

        #path_lengths_per_streamline = np.sum(T[:, None]*(streamlines_path == -1), axis=1)
        path_lengths_per_streamline = np.sum((streamlines_path != -1), axis=1)[:, :i+1]

        # Compute confidence interval on mean cluster's size for each streamlines
        sterror_path_length_per_streamline = np.std(path_lengths_per_streamline, axis=1, ddof=1) / np.sqrt(i+1)
        print "Avg. sterror:", sterror_factor*sterror_path_length_per_streamline.mean()
        print "Max. sterror:", sterror_factor*sterror_path_length_per_streamline.max()

        if sterror_factor*sterror_path_length_per_streamline.mean() < 0.5:
            break

    #summary2 = path_lengths_per_streamline / np.max(path_lengths_per_streamline, axis=1, keepdims=True)
    #summary2 = np.mean(summary2, axis=1)
    summary2 = np.mean(path_lengths_per_streamline, axis=1)
    summary2 /= summary2.max()

    summary = streamlines_prob / np.max(streamlines_prob, axis=1, keepdims=True)
    summary = np.mean(summary, axis=1)
    #summary = np.mean(streamlines_prob, axis=1)
    #summary /= summary.max()

    # import pylab as plt
    # plt.hist(summary, bins=100, alpha=0.5, label="proba")
    # plt.hist(summary2, bins=100, alpha=0.5, label="length")
    # plt.show(False)
    # plt.legend()

    return summary2, tree


def apply_on_specific_bundle(streamlines, confidence):
    from dipy.viz import fvtk
    import os
    import pickle

    #rng = np.random.RandomState(42)
    #offset = np.array([50, 50, 0])
    #variance = 5.3
    #streamlines = streamlines + [s + offset for s in streamlines]
    #streamlines = streamlines + [s + np.sqrt(variance)*rng.randn(*s.shape) for s in streamlines]

    cam_infos = (None, None, None)
    cam_infos_file = "/tmp/cam_infos.pkl"
    # if os.path.isfile(cam_infos_file):
    #     cam_infos = pickle.load(open(cam_infos_file))

    cam_infos = show_clusters([streamlines], [fvtk.colors.green], *cam_infos)
    pickle.dump(cam_infos, open(cam_infos_file, 'w'))

    rstreamlines = set_number_of_points(streamlines, 50)

    import pylab as plt
    # for nb_thresholds in [50, 20, 10]:
    #     summary = outliers_removal_using_hierarchical_quickbundles(rstreamlines, confidence=1, nb_samplings_max=10, nb_thresholds=nb_thresholds)
    #     plt.hist(summary, bins=100, alpha=0.5, label=str(nb_thresholds))

    #summary = outliers_removal_using_hierarchical_quickbundles_improved(rstreamlines, confidence=1, nb_samplings_max=10)
    #plt.hist(summary, bins=100, alpha=0.5, label="improved")
    summary, tree = outliers_removal_using_hierarchical_quickbundles_improved_proba(rstreamlines, confidence=confidence, nb_samplings_max=30)
    show_clusters_tree_view(tree, show_circles=True)
    from ipdb import set_trace as dbg
    dbg()

    plt.hist(summary, bins=100, alpha=0.5, label="proba")

    plt.show(False)
    plt.legend()

    show_clusters(zip(streamlines), fvtk.create_colormap(1-summary), *cam_infos)
    from ipdb import set_trace as dbg
    dbg()

    #summary_old, tree, thresholds = outliers_removal_using_hierarchical_quickbundles(rstreamlines, confidence=confidence, nb_samplings_max=2)
    #show_clusters_tree_view(tree, thresholds, show_circles=True)

    while True:
        print "{} ± {}".format(summary.mean(), summary.std())
        print "3sigma: {}".format(summary.mean() - 3*summary.std())
        print "---\nNew prunning threshold:",
        alpha = float(raw_input())

        outliers, rest = prune(rstreamlines, alpha, summary)
        print "Pruned {0} out of {1} streamlines at {2:.2f}%".format(len(outliers), len(streamlines), alpha*100)

        outliers_cluster = Cluster(indices=outliers, refdata=streamlines)
        rest_cluster = Cluster(indices=rest, refdata=streamlines)
        show_clusters_grid_view([rest_cluster, outliers_cluster], colormap=[fvtk.colors.green, fvtk.colors.orange_red])
        #show_clusters(zip(outliers_cluster))
        #show_clusters([rest_cluster, outliers_cluster], colormap=[fvtk.colors.green, fvtk.colors.orange_red], *cam_infos)
        show_clusters([rest_cluster], [fvtk.colors.turquoise_dark], *cam_infos)
        show_clusters([outliers_cluster], [fvtk.colors.orange_red], *cam_infos)
        show_clusters([rest_cluster, outliers_cluster], [fvtk.colors.turquoise_dark, fvtk.colors.orange_red], *cam_infos)
        show_clusters(zip(streamlines), fvtk.create_colormap(1-summary), *cam_infos)

def apply_on_specific_bundle_split(streamlines, confidence):
    #show_clusters([streamlines], colormap=[fvtk.colors.green])
    rstreamlines = set_number_of_points(streamlines, 20)

    summary, infos = outliers_removal_using_hierarchical_quickbundles(rstreamlines, confidence=confidence)
    #summary, infos = outliers_removal_using_hierarchical_quickbundles_improved(rstreamlines, confidence=confidence)
    #summary, infos = outliers_removal_using_hierarchical_quickbundles_improved_proba(rstreamlines, confidence=confidence)

    import pylab as plt
    plt.hist(summary, bins=100)
    plt.show(False)

    while True:
        print "---\nNew prunning threshold:",
        alphas = map(float, raw_input().split())

        clusters_indices = prune_split(rstreamlines, alphas, summary)
        clusters = []
        for i, idx in enumerate(clusters_indices):
            print "#{} - Pruned {} out of {} streamlines".format(1, len(idx), len(streamlines))

            clusters.append(Cluster(indices=idx, refdata=streamlines))

        makelabel = lambda c: str(len(c))
        show_clusters_grid_view(clusters, makelabel=makelabel)


def load_specific_bundle(bundlename):
    import nibabel as nib
    dname = '/home/marc/research/data/streamlines/ismrm/'
    fname = dname + 'bundles_{}.trk'.format(bundlename)

    streams, hdr = nib.trackvis.read(fname)
    return [i[0] for i in streams]


if __name__ == '__main__':
    np.random.seed(43)

    confidence = 0.95
    if len(sys.argv) > 2:
        confidence = float(sys.argv[2])

    streamlines = load_specific_bundle(sys.argv[1])

    mean = np.mean([s.mean(0) for s in streamlines], axis=0)
    streamlines = [s - mean for s in streamlines]

    # Rotation
    from dipy.core.geometry import rodrigues_axis_rotation
    rot_axis = np.array([1, 2, 3])
    M_rotation = rodrigues_axis_rotation([1, 0, 0], 270.)
    M_rotation = np.dot(rodrigues_axis_rotation([0, 1, 0], 270.), M_rotation)
    streamlines = [np.dot(M_rotation, s.T).T for s in streamlines]

    apply_on_specific_bundle(streamlines, confidence)
    exit()



    rstreamlines = set_number_of_points(streamlines, nb_points=20)

    from dipy.segment.clustering import HierarchicalQuickBundles
    from dipy.viz.streamline import show_hierarchical_clusters
    hqb = HierarchicalQuickBundles(metric="MDF", min_threshold=1.)
    hclusters = hqb.cluster(rstreamlines)
    show_hierarchical_clusters(hclusters, show_circles=True)
    exit()
    print len(hclusters.leaves)

    from dipy.segment.clustering import quickbundles_with_merging
    qb = QuickBundles(metric="MDF", threshold=10.)
    #clusters = qb.cluster(rstreamlines)
    clusters = quickbundles_with_merging(rstreamlines, qb)
    print len(clusters)

    from ipdb import set_trace as dbg
    dbg()

    show_hierarchical_clusters(hclusters, show_circles=True)
    exit()
