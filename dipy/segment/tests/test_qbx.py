import itertools
import numpy as np

from dipy.segment.clustering import QuickBundlesX
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.tracking.streamline import set_number_of_points
from dipy.data import get_data
import nibabel.trackvis as tv


def straight_bundle(nb_streamlines=1, nb_pts=30, step_size=1,
                    radius=1, rng=np.random.RandomState(42)):
    bundle = []

    bundle_length = step_size * nb_pts

    Z = -np.linspace(0, bundle_length, nb_pts)
    for k in range(nb_streamlines):
        theta = rng.rand() * (2*np.pi)
        r = radius * rng.rand()

        Xk = np.ones(nb_pts) * (r * np.cos(theta))
        Yk = np.ones(nb_pts) * (r * np.sin(theta))
        Zk = Z.copy()

        bundle.append(np.c_[Xk, Yk, Zk])

    return bundle


def bearing_bundles(nb_balls=6, bearing_radius=2):
    bundles = []

    for theta in np.linspace(0, 2*np.pi, nb_balls, endpoint=False):
        x = bearing_radius * np.cos(theta)
        y = bearing_radius * np.sin(theta)

        bundle = np.array(straight_bundle(nb_streamlines=100))
        bundle += (x, y, 0)
        bundles.append(bundle)

    return bundles


def streamlines_in_circle(nb_streamlines=1, nb_pts=30, step_size=1,
                          radius=1):
    bundle = []

    bundle_length = step_size * nb_pts

    Z = np.linspace(0, bundle_length, nb_pts)
    theta = 0
    for theta in np.linspace(0, 2*np.pi, nb_streamlines, endpoint=False):
        Xk = np.ones(nb_pts) * (radius * np.cos(theta))
        Yk = np.ones(nb_pts) * (radius * np.sin(theta))
        Zk = Z.copy()

        bundle.append(np.c_[Xk, Yk, Zk])

    return bundle


def streamlines_parallel(nb_streamlines=1, nb_pts=30, step_size=1,
                         delta=1):
    bundle = []

    bundle_length = step_size * nb_pts

    Z = np.linspace(0, bundle_length, nb_pts)
    for x in delta*np.arange(0, nb_streamlines):
        Xk = np.ones(nb_pts) * x
        Yk = np.zeros(nb_pts)
        Zk = Z.copy()

        bundle.append(np.c_[Xk, Yk, Zk])

    return bundle


def simulated_bundle(no_streamlines=10, waves=False, no_pts=12):
    t = np.linspace(-10, 10, 200)
    # parallel waves or parallel lines
    bundle = []
    for i in np.linspace(-5, 5, no_streamlines):
        if waves:
            pts = np.vstack((np.cos(t), t, i * np.ones(t.shape))).T
        else:
            pts = np.vstack((np.zeros(t.shape), t, i * np.ones(t.shape))).T
        pts = set_number_of_points(pts, no_pts)
        bundle.append(pts)

    return bundle


def fornix_streamlines(no_pts=12):
    fname = get_data('fornix')
    streams, hdr = tv.read(fname)
    streamlines = [set_number_of_points(i[0], no_pts) for i in streams]
    return streamlines


def test_3D_segments():
    points = np.array([[[1, 0, 0],
                        [1, 1, 0]],
                       [[3, 1, 0],
                        [3, 0, 0]],
                       [[2, 0, 0],
                        [2, 1, 0]],
                       [[5, 1, 0],
                        [5, 0, 0]],
                       [[5.5, 0, 0],
                        [5.5, 1, 0]]], dtype="f4")

    thresholds = [4, 2, 1]
    qbx_class = QuickBundlesX(thresholds)
    qbx = qbx_class.cluster(points)
    print qbx


def test_3D_points():
    points = np.array([[[1, 0, 0]],
                       [[3, 0, 0]],
                       [[2, 0, 0]],
                       [[5, 0, 0]],
                       [[5.5, 0, 0]]], dtype="f4")

    thresholds = [4, 2, 1]
    qbx_class = QuickBundlesX(thresholds)
    qbx = qbx_class.cluster(points)
    print qbx


def test_with_simulated_bundles():

    streamlines = simulated_bundle(3, False, 2)

    from dipy.viz import actor, window

    renderer = window.Renderer()
    bundle_actor = actor.line(streamlines)
    renderer.add(bundle_actor)

    window.show(renderer)

    thresholds = [10, 3, 1]
    qbx_class = QuickBundlesX(thresholds)
    qbx = qbx_class.cluster(streamlines)
    print qbx

    renderer.clear()

    for level in range(len(thresholds) + 1):
        clusters = qbx.get_clusters(level)
        clusters_actor = actor.line(clusters.centroids)
        renderer.add(clusters_actor)
        window.show(renderer)
        renderer.clear()

    from ipdb import set_trace
    set_trace()


def test_with_simulated_bundles2():
    # Generate synthetic streamlines
    bundles = bearing_bundles(4, 2)
    bundles.append(straight_bundle(1))
    streamlines = list(itertools.chain(*bundles))

    from dipy.viz import actor, window

    renderer = window.Renderer()
    bundle_actor = actor.line(streamlines)
    renderer.add(bundle_actor)

    window.show(renderer)

    thresholds = [10, 2, 1]
    qbx_class = QuickBundlesX(thresholds)
    print "Adding streamlines..."
    qbx = qbx_class.cluster(streamlines)

    renderer.clear()

    # for level in range(len(thresholds) + 1):
    #     clusters = qbx.get_clusters(level)
    #     clusters_actor = actor.line(clusters.centroids)
    #     renderer.add(clusters_actor)
    #     window.show(renderer)
    #     renderer.clear()

    from dipy.viz.clustering import show_hierarchical_clusters
    tree = qbx.get_tree_cluster_map()
    tree.refdata = streamlines
    show_hierarchical_clusters(tree, show_circles=True)

    from ipdb import set_trace
    set_trace()


def test_show_qbx():
    filename = "/home/marc/research/dat/streamlines/MPI_Camille/myBrain.trk"
    import nibabel as nib
    print "Loading streamlines..."

    import os
    tmp_filename = "/tmp/streamlines.npz"
    if os.path.isfile(tmp_filename):
        streamlines = nib.streamlines.compact_list.load_compact_list(tmp_filename)
    else:
        streamlines = nib.streamlines.load(filename).streamlines[::100]
        nib.streamlines.compact_list.save_compact_list(tmp_filename, streamlines)

    print "Displaying {} streamlines...".format(len(streamlines))
    #from dipy.viz import actor, window
    #renderer = window.Renderer()
    #bundle_actor = actor.line(streamlines)
    #renderer.add(bundle_actor)
    #window.show(renderer)

    thresholds = [40, 30, 25, 20, 15]
    qbx_class = QuickBundlesX(thresholds)
    print "Clustering {} streamlines ({})...".format(len(streamlines), thresholds)
    qbx = qbx_class.cluster(streamlines)

    print "Displaying clusters graph..."
    tree = qbx.get_tree_cluster_map()
    tree.refdata = streamlines
    from dipy.viz.clustering import show_clusters_graph
    show_clusters_graph(tree)

    from ipdb import set_trace as dbg
    dbg()

    clusters = qbx.get_clusters(len(thresholds))
    clusters.refdata = streamlines

    from dipy.viz.clustering import show_bundles
    print "Displaying {} clusters...".format(len(clusters))

    for level in range(1, len(thresholds) + 1):
        print thresholds[level-1]
        clusters = qbx.get_clusters(level)
        clusters.refdata = streamlines
        show_bundles(clusters)


if __name__ == '__main__':
    #test_with_simulated_bundles2()
    test_show_qbx()
