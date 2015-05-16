import numpy as np
from nibabel.affines import apply_affine
from dipy.tracking.streamlinespeed import set_number_of_points
from dipy.tracking.streamlinespeed import length


def unlist_streamlines(streamlines):
    """ Return the streamlines not as a list but as an array and an offset

    Parameters
    ----------
    streamlines: sequence

    Returns
    -------
    points : array
    offsets : array

    """

    points = np.concatenate(streamlines, axis=0)
    offsets = np.zeros(len(streamlines), dtype='i8')

    curr_pos = 0
    prev_pos = 0
    for (i, s) in enumerate(streamlines):

            prev_pos = curr_pos
            curr_pos += s.shape[0]
            points[prev_pos:curr_pos] = s
            offsets[i] = curr_pos

    return points, offsets


def relist_streamlines(points, offsets):
    """ Given a representation of a set of streamlines as a large array and
    an offsets array return the streamlines as a list of shorter arrays.

    Parameters
    -----------
    points : array
    offsets : array

    Returns
    -------
    streamlines: sequence
    """

    streamlines = []

    streamlines.append(points[0: offsets[0]])

    for i in range(len(offsets) - 1):
        streamlines.append(points[offsets[i]: offsets[i + 1]])

    return streamlines


def center_streamlines(streamlines):
    """ Move streamlines to the origin

    Parameters
    ----------
    streamlines : list
        List of 2D ndarrays of shape[-1]==3

    Returns
    -------
    new_streamlines : list
        List of 2D ndarrays of shape[-1]==3
    inv_shift : ndarray
        Translation in x,y,z to go back in the initial position

    """
    center = np.mean(np.concatenate(streamlines, axis=0), axis=0)
    return [s - center for s in streamlines], center


def transform_streamlines(streamlines, mat):
    """ Apply affine transformation to streamlines

    Parameters
    ----------
    streamlines : list
        List of 2D ndarrays of shape[-1]==3

    Returns
    -------
    new_streamlines : list
        List of the transformed 2D ndarrays of shape[-1]==3
    """

    return [apply_affine(mat, s) for s in streamlines]


def select_random_set_of_streamlines(streamlines, select):
    """ Select a random set of streamlines

    Parameters
    ----------
    streamlines : list
        List of 2D ndarrays of shape[-1]==3

    select : int
        Number of streamlines to select. If there are less streamlines
        than ``select`` then ``select=len(streamlines)``.

    Returns
    -------
    selected_streamlines : list
    """
    len_s = len(streamlines)
    index = np.random.randint(0, len_s, min(select, len_s))
    return [streamlines[i] for i in index]


def get_bounding_box_streamlines(streamlines):
    """ Returns the axis aligned bounding box (AABB) envlopping `streamlines`.

    Parameters
    ----------
    streamlines : list of 2D arrays
        Each 2D array represents a sequence of 3D points (nb_points, 3).

    Returns
    -------
    box_min : ndarray
        Coordinate of the bounding box corner having the minimum (X, Y, Z).
    box_max : ndarray
        Coordinate of the bounding box corner having the maximum (X, Y, Z).
    """
    box_min = np.array([np.inf, np.inf, np.inf])
    box_max = -np.array([np.inf, np.inf, np.inf])

    for s in streamlines:
        box_min = np.minimum(box_min, np.min(s, axis=0))
        box_max = np.maximum(box_max, np.max(s, axis=0))

    return box_min, box_max


def detect_outlier_in_streamlines(streamlines, confidence=0.95,
                                  min_threshold=1, nb_samplings_max=30):
    """ Detect streamlines considered as tractography outlier.

    Parameters
    ----------
    streamlines : list of 2D arrays
        Each 2D array represents a sequence of 3D points (nb_points, 3).

    confidence : float
        Coefficient of the confidence interval on the mean path length of a
        streamline in the clustering tree.

    min_threshold

    """
    sterror_factor = ndtri(confidence)
    metric = "mdf"

    box_min, box_max = get_streamlines_bounding_box(streamlines)
    #initial_threshold = np.sqrt(np.sum((box_max - box_min)**2)) / 4.  # Half of the bounding box's halved diagonal length.
    initial_threshold = np.min(np.abs(box_max - box_min)) / 2.

    # Quickbundle's threshold is halved between hierarchical level.
    thresholds = list(takewhile(lambda t: t >= min_threshold, (initial_threshold / 1.2**i for i in count())))

    start_time = time()
    ordering = np.arange(len(streamlines))
    nb_clusterings = 0

    streamlines_path = np.ones((len(streamlines), len(thresholds), nb_samplings_max), dtype=int) * -1
    for i in range(nb_samplings_max):
        np.random.shuffle(ordering)

        cluster_orderings = [ordering]
        for j, threshold in enumerate(thresholds):
            id_cluster = 0
            print "Ordering #{0}, QB/{2}mm, {1} clusters to process".format(i+1, len(cluster_orderings), threshold)

            next_cluster_orderings = []
            qb = QuickBundles(metric=metric, threshold=threshold)
            for cluster_ordering in cluster_orderings:
                clusters = qb.cluster(streamlines, ordering=cluster_ordering)
                nb_clusterings += 1

                for k, cluster in enumerate(clusters):
                    streamlines_path[cluster.indices, j, i] = id_cluster
                    id_cluster += 1
                    if len(cluster) > 10:
                        next_cluster_orderings.append(cluster.indices)

            cluster_orderings = next_cluster_orderings

        print "{} qb done in {:.2f} sec on {} streamlines".format(nb_clusterings, time()-start_time, len(streamlines))

        #path_lengths_per_streamline = np.sum(T[:, None]*(streamlines_path == -1), axis=1)
        path_lengths_per_streamline = np.sum((streamlines_path != -1), axis=1)[:, :i]

        # Compute confidence interval on mean cluster's size for each streamlines
        sterror_path_length_per_streamline = np.std(path_lengths_per_streamline, axis=1, ddof=1) / np.sqrt(i+1)
        print "Avg. sterror:", sterror_factor*sterror_path_length_per_streamline.mean()
        print "Max. sterror:", sterror_factor*sterror_path_length_per_streamline.max()

        if sterror_factor*sterror_path_length_per_streamline.mean() < 0.5:
            break

    summary = np.mean(path_lengths_per_streamline, axis=1) / np.max(path_lengths_per_streamline)
    return summary

