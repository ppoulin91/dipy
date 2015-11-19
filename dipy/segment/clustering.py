import operator
import numpy as np

from abc import ABCMeta, abstractmethod
from dipy.segment.metric import Metric
from dipy.segment.metric import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric

from dipy.segment.metric import metric_factory
from dipy.tracking.streamline import get_bounding_box_streamlines


class Identity:
    """ Provides identity indexing functionality.

    This can replace any class supporting indexing used for referencing
    (e.g. list, tuple). Indexing an instance of this class will return the
    index provided instead of the element. It does not support slicing.
    """
    def __getitem__(self, idx):
        return idx


class Cluster(object):
    """ Provides functionalities for interacting with a cluster.

    Useful container to retrieve index of elements grouped together. If
    a reference to the data is provided to `cluster_map`, elements will
    be returned instead of their index when possible.

    Parameters
    ----------
    cluster_map : `ClusterMap` object
        Reference to the set of clusters this cluster is being part of.
    id : int
        Id of this cluster in its associated `cluster_map` object.
    refdata : list (optional)
        Actual elements that clustered indices refer to.

    Notes
    -----
    A cluster does not contain actual data but instead knows how to
    retrieve them using its `ClusterMap` object.
    """
    def __init__(self, id=0, indices=None, refdata=Identity()):
        self.id = id
        self.refdata = refdata
        self.indices = indices if indices is not None else []

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """ Gets element(s) through indexing.

        If a reference to the data was provided (via refdata property)
        elements will be returned instead of their index.

        Parameters
        ----------
        idx : int, slice or list
            Index of the element(s) to get.

        Returns
        -------
        `Cluster` object(s)
            When `idx` is a int, returns a single element.

            When `idx` is either a slice or a list, returns a list of elements.
        """
        if isinstance(idx, int) or isinstance(idx, np.integer):
            return self.refdata[self.indices[idx]]
        elif type(idx) is slice:
            return [self.refdata[i] for i in self.indices[idx]]
        elif type(idx) is list:
            return [self[i] for i in idx]

        raise TypeError("Index must be a int or a slice! Not " + str(type(idx)))

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __str__(self):
        return "[" + ", ".join(map(str, self.indices)) + "]"

    def __repr__(self):
        return "Cluster(" + str(self) + ")"

    def __eq__(self, other):
        return isinstance(other, Cluster) and self.indices == other.indices

    def __ne__(self, other):
        return not self == other

    def __cmp__(self, other):
        raise TypeError("Cannot compare Cluster objects.")

    def assign(self, *indices):
        """ Assigns indices to this cluster.

        Parameters
        ----------
        *indices : list of indices
            Indices to add to this cluster.
        """
        self.indices += indices


class ClusterCentroid(Cluster):
    """ Provides functionalities for interacting with a cluster.

    Useful container to retrieve the indices of elements grouped together and
    the cluster's centroid. If a reference to the data is provided to
    `cluster_map`, elements will be returned instead of their index when
    possible.

    Parameters
    ----------
    cluster_map : `ClusterMapCentroid` object
        Reference to the set of clusters this cluster is being part of.
    id : int
        Id of this cluster in its associated `cluster_map` object.
    refdata : list (optional)
        Actual elements that clustered indices refer to.

    Notes
    -----
    A cluster does not contain actual data but instead knows how to
    retrieve them using its `ClusterMapCentroid` object.
    """
    def __init__(self, centroid, id=0, indices=None, refdata=Identity()):
        super(ClusterCentroid, self).__init__(id, indices, refdata)
        self.centroid = centroid.copy()
        self.new_centroid = centroid.copy()

    def __eq__(self, other):
        return isinstance(other, ClusterCentroid) \
            and np.all(self.centroid == other.centroid) \
            and super(ClusterCentroid, self).__eq__(other)

    def assign(self, id_datum, features):
        """ Assigns a data point to this cluster.

        Parameters
        ----------
        id_datum : int
            Index of the data point to add to this cluster.
        features : 2D array
            Data point's features to modify this cluster's centroid.
        """
        N = len(self)
        self.new_centroid = ((self.new_centroid * N) + features) / (N+1.)
        super(ClusterCentroid, self).assign(id_datum)

    def update(self):
        """ Update centroid of this cluster.

        Returns
        -------
        converged : bool
            Tells if the centroid has moved.
        """
        converged = np.equal(self.centroid, self.new_centroid)
        self.centroid = self.new_centroid.copy()
        return converged


class HierarchicalCluster(Cluster):
    def __init__(self, cluster, threshold, parent=None):
        self._cluster = cluster
        self.threshold = threshold
        self.parent = parent
        self.children = []

    def add(self, child):
        self.children.append(child)

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def indices(self):
        return self._cluster.indices

    def __getitem__(self, idx):
        """ Gets element(s) through indexing. """
        return self._cluster[idx]


class ClusterMap(object):
    """ Provides functionalities for interacting with clustering outputs.

    Useful container to create, remove, retrieve and filter clusters.
    If `refdata` is given, elements will be returned instead of their
    index when using `Cluster` objects.

    Parameters
    ----------
    refdata : list
        Actual elements that clustered indices refer to.
    """
    def __init__(self, refdata=Identity()):
        self._clusters = []
        self.refdata = refdata

    @property
    def clusters(self):
        return self._clusters

    @property
    def refdata(self):
        return self._refdata

    @refdata.setter
    def refdata(self, value):
        if value is None:
            value = Identity()

        self._refdata = value
        for cluster in self.clusters:
            cluster.refdata = self._refdata

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, idx):
        """ Gets cluster(s) through indexing.

        Parameters
        ----------
        idx : int, slice, list or boolean array
            Index of the element(s) to get.

        Returns
        -------
        `Cluster` object(s)
            When `idx` is a int, returns a single `Cluster` object.

            When `idx`is either a slice, list or boolean array, returns
            a list of `Cluster` objects.
        """
        if isinstance(idx, np.ndarray) and idx.dtype == np.bool:
            return [self.clusters[i] for i, take_it in enumerate(idx) if take_it]
        elif type(idx) is slice:
            return [self.clusters[i] for i in range(*idx.indices(len(self)))]
        elif type(idx) is list:
            return [self.clusters[i] for i in idx]

        return self.clusters[idx]

    def __iter__(self):
        return iter(self.clusters)

    def __str__(self):
        return "[" + ", ".join(map(str, self)) + "]"

    def __repr__(self):
        return "ClusterMap(" + str(self) + ")"

    def _richcmp(self, other, op):
        """ Compares this cluster map with another cluster map or an integer.

        Two `ClusterMap` objects are equal if they contain the same clusters.
        When comparing a `ClusterMap` object with an integer, the comparison
        will be performed on the size of the clusters instead.

        Parameters
        ----------
        other : `ClusterMap` object or int
            Object to compare to.
        op : rich comparison operators (see module `operator`)
            Valid operators are: lt, le, eq, ne, gt or ge.

        Returns
        -------
        bool or 1D array (bool)
            When comparing to another `ClusterMap` object, it returns whether
            the two `ClusterMap` objects contain the same clusters or not.

            When comparing to an integer the comparison is performed on the
            clusters sizes, it returns an array of boolean.
        """
        if isinstance(other, ClusterMap):
            if op is operator.eq:
                return isinstance(other, ClusterMap) \
                    and len(self) == len(other) \
                    and self.clusters == other.clusters
            elif op is operator.ne:
                return not self == other

            raise NotImplementedError("Can only check if two ClusterMap instances are equal or not.")

        elif isinstance(other, int):
            return np.array([op(len(cluster), other) for cluster in self])

        raise NotImplementedError("ClusterMap only supports comparison with a int or another instance of Clustermap.")

    def __eq__(self, other):
        return self._richcmp(other, operator.eq)

    def __ne__(self, other):
        return self._richcmp(other, operator.ne)

    def __lt__(self, other):
        return self._richcmp(other, operator.lt)

    def __le__(self, other):
        return self._richcmp(other, operator.le)

    def __gt__(self, other):
        return self._richcmp(other, operator.gt)

    def __ge__(self, other):
        return self._richcmp(other, operator.ge)

    def add_cluster(self, *clusters):
        """ Adds one or multiple clusters to this cluster map.

        Parameters
        ----------
        *clusters : `Cluster` object, ...
            Cluster(s) to be added in this cluster map.
        """
        for cluster in clusters:
            self.clusters.append(cluster)
            cluster.refdata = self.refdata

    def remove_cluster(self, *clusters):
        """ Remove one or multiple clusters from this cluster map.

        Parameters
        ----------
        *clusters : `Cluster` object, ...
            Cluster(s) to be removed from this cluster map.
        """
        for cluster in clusters:
            self.clusters.remove(cluster)

    def clear(self):
        """ Remove all clusters from this cluster map. """
        del self.clusters[:]

    def get_size(self):
        """ Gets number of clusters contained in this cluster map. """
        return len(self)

    def get_clusters_sizes(self):
        """ Gets the size of every clusters contained in this cluster map.

        Returns
        -------
        list of int
            Sizes of every clusters in this cluster map.
        """
        return list(map(len, self))

    def get_large_clusters(self, min_size):
        """ Gets clusters which contains at least `min_size` elements.

        Parameters
        ----------
        min_size : int
            Minimum number of elements a cluster needs to have to be selected.

        Returns
        -------
        list of `Cluster` objects
            Clusters having at least `min_size` elements.
        """
        return self[self >= min_size]

    def get_small_clusters(self, max_size):
        """ Gets clusters which contains at most `max_size` elements.

        Parameters
        ----------
        max_size : int
            Maximum number of elements a cluster can have to be selected.

        Returns
        -------
        list of `Cluster` objects
            Clusters having at most `max_size` elements.
        """
        return self[self <= max_size]


class ClusterMapCentroid(ClusterMap):
    """ Provides functionalities for interacting with clustering outputs
    that have centroids.

    Allows to retrieve easely the centroid of every clusters. Also, it is
    a useful container to create, remove, retrieve and filter clusters.
    If `refdata` is given, elements will be returned instead of their
    index when using `ClusterCentroid` objects.

    Parameters
    ----------
    refdata : list
        Actual elements that clustered indices refer to.
    """
    @property
    def centroids(self):
        return [cluster.centroid for cluster in self.clusters]


class HierarchicalClusterMap(ClusterMap):
    def __init__(self, root):
        self.root = root
        self.leaves = []

        def _retrieves_leaves(node):
            if node.is_leaf:
                self.leaves.append(node)

        self.traverse_postorder(self.root, _retrieves_leaves)

    @property
    def refdata(self):
        return self._refdata

    @refdata.setter
    def refdata(self, value):
        if value is None:
            value = Identity()

        self._refdata = value

        def _set_refdata(node):
            node.refdata = self._refdata

        self.traverse_postorder(self.root, _set_refdata)

    def traverse_postorder(self, node, visit):
        for child in node.children:
            self.traverse_postorder(child, visit)

        visit(node)

    def iter_preorder(self, node):
        parent_stack = []
        while len(parent_stack) > 0 or node is not None:
            if node is not None:
                yield node
                if len(node.children) > 0:
                    parent_stack += node.children[1:]
                    node = node.children[0]
                else:
                    node = None
            else:
                node = parent_stack.pop()

    def __iter__(self):
        return self.iter_preorder(self.root)


class Clustering(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def cluster(self, data, ordering=None):
        """ Clusters `data`.

        Subclasses will perform their clustering algorithm here.

        Parameters
        ----------
        data : list of N-dimensional arrays
            Each array represents a data point.
        ordering : iterable of indices, optional
            Specifies the order in which data points will be clustered.

        Returns
        -------
        `ClusterMap` object
            Result of the clustering.
        """
        raise NotImplementedError("Subclass has to define method 'cluster(data, ordering)'!")


class QuickBundles(Clustering):
    r""" Clusters streamlines using QuickBundles [Garyfallidis12]_.

    Given a list of streamlines, the QuickBundles algorithm sequentially
    assigns each streamline to its closest bundle in $\mathcal{O}(Nk)$ where
    $N$ is the number of streamlines and $k$ is the final number of bundles.
    If for a given streamline its closest bundle is farther than `threshold`,
    a new bundle is created and the streamline is assigned to it except if the
    number of bundles has already exceeded `max_nb_clusters`.

    Parameters
    ----------
    threshold : float
        The maximum distance from a bundle for a streamline to be still
        considered as part of it.
    metric : str or `Metric` object (optional)
        The distance metric to use when comparing two streamlines. By default,
        the Minimum average Direct-Flip (MDF) distance [Garyfallidis12]_ is
        used and streamlines are automatically resampled so they have 12 points.
    max_nb_clusters : int
        Limits the creation of bundles.

    Examples
    --------
    >>> from dipy.segment.clustering import QuickBundles
    >>> from dipy.data import get_data
    >>> from nibabel import trackvis as tv
    >>> streams, hdr = tv.read(get_data('fornix'))
    >>> streamlines = [i[0] for i in streams]
    >>> # Segment fornix with a treshold of 10mm and streamlines resampled to 12 points.
    >>> qb = QuickBundles(threshold=10.)
    >>> clusters = qb.cluster(streamlines)
    >>> len(clusters)
    4
    >>> list(map(len, clusters))
    [61, 191, 47, 1]
    >>> # Resampling streamlines differently is done explicitly as follows.
    >>> # Note this has an impact on the speed and the accuracy (tradeoff).
    >>> from dipy.segment.metric import ResampleFeature
    >>> from dipy.segment.metric import AveragePointwiseEuclideanMetric
    >>> feature = ResampleFeature(nb_points=2)
    >>> metric = AveragePointwiseEuclideanMetric(feature)
    >>> qb = QuickBundles(threshold=10., metric=metric)
    >>> clusters = qb.cluster(streamlines)
    >>> len(clusters)
    4
    >>> list(map(len, clusters))
    [58, 142, 72, 28]


    References
    ----------
    .. [Garyfallidis12] Garyfallidis E. et al., QuickBundles a method for
                        tractography simplification, Frontiers in Neuroscience,
                        vol 6, no 175, 2012.
    """
    def __init__(self, threshold, metric="MDF_12points", max_nb_clusters=np.iinfo('i4').max):
        self.threshold = threshold
        self.max_nb_clusters = max_nb_clusters
        self.metric = metric_factory(metric)

    def cluster(self, streamlines, ordering=None):
        """ Clusters `streamlines` into bundles.

        Performs quickbundles algorithm using predefined metric and threshold.

        Parameters
        ----------
        streamlines : list of 2D arrays
            Each 2D array represents a sequence of 3D points (points, 3).
        ordering : iterable of indices
            Specifies the order in which data points will be clustered.

        Returns
        -------
        `ClusterMapCentroid` object
            Result of the clustering.
        """
        from dipy.segment.clustering_algorithms import quickbundles
        cluster_map = quickbundles(streamlines, self.metric,
                                   threshold=self.threshold,
                                   max_nb_clusters=self.max_nb_clusters,
                                   ordering=ordering)

        cluster_map.refdata = streamlines
        return cluster_map


def quickbundles_with_merging(streamlines, qb, ordering=None):
    cluster_map = qb.cluster(streamlines, ordering=ordering)
    if len(streamlines) == len(cluster_map):
        return cluster_map

    qb_for_merging = QuickBundles(metric=qb.metric, threshold=qb.threshold)
    clusters = quickbundles_with_merging(cluster_map.centroids, qb_for_merging, None)

    merged_clusters = ClusterMapCentroid()
    for cluster in clusters:
        merged_cluster = ClusterCentroid(centroid=cluster.centroid)

        for i in cluster.indices:
            merged_cluster.indices.extend(cluster_map[i].indices)

        merged_clusters.add_cluster(merged_cluster)

    merged_clusters.refdata = cluster_map.refdata
    return merged_clusters


class HierarchicalQuickBundles(Clustering):
    r""" Clusters streamlines using hierarchical QuickBundles.

    Hierarchical QuickBundles is a divisive approach where all streamlines start
    in one cluster and splits are performed recursively using QuickBundles
    [Garyfallidis12]_ (see ``QuickBundles``).

    More specifically, QuickBundles is first run with a large threshold to
    obtain a set of clusters of streamlines. Then, it is applied again on each
    of those clusters with a smaller threshold. The process is repeated until
    either the threshold has reached `min_threhold` or every streamlines belong
    to a cluster with less or equal than `min_cluster_size`.

    Parameters
    ----------
    metric : str or `Metric` object (optional)
        The distance metric to use when comparing two streamlines. By default,
        the Minimum average Direct-Flip (MDF) distance [Garyfallidis12]_ is
        used and requires streamlines to have the same number of points.
    min_threshold : float
        Algorithm stop when the moving threshold reaches this value.
    min_cluster_size : int
        Algorithm stop when the size of every clusters reach this value.

    References
    ----------
    .. [Garyfallidis12] Garyfallidis E. et al., QuickBundles a method for
                        tractography simplification, Frontiers in Neuroscience,
                        vol 6, no 175, 2012.
    """
    def __init__(self, metric="MDF_12points", min_threshold=0, min_cluster_size=1):
        self.min_threshold = min_threshold
        self.min_cluster_size = min_cluster_size
        self.metric = metric_factory(metric)

    def cluster(self, streamlines, ordering=None):
        """ Clusters `streamlines` into a hierarchy of bundles.

        Performs hierarchical quickbundles algorithm using predefined metric.

        Parameters
        ----------
        streamlines : list of 2D arrays
            Each 2D array represents a sequence of 3D points (nb_points, 3).
        ordering : iterable of indices
            Specifies the order in which data points will be clustered.

        Returns
        -------
        `HierarchicalClusterMap` object
            Result of the clustering.
        """
        # QuickBundles threshold decreases as we go down in the hierarchy.
        #reduction_factor = 0.2  # TODO: explore what would be an optimal reduction scheme
        reduction_factor = 2  # TODO: explore what would be an optimal reduction scheme

        # Simple heuristic to determine the initial threshold, we take
        # the bounding box diagonal length.
        box_min, box_max = get_bounding_box_streamlines(streamlines)
        threshold = np.sqrt(np.sum((box_max - box_min)**2))

        # Find the tightest root of the hierarchical quickbundles.
        while True:
            qb = QuickBundles(metric=self.metric, threshold=threshold)
            clusters = qb.cluster(streamlines, ordering=ordering)
            #clusters = quickbundles_with_merging(streamlines, qb, ordering=ordering)
            if len(clusters) > 1:
                break

            root = HierarchicalCluster(clusters[0], threshold=threshold)
            #threshold -= reduction_factor  # Linear reduction
            threshold /= reduction_factor  # Exponential reduction

        nodes = [root]
        while len(nodes) > 0:
            next_nodes = []
            for node in nodes:
                clusters = []
                threshold = max(node.threshold-reduction_factor, self.min_threshold)
                indices = node.indices
                #np.random.shuffle(indices)
                while threshold >= self.min_threshold:
                    qb = QuickBundles(metric=self.metric, threshold=threshold)
                    clusters = qb.cluster(streamlines, ordering=indices)
                    #clusters = quickbundles_with_merging(streamlines, qb, ordering=indices)
                    if len(clusters) > 1:
                        break

                    #threshold -= reduction_factor  # Linear reduction
                    threshold /= reduction_factor  # Exponential reduction

                # We do not further down the hierarchy.
                if len(clusters) <= 1:
                    continue

                for cluster in clusters:
                    new_node = HierarchicalCluster(cluster, threshold=threshold, parent=node)
                    node.add(new_node)

                    # Check if cluster still contains enough streamlines.
                    if len(cluster) > self.min_cluster_size:
                        next_nodes.append(new_node)

            nodes = next_nodes

        cluster_map = HierarchicalClusterMap(root)
        cluster_map.refdata = streamlines
        return cluster_map


def outlier_rejection(streamlines, threshold=0.2, confidence=0.95,
                      hqb=HierarchicalQuickBundles(), nb_samplings_max=50, seed=1234,
                      return_outlierness=False, verbose=False):
    """
    Detects outliers in a set of streamlines.
    This technique uses the Hierarchical QuickBundles which provides more
    information about how streamlines get along with each others. In particular,
    this method relies on the level
    Specify the impact of the ordering
    Parameters
    ----------
    streamlines : list of 2D arrays
        Each 2D array represents a sequence of 3D points (nb_points, 3).
    threshold : float (optional)
        TODO: Threshold on the outlierness
    confidence : float (optional)
        Level of confidence of the confidence interval around the mean
        streamlines path length in the clustering tree.
    hqb : `HierarchicalQuickBundles` object (optional)
        The clustering technique that will be used. By default,
        metric="MDF_12points", min_threshold=0, min_cluster_size=1.
    nb_samplings_max : int (optional)
        The maximum number of different orderings to try
    seed : int (optional)
        Controls the shuffling of the ordering
    return_outlierness : bool (optional)
        If `True`, the outlierness of each streamline is returned.
    verbose : bool (optional)
        Display information about the ongoing process.
    Returns
    -------
    inliers : `Cluster` object
        The cluster of streamlines considered inliers i.e. having an
        outlierness below or equal to `threshold`.
    outliers : `Cluster` object
        The cluster of streamlines considered outliers i.e. having an
        outlierness over `threshold`.
    outlierness : 1D array (optional)
        The outlierness of each streamline is returned only if
        `return_outlierness` is `True`.
    """

    if nb_samplings_max < 2:
        raise ValueError("'nb_samplings_max' must be >= 2")

    from scipy.special import ndtri
    sterror_factor = ndtri(confidence)

    rng = np.random.RandomState(seed)

    paths_length = np.zeros((len(streamlines), nb_samplings_max), dtype=int)
    ordering = np.arange(len(streamlines))
    for ordering_no in range(1, nb_samplings_max+1):
        if verbose:
            print "Ordering #{0}".format(ordering_no)
        rng.shuffle(ordering)

        tree_clusters = hqb.cluster(streamlines, ordering=ordering)
        # Compute streamlines path length in the clustering tree for this ordering.
        for node in tree_clusters:
            if node.parent is None:
                continue

            paths_length[node.indices, ordering_no-1] += 1
            #paths_length[node.indices, ordering_no-1] += (node.parent.threshold - node.threshold)

        if ordering_no < 2:  # Needs at least two orderings to compute stderror.
            continue

        # TODO: we should probably use Student's t distribution instead of the normal
        #       see http://brownmath.com/stat/sampsiz.htm#Case1
        # Compute confidence interval on mean path length for each streamline
        sterror_path_length = np.std(paths_length[:, :ordering_no], axis=1, ddof=1) / np.sqrt(ordering_no)

        if verbose:
            print "  Avg. sterror:", sterror_factor*sterror_path_length.mean()
            print "  Max. sterror:", sterror_factor*sterror_path_length.max()

        # Stop when the error margin is less than 0.5,
        # i.e. we are `confident`% confident in the means with a margin of
        # error no more than 1 length unit.
        if sterror_factor*sterror_path_length.mean() < 0.5:
            break

    # Compute the mean of paths length normalized by the max path length for each ordering.
    mean_paths_length = np.mean(paths_length[:, :ordering_no], axis=1)
    outlierness = 1 - (mean_paths_length/mean_paths_length.max())

    indices = np.arange(len(streamlines))
    outliers = Cluster(indices=indices[outlierness > threshold], refdata=streamlines)
    inliers = Cluster(indices=indices[outlierness <= threshold], refdata=streamlines)

    if return_outlierness:
        return inliers, outliers, outlierness

    return inliers, outliers
