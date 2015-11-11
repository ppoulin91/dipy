#! /usr/bin/env python

import itertools

import numpy as np
import nibabel as nib

from dipy.fixes import argparse
from dipy.tracking.utils import density_map
from dipy.tracking.streamline import set_number_of_points

from dipy.segment.clustering import QuickBundles, quickbundles_with_merging
from dipy.segment.metric import ResampleFeature, AveragePointwiseEuclideanMetric


def build_args_parser():
    description = "Generate a heat map of the centroids found in a given tractogram."
    p = argparse.ArgumentParser(description=description)

    p.add_argument("tractogram", help="File containing streamlines (.trk).")

    p.add_argument("--threshold", type=float, default=18,
                   help="QuickBundles' threshold (in mm). Default: 18mm")

    p.add_argument("--nb-orderings", type=int, default=30,
                   help="Number of time QuickBundles is applied with different ordering. Default: 30")

    p.add_argument("--out", default="heatmap.nii.gz",
                   help="Output filename for the heat map (.nii.gz). Default: 'heatmap.nii.gz'")

    p.add_argument("--method", default="QB", choices=["QB", "QBm"],
                   help="QuickBundles variant to used (QB, QBm). Default: 'QB'")

    p.add_argument("--ref",
                   help="If specified, streamlines are consider in that reference space (.nii|.nii.gz).")

    p.add_argument("--seed", type=int, default=1234, help="Seed used to shuffle orderings. Default: 1234")
    p.add_argument('-v', "--verbose", action="store_true",
                   help="Enable verbose mode.")

    return p


def load_tractogram(filename, points_space="voxmm"):
    streams, hdr = nib.trackvis.read(filename, points_space)
    streamlines, colors, properties = zip(*streams)
    return streamlines, colors, properties, hdr


def save_tractogram(filename, streamlines, colors, properties, hdr, points_space="voxmm"):
    data = itertools.izip_longest(streamlines, colors, properties)
    nib.trackvis.write(filename.split(".trk")[0] + ".trk", data, hdr, points_space=points_space)


def save_nifti(filename, heatmap, affine):
    img = nib.Nifti1Image(heatmap, affine)
    nib.save(img, filename)


def perform_clustering(streamlines, threshold, ordering, method="QB"):
    resample = ResampleFeature(nb_points=20)
    metric = AveragePointwiseEuclideanMetric(resample)

    if method == "QB":
        qb = QuickBundles(threshold=threshold, metric=metric)
        clusters = qb.cluster(streamlines, ordering=ordering)
    elif method == "QBm":
        qb = QuickBundles(threshold=threshold, metric=metric)
        clusters = quickbundles_with_merging(streamlines, qb, ordering=ordering)

    return clusters


def fuse_centroids(streamlines, threshold, orderings, method="QB", verbose=False):
    centroids = []
    for i, ordering in enumerate(orderings):
        if verbose:
            print "Ordering #{}".format(i)

        clusters = perform_clustering(streamlines, threshold, ordering, method=method)
        centroids.extend(clusters.centroids)

    return centroids


def main():
    parser = build_args_parser()
    args = parser.parse_args()

    streamlines, colors, properties, hdr = load_tractogram(args.tractogram, points_space='voxmm')
    vol_dims = hdr['dim']
    affine = hdr['vox_to_ras']

    if args.ref is not None:
        nii = nib.load(args.ref)
        vol_dims = nii.shape
        affine = nii.affine

    rng = np.random.RandomState(args.seed)
    orderings = [rng.choice(len(streamlines), size=len(streamlines), replace=False) for _ in range(args.nb_orderings)]
    centroids = fuse_centroids(streamlines, args.threshold, orderings, method=args.method, verbose=args.verbose)
    centroids = set_number_of_points(centroids, nb_points=300)

    #data = density_map(centroids, vol_dims, affine=affine).astype(float)
    data = density_map(centroids, vol_dims, affine=np.eye(4)).astype(float)
    save_nifti(args.out, data, affine=affine)

    save_tractogram("centroids.trk", centroids, [None]*len(centroids), [None]*len(centroids), hdr, points_space="voxmm")


if __name__ == "__main__":
    main()
