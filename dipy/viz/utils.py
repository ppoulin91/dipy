
from __future__ import division, print_function, absolute_import

import os
import numpy as np
from scipy.ndimage import map_coordinates
from dipy.viz.colormap import line_colors
from nibabel.tmpdirs import InTemporaryDirectory

# Conditional import machinery for vtk
from dipy.utils.optpkg import optional_package

from dipy.core.geometry import vec2vec_rotmat, normalized_vector

# import vtk
# Allow import, but disable doctests if we don't have vtk
vtk, have_vtk, setup_module = optional_package('vtk')
ns, have_numpy_support, _ = optional_package('vtk.util.numpy_support')
matplotlib, have_mpl, _ = optional_package("matplotlib")

if have_mpl:
    from matplotlib.pylab import imread

def numpy_to_vtk_points(points):
    """ Numpy points array to a vtk points array

    Parameters
    ----------
    points : ndarray

    Returns
    -------
    vtk_points : vtkPoints()
    """
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(ns.numpy_to_vtk(np.asarray(points), deep=True))
    return vtk_points


def numpy_to_vtk_colors(colors):
    """ Numpy color array to a vtk color array

    Parameters
    ----------
    colors: ndarray

    Returns
    -------
    vtk_colors : vtkDataArray

    Notes
    -----
    If colors are not already in UNSIGNED_CHAR you may need to multiply by 255.

    Examples
    --------
    >>> import numpy as np
    >>> from dipy.viz.utils import numpy_to_vtk_colors
    >>> rgb_array = np.random.rand(100, 3)
    >>> vtk_colors = numpy_to_vtk_colors(255 * rgb_array)
    """
    vtk_colors = ns.numpy_to_vtk(np.asarray(colors), deep=True,
                                 array_type=vtk.VTK_UNSIGNED_CHAR)
    return vtk_colors


def numpy_to_vtk_matrix(array):
    """ Converts a numpy array to a VTK matrix.
    """
    if array is None:
        return None

    if array.shape == (4, 4):
        matrix = vtk.vtkMatrix4x4()
    elif array.shape == (3, 3):
        matrix = vtk.vtkMatrix3x3()
    else:
        raise ValueError("Invalid matrix shape: {0}".format(array.shape))

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            matrix.SetElement(i, j, array[i, j])

    return matrix


def set_input(vtk_object, inp):
    """ Generic input function which takes into account VTK 5 or 6

    Parameters
    ----------
    vtk_object: vtk object
    inp: vtkPolyData or vtkImageData or vtkAlgorithmOutput

    Returns
    -------
    vtk_object

    Notes
    -------
    This can be used in the following way::
        from dipy.viz.utils import set_input
        poly_mapper = set_input(vtk.vtkPolyDataMapper(), poly_data)
    """
    if isinstance(inp, vtk.vtkPolyData) \
       or isinstance(inp, vtk.vtkImageData):
        if vtk.VTK_MAJOR_VERSION <= 5:
            vtk_object.SetInput(inp)
        else:
            vtk_object.SetInputData(inp)
    elif isinstance(inp, vtk.vtkAlgorithmOutput):
        vtk_object.SetInputConnection(inp)

    vtk_object.Update()
    return vtk_object


def map_coordinates_3d_4d(input_array, indices):
    """ Evaluate the input_array data at the given indices
    using trilinear interpolation

    Parameters
    ----------
    input_array : ndarray,
        3D or 4D array
    indices : ndarray

    Returns
    -------
    output : ndarray
        1D or 2D array
    """

    if input_array.ndim <= 2 or input_array.ndim >= 5:
        raise ValueError("Input array can only be 3d or 4d")

    if input_array.ndim == 3:
        return map_coordinates(input_array, indices.T, order=1)

    if input_array.ndim == 4:
        values_4d = []
        for i in range(input_array.shape[-1]):
            values_tmp = map_coordinates(input_array[..., i],
                                         indices.T, order=1)
            values_4d.append(values_tmp)
        return np.ascontiguousarray(np.array(values_4d).T)


def lines_to_vtk_polydata(lines, colors=None):
    """ Create a vtkPolyData with lines and colors

    Parameters
    ----------
    lines : list
        list of N curves represented as 2D ndarrays
    colors : array (N, 3), list of arrays, tuple (3,), array (K,), None
        If None then a standard orientation colormap is used for every line.
        If one tuple of color is used. Then all streamlines will have the same
        colour.
        If an array (N, 3) is given, where N is equal to the number of lines.
        Then every line is coloured with a different RGB color.
        If a list of RGB arrays is given then every point of every line takes
        a different color.
        If an array (K,) is given, where K is the number of points of all
        lines then these are considered as the values to be used by the
        colormap.
        If an array (L,) is given, where L is the number of streamlines then
        these are considered as the values to be used by the colormap per
        streamline.
        If an array (X, Y, Z) or (X, Y, Z, 3) is given then the values for the
        colormap are interpolated automatically using trilinear interpolation.

    Returns
    -------
    poly_data : vtkPolyData
    is_colormap : bool, true if the input color array was a colormap
    """

    # Get the 3d points_array
    points_array = np.vstack(lines)

    nb_lines = len(lines)
    nb_points = len(points_array)

    lines_range = range(nb_lines)

    # Get lines_array in vtk input format
    lines_array = []
    points_per_line = np.zeros([nb_lines], np.int64)
    current_position = 0
    for i in lines_range:
        current_len = len(lines[i])
        points_per_line[i] = current_len

        end_position = current_position + current_len
        lines_array += [current_len]
        lines_array += range(current_position, end_position)
        current_position = end_position

    lines_array = np.array(lines_array)

    # Set Points to vtk array format
    vtk_points = numpy_to_vtk_points(points_array)

    # Set Lines to vtk array format
    vtk_lines = vtk.vtkCellArray()
    vtk_lines.GetData().DeepCopy(ns.numpy_to_vtk(lines_array))
    vtk_lines.SetNumberOfCells(nb_lines)

    is_colormap = False
    # Get colors_array (reformat to have colors for each points)
    #           - if/else tested and work in normal simple case
    if colors is None:  # set automatic rgb colors
        cols_arr = line_colors(lines)
        colors_mapper = np.repeat(lines_range, points_per_line, axis=0)
        vtk_colors = numpy_to_vtk_colors(255 * cols_arr[colors_mapper])
    else:
        cols_arr = np.asarray(colors)
        if cols_arr.dtype == np.object:  # colors is a list of colors
            vtk_colors = numpy_to_vtk_colors(255 * np.vstack(colors))
        else:
            if len(cols_arr) == nb_points:
                vtk_colors = ns.numpy_to_vtk(cols_arr, deep=True)
                is_colormap = True

            elif cols_arr.ndim == 1:
                if len(cols_arr) == nb_lines:  # values for every streamline
                    cols_arrx = []
                    for (i, value) in enumerate(colors):
                        cols_arrx += lines[i].shape[0]*[value]
                    cols_arrx = np.array(cols_arrx)
                    vtk_colors = ns.numpy_to_vtk(cols_arrx, deep=True)
                    is_colormap = True
                else:  # the same colors for all points
                    vtk_colors = numpy_to_vtk_colors(
                        np.tile(255 * cols_arr, (nb_points, 1)))

            elif cols_arr.ndim == 2:  # map color to each line
                colors_mapper = np.repeat(lines_range, points_per_line, axis=0)
                vtk_colors = numpy_to_vtk_colors(255 * cols_arr[colors_mapper])
            else:  # colormap
                #  get colors for each vertex
                cols_arr = map_coordinates_3d_4d(cols_arr, points_array)
                vtk_colors = ns.numpy_to_vtk(cols_arr, deep=True)
                is_colormap = True

    vtk_colors.SetName("Colors")

    # Create the poly_data
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(vtk_points)
    poly_data.SetLines(vtk_lines)
    poly_data.GetPointData().SetScalars(vtk_colors)
    return poly_data, is_colormap


def auto_orient(actor, direction, bbox_type="OBB", data_up=None, ref_up=(0, 1, 0), show_bounds=False):
    """ Orients an actor so its largest bounding box side is orthogonal to a
    given direction.
    This function returns a shallow copy of `actor` that have been automatically
    oriented so that its largest bounding box (either OBB or AABB) side faces
    the camera.
    Parameters
    ----------
    actor : `vtkProp3D` object
        Actor to orient.
    direction : 3-tuple
        Direction in which the largest bounding box side of the actor must be
        orthogonal to.
    bbox_type : str (optional)
        Type of bounding to use. Choices are "OBB" for Oriented Bounding Box or
        "AABB" for Axis-Aligned Bounding Box. Default: "OBB".
    data_up : tuple (optional)
        If provided, align this up vector with `ref_up` vector using rotation
        around `direction` axis.
    ref_up : tuple (optional)
        Use to align `data_up` vector. Default: (0, 1, 0).
    show_bounds : bool
        Whether to display or not the actor bounds used by this function.
        Default: False.
    Returns
    -------
    `vtkProp3D` object
        Shallow copy of `actor` that have been oriented accordingly to the
        given options.
    """
    new_actor = vtk.vtkActor()
    new_actor.ShallowCopy(actor)

    if bbox_type == "AABB":
        x1, x2, y1, y2, z1, z2 = new_actor.GetBounds()
        width, height, depth = x2-x1, y2-y1, z2-z1
        canonical_axes = (width, 0, 0), (0, height, 0), (0, 0, depth)
        idx = np.argsort([width, height, depth])
        coord_min = np.array(canonical_axes[idx[0]])
        coord_mid = np.array(canonical_axes[idx[1]])
        coord_max = np.array(canonical_axes[idx[2]])
        corner = np.array((x1, y1, z1))
    elif bbox_type == "OBB":
        corner = np.zeros(3)
        coord_max = np.zeros(3)
        coord_mid = np.zeros(3)
        coord_min = np.zeros(3)
        sizes = np.zeros(3)

        points = new_actor.GetMapper().GetInput().GetPoints()
        vtk.vtkOBBTree.ComputeOBB(points, corner, coord_max, coord_mid, coord_min, sizes)
    else:
        raise ValueError("Unknown `bbox_type`: {0}".format(bbox_type))

    if show_bounds:
        from dipy.viz.actor import line
        assembly = vtk.vtkAssembly()
        assembly.AddPart(new_actor)
        #assembly.AddPart(line([np.array([new_actor.GetCenter(), np.array(new_actor.GetCenter())+(0,0,20)])], colors=(1, 1, 0)))
        assembly.AddPart(line([np.array([corner, corner+coord_max])], colors=(1, 0, 0)))
        assembly.AddPart(line([np.array([corner, corner+coord_mid])], colors=(0, 1, 0)))
        assembly.AddPart(line([np.array([corner, corner+coord_min])], colors=(0, 0, 1)))

        # from dipy.viz.actor import axes
        # local_axes = axes(scale=20)
        # local_axes.SetPosition(new_actor.GetCenter())
        # assembly.AddPart(local_axes)
        new_actor = assembly

    normal = np.cross(coord_mid, coord_max)

    direction = normalized_vector(direction)
    normal = normalized_vector(normal)
    R = vec2vec_rotmat(normal, direction)
    M = np.eye(4)
    M[:3, :3] = R

    transform = vtk.vtkTransform()
    transform.PostMultiply()
    transform.SetMatrix(numpy_to_vtk_matrix(M))

    # TODO: I think we also need the right/depth vector in addition to the up vector for the data.
    if data_up is not None:
        # Find the rotation around `direction` axis to align top of the brain with the camera up.
        data_up = normalized_vector(data_up)
        ref_up = normalized_vector(ref_up)
        up = np.dot(R, np.array(data_up))
        up[2] = 0  # Orthogonal projection onto the XY-plane.
        up = normalized_vector(up)

        # Angle between oriented `data_up` and `ref_up`.
        angle = np.arccos(np.dot(up, np.array(ref_up)))
        angle = angle/np.pi*180.

        # Check if the rotation should be clockwise or anticlockwise.
        if up[0] < 0:
            angle = -angle

        transform.RotateWXYZ(angle, -direction)

    # Apply orientation change to the new actor.
    new_actor.AddOrientation(transform.GetOrientation())

    return new_actor

def matplotlib_figure_to_numpy(fig, dpi=100, fname=None, flip_up_down=True,
                               transparent=False):
    r""" Convert a Matplotlib figure to a 3D numpy array with RGBA channels

    Parameters
    ----------
    fig : obj,
        A matplotlib figure object

    dpi : int
        Dots per inch

    fname : str
        If ``fname`` is given then the array will be saved as a png to this
        position.

    flip_up_down : bool
        The origin is different from matlplotlib default and VTK's default
        behaviour (default True).

    transparent : bool
        Make background transparent (default False).

    Returns
    -------
    arr : ndarray
        a numpy 3D array of RGBA values

    Notes
    ------
    The safest way to read the pixel values from the figure was to save them
    using savefig as a png and then read again the png. There is a cleaner
    way found here http://www.icare.univ-lille1.fr/drupal/node/1141 where
    you can actually use fig.canvas.tostring_argb() to get the values directly
    without saving to the disk. However, this was not stable across different
    machines and needed more investigation from what time permited.
    """

    if fname is None:
        with InTemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'tmp.png')
            fig.savefig(fname, dpi=dpi, transparent=transparent,
                        bbox_inches='tight', pad_inches=0.)
            arr = (imread(fname) * 255).astype('uint8')
    else:
        fig.savefig(fname, dpi=dpi, transparent=transparent,
                    bbox_inches='tight', pad_inches=0.)
        arr = (imread(fname) * 255).astype('uint8')

    if flip_up_down:
        arr = np.flipud(arr)

    return arr
