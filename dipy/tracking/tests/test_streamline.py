from __future__ import print_function

import numpy as np

from nose.tools import assert_true, assert_equal, assert_almost_equal
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_raises, run_module_suite)

import dipy.tracking.streamline as dpstreamline
from dipy.tracking.streamline import (relist_streamlines, unlist_streamlines,
                                      center_streamlines, transform_streamlines,
                                      select_random_set_of_streamlines)


# Define an actual streamline
line = np.array([[82.20181274,  91.36505890,  43.15737152],
                 [82.38442230,  91.79336548,  43.87036514],
                 [82.48710632,  92.27861023,  44.56298065],
                 [82.53310394,  92.78545380,  45.24635315],
                 [82.53793335,  93.26902008,  45.94785309],
                 [82.48797607,  93.75003815,  46.64939880],
                 [82.35533142,  94.25181580,  47.32533264],
                 [82.15484619,  94.76634216,  47.97451019],
                 [81.90982819,  95.28792572,  48.60244370],
                 [81.63336945,  95.78153229,  49.23971176],
                 [81.35479736,  96.24868011,  49.89558792],
                 [81.08713531,  96.69807434,  50.56812668],
                 [80.81504822,  97.14285278,  51.24193192],
                 [80.52591705,  97.56719971,  51.92168427],
                 [80.26599884,  97.98269653,  52.61848068],
                 [80.04635620,  98.38131714,  53.33855820],
                 [79.84691620,  98.77052307,  54.06955338],
                 [79.57667542,  99.13599396,  54.78985596],
                 [79.23351288,  99.43207550,  55.51065063],
                 [78.84815979,  99.64141846,  56.24016571],
                 [78.47383881,  99.77347565,  56.99299240],
                 [78.12837219,  99.81330872,  57.76969528],
                 [77.80438995,  99.85082245,  58.55574799],
                 [77.49439240,  99.88065338,  59.34777069],
                 [77.21414185,  99.85343933,  60.15090561],
                 [76.96416473,  99.82772827,  60.96406937],
                 [76.74712372,  99.80519104,  61.78676605],
                 [76.52263641,  99.79122162,  62.60765076],
                 [76.03757477, 100.08692169,  63.24152374],
                 [75.44867706, 100.35265350,  63.79513168],
                 [74.78033447, 100.57255554,  64.27278900],
                 [74.11605835, 100.77330780,  64.76428986],
                 [73.51222992, 100.98779297,  65.32373047],
                 [72.97387695, 101.23387146,  65.93502045],
                 [72.47355652, 101.49151611,  66.57343292],
                 [71.99834442, 101.72480774,  67.23979950],
                 [71.56909180, 101.98665619,  67.92664337],
                 [71.18083191, 102.29483795,  68.61888123],
                 [70.81879425, 102.63343048,  69.31127167],
                 [70.47422791, 102.98672485,  70.00532532],
                 [70.10092926, 103.28502655,  70.70999908],
                 [69.69512177, 103.51667023,  71.42147064],
                 [69.27423096, 103.71351624,  72.13452911],
                 [68.91260529, 103.81676483,  72.89796448],
                 [68.60788727, 103.81982422,  73.69258118],
                 [68.34162903, 103.76619720,  74.49915314],
                 [68.08542633, 103.70635223,  75.30856323],
                 [67.83590698, 103.60187531,  76.11553955],
                 [67.56822968, 103.44821930,  76.90870667],
                 [67.28399658, 103.25878906,  77.68825531],
                 [67.00117493, 103.03740692,  78.45989227],
                 [66.72718048, 102.80329895,  79.23099518],
                 [66.46197510, 102.54130554,  79.99622345],
                 [66.20803833, 102.22305298,  80.74387360],
                 [65.96872711, 101.88980865,  81.48987579],
                 [65.72864532, 101.59316254,  82.25085449],
                 [65.47808075, 101.33383942,  83.02194214],
                 [65.21841431, 101.11295319,  83.80186462],
                 [64.95678711, 100.94080353,  84.59326935],
                 [64.71759033, 100.82022095,  85.40114594],
                 [64.48053741, 100.73490143,  86.21411896],
                 [64.24304199, 100.65074158,  87.02709198],
                 [64.01773834, 100.55318451,  87.84204865],
                 [63.83801651, 100.41996765,  88.66333008],
                 [63.70982361, 100.25119019,  89.48779297],
                 [63.60707855, 100.06730652,  90.31262207],
                 [63.46164322,  99.91001892,  91.13648224],
                 [63.26287842,  99.78648376,  91.95485687],
                 [63.03713226,  99.68377686,  92.76905823],
                 [62.81192398,  99.56619263,  93.58140564],
                 [62.57145309,  99.42708588,  94.38592529],
                 [62.32259369,  99.25592804,  95.18167114],
                 [62.07497787,  99.05770111,  95.97154236],
                 [61.82253647,  98.83877563,  96.75438690],
                 [61.59536743,  98.59293365,  97.53706360],
                 [61.46530151,  98.30503845,  98.32772827],
                 [61.39904785,  97.97928619,  99.11172485],
                 [61.33279419,  97.65353394,  99.89572906],
                 [61.26067352,  97.30914307, 100.67123413],
                 [61.19459534,  96.96743011, 101.44847107],
                 [61.19580460,  96.63417053, 102.23215485],
                 [61.26572037,  96.29887390, 103.01185608],
                 [61.39840698,  95.96297455, 103.78307343],
                 [61.57207870,  95.64262390, 104.55268097],
                 [61.78163528,  95.35540771, 105.32629395],
                 [62.06700134,  95.09746552, 106.08564758],
                 [62.39427185,  94.85724640, 106.83369446],
                 [62.74076462,  94.62278748, 107.57482147],
                 [63.11461639,  94.40107727, 108.30641937],
                 [63.53397751,  94.20418549, 109.02002716],
                 [64.00019836,  94.03809357, 109.71183777],
                 [64.43580627,  93.87523651, 110.42416382],
                 [64.84857941,  93.69993591, 111.14715576],
                 [65.26740265,  93.51858521, 111.86515808],
                 [65.69511414,  93.36718750, 112.58474731],
                 [66.10470581,  93.22719574, 113.31711578],
                 [66.45891571,  93.06028748, 114.07256317],
                 [66.78582001,  92.90560913, 114.84281921],
                 [67.11138916,  92.79004669, 115.62040710],
                 [67.44729614,  92.75711823, 116.40135193],
                 [67.75688171,  92.98265076, 117.16111755],
                 [68.02041626,  93.28012848, 117.91371155],
                 [68.25725555,  93.53466797, 118.69052124],
                 [68.46047974,  93.63263702, 119.51107788],
                 [68.62039948,  93.62007141, 120.34690094],
                 [68.76782227,  93.56475067, 121.18331909],
                 [68.90222168,  93.46326447, 122.01765442],
                 [68.99872589,  93.30039978, 122.84759521],
                 [69.04119873,  93.05428314, 123.66156769],
                 [69.05086517,  92.74394989, 124.45450592],
                 [69.02742004,  92.40427399, 125.23509979],
                 [68.95466614,  92.09059143, 126.02339935],
                 [68.84975433,  91.79674530, 126.81564331],
                 [68.72673798,  91.53726196, 127.61715698],
                 [68.60685730,  91.30300140, 128.42681885],
                 [68.50636292,  91.12481689, 129.25317383],
                 [68.39311218,  91.01572418, 130.08976746],
                 [68.25946808,  90.94654083, 130.92756653]],
                dtype=np.float32)

line_64bit = line.astype(np.float64)

lines = [line[[0, 10]], line,
         line[::2], line[::3],
         line[::5], line[::6]]
lines_64bit = [line_64bit[[0, 10]], line_64bit,
               line_64bit[::2], line_64bit[::3],
               line_64bit[::4], line_64bit[::5]]

heterogeneous_lines = [line_64bit,
                       line_64bit.reshape((-1, 6)),
                       line_64bit.reshape((-1, 2))]


def length_python(xyz, along=False):
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape[0] < 2:
        if along:
            return np.array([0])
        return 0
    dists = np.sqrt((np.diff(xyz, axis=0)**2).sum(axis=1))
    if along:
        return np.cumsum(dists)
    return np.sum(dists)


def set_number_of_points_python(xyz, n_pols=3):
    def _extrap(xyz, cumlen, distance):
        ''' Helper function for extrapolate '''
        ind = np.where((cumlen-distance) > 0)[0][0]
        len0 = cumlen[ind-1]
        len1 = cumlen[ind]
        Ds = distance-len0
        Lambda = Ds/(len1-len0)
        return Lambda*xyz[ind] + (1-Lambda)*xyz[ind-1]

    cumlen = np.zeros(xyz.shape[0])
    cumlen[1:] = length_python(xyz, along=True)
    step = cumlen[-1] / (n_pols-1)

    ar = np.arange(0, cumlen[-1], step)
    if np.abs(ar[-1] - cumlen[-1]) < np.finfo('f4').eps:
        ar = ar[:-1]

    xyz2 = [_extrap(xyz, cumlen, distance) for distance in ar]
    return np.vstack((np.array(xyz2), xyz[-1]))


def test_set_number_of_points():
    # Test resampling of only one line
    nb_points = 12
    new_line_cython = dpstreamline.set_number_of_points(line, nb_points)
    new_line_python = set_number_of_points_python(line, nb_points)
    assert_equal(len(new_line_cython), nb_points)
    # Using a 5 digits precision because of line is in float32.
    assert_array_almost_equal(new_line_cython, new_line_python, 5)

    new_line_cython = dpstreamline.set_number_of_points(line_64bit, nb_points)
    new_line_python = set_number_of_points_python(line_64bit, nb_points)
    assert_equal(len(new_line_cython), nb_points)
    assert_array_almost_equal(new_line_cython, new_line_python)

    res = []
    simple_line = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], 'f4')
    for nb_points in range(2, 200):
        new_line_cython = dpstreamline.set_number_of_points(simple_line,
                                                            nb_points)
        res.append(nb_points - len(new_line_cython))

    assert_equal(np.sum(res), 0)

    # Test resampling of multiple lines of different nb_points
    nb_points = 12
    new_lines_cython = dpstreamline.set_number_of_points(lines, nb_points)

    for i, s in enumerate(lines):
        new_line_python = set_number_of_points_python(s, nb_points)
        # Using a 5 digits precision because of line is in float32.
        assert_array_almost_equal(new_lines_cython[i], new_line_python, 5)

    new_lines_cython = dpstreamline.set_number_of_points(lines_64bit,
                                                         nb_points)

    for i, s in enumerate(lines_64bit):
        new_line_python = set_number_of_points_python(s, nb_points)
        assert_array_almost_equal(new_lines_cython[i], new_line_python)

    # Test lines with mixed dtype
    lines_mixed_dtype = [line, line_64bit,
                         line.astype(np.int32), line.astype(np.int64)]
    nb_points_mixed_dtype = []
    for s in dpstreamline.set_number_of_points(lines_mixed_dtype, nb_points):
        nb_points_mixed_dtype.append(len(s))

    assert_array_equal(nb_points_mixed_dtype,
                       [nb_points]*len(lines_mixed_dtype))

    # Test lines with differente shape
    new_lines_cython = dpstreamline.set_number_of_points(heterogeneous_lines,
                                                         nb_points)

    for i, s in enumerate(heterogeneous_lines):
        new_line_python = set_number_of_points_python(s, nb_points)
        assert_array_almost_equal(new_lines_cython[i], new_line_python)

    # Test line with integer dtype
    new_line = dpstreamline.set_number_of_points(line.astype(np.int32))
    assert_true(new_line.dtype == np.float32)
    new_line = dpstreamline.set_number_of_points(line.astype(np.int64))
    assert_true(new_line.dtype == np.float64)

    # Test empty list
    assert_equal(dpstreamline.set_number_of_points([]), [])

    # Test line having only one point
    assert_raises(ValueError,
                  dpstreamline.set_number_of_points, np.array([[1, 2, 3]]))

    # We do not support list of lists, it should be numpy ndarray.
    line_unsupported = [[1, 2, 3], [4, 5, 5], [2, 1, 3], [4, 2, 1]]
    assert_raises(AttributeError,
                  dpstreamline.set_number_of_points, line_unsupported)

    # Test setting number of points of a numpy with flag WRITABLE=False
    line_readonly = line.copy()
    line_readonly.setflags(write=False)
    assert_equal(len(dpstreamline.set_number_of_points(line_readonly,
                                                       nb_points=42)),
                 42)

    # Test setting computing length of a numpy with flag WRITABLE=False
    lines_readonly = []
    for s in lines:
        lines_readonly.append(s.copy())
        lines_readonly[-1].setflags(write=False)

    assert_equal(len(dpstreamline.set_number_of_points(lines_readonly,
                                                       nb_points=42)),
                 len(lines_readonly))

    lines_readonly = []
    for s in lines_64bit:
        lines_readonly.append(s.copy())
        lines_readonly[-1].setflags(write=False)

    assert_equal(len(dpstreamline.set_number_of_points(lines_readonly,
                                                       nb_points=42)),
                 len(lines_readonly))


def test_length():
    # Test length of only one line
    length_line_cython = dpstreamline.length(line)
    length_line_python = length_python(line)
    assert_equal(length_line_cython, length_line_python)

    length_line_cython = dpstreamline.length(line_64bit)
    length_line_python = length_python(line_64bit)
    assert_equal(length_line_cython, length_line_python)

    # Test computing length of multiple lines of different nb_points
    length_lines_cython = dpstreamline.length(lines)

    for i, s in enumerate(lines):
        length_line_python = length_python(s)
        assert_array_almost_equal(length_lines_cython[i], length_line_python)

    length_lines_cython = dpstreamline.length(lines_64bit)

    for i, s in enumerate(lines_64bit):
        length_line_python = length_python(s)
        assert_array_almost_equal(length_lines_cython[i], length_line_python)

    # Test lines having mixed dtype
    lines_mixed_dtype = [line, line_64bit,
                         line.astype(np.int32), line.astype(np.int64)]
    lengths_mixed_dtype = [dpstreamline.length(s) for s in lines_mixed_dtype]
    assert_array_equal(dpstreamline.length(lines_mixed_dtype),
                       lengths_mixed_dtype)

    # Test lines with differente shape
    length_lines_cython = dpstreamline.length(heterogeneous_lines)

    for i, s in enumerate(heterogeneous_lines):
        length_line_python = length_python(s)
        assert_array_almost_equal(length_lines_cython[i], length_line_python)

    # Test line having integer dtype
    length_line = dpstreamline.length(line.astype('int'))
    assert_true(length_line.dtype == np.float64)

    # Test empty list
    assert_equal(dpstreamline.length([]), 0.0)

    # Test line having only one point
    assert_equal(dpstreamline.length(np.array([[1, 2, 3]])), 0.0)

    # We do not support list of lists, it should be numpy ndarray.
    line_unsupported = [[1, 2, 3], [4, 5, 5], [2, 1, 3], [4, 2, 1]]
    assert_raises(AttributeError, dpstreamline.length, line_unsupported)

    # Test setting computing length of a numpy with flag WRITABLE=False
    lines_readonly = []
    for s in lines:
        lines_readonly.append(s.copy())
        lines_readonly[-1].setflags(write=False)

    expected = [length_python(s) for s in lines_readonly]
    assert_array_equal(dpstreamline.length(lines_readonly), expected)

    lines_readonly = []
    for s in lines_64bit:
        lines_readonly.append(s.copy())
        lines_readonly[-1].setflags(write=False)

    expected = [length_python(s) for s in lines_readonly]
    assert_array_equal(dpstreamline.length(lines_readonly), expected)


def test_unlist_relist_streamlines():
    streamlines = [np.random.rand(10, 3),
                   np.random.rand(20, 3),
                   np.random.rand(5, 3)]

    points, offsets = unlist_streamlines(streamlines)

    assert_equal(offsets.dtype, np.dtype('i8'))

    assert_equal(points.shape, (35, 3))
    assert_equal(len(offsets), len(streamlines))

    streamlines2 = relist_streamlines(points, offsets)

    assert_equal(len(streamlines), len(streamlines2))

    for i in range(len(streamlines)):
        assert_array_equal(streamlines[i], streamlines2[i])


def test_center_and_transform():
    A = np.array([[1, 2, 3], [1, 2, 3.]])
    streamlines = [A for i in range(10)]

    streamlines2, center = center_streamlines(streamlines)

    B = np.zeros((2, 3))
    assert_array_equal(streamlines2[0], B)
    assert_array_equal(center, A[0])

    affine = np.eye(4)
    affine[0, 0] = 2
    affine[:3, -1] = - np.array([2, 1, 1]) * center

    streamlines3 = transform_streamlines(streamlines, affine)
    assert_array_equal(streamlines3[0], B)


def test_select_streamlines():
    streamlines = [np.random.rand(10, 3),
                   np.random.rand(20, 3),
                   np.random.rand(5, 3)]

    new_streamlines = select_random_set_of_streamlines(streamlines, 2)

    assert_equal(len(new_streamlines), 2)

    new_streamlines = select_random_set_of_streamlines(streamlines, 4)

    assert_equal(len(new_streamlines), 3)



if __name__ == '__main__':
    run_module_suite()
