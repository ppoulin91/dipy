import numpy as np
from dipy.viz import actor, window, widget
from dipy.data import fetch_viz_icons
from dipy.data import read_viz_icons
from dipy.tracking.streamline import transform_streamlines
import nibabel.trackvis as tv
import nibabel as nib
from dipy.segment.clustering import QuickBundles
from dipy.viz.axycolor import distinguishable_colormap
from dipy.tracking.streamline import compress_streamlines

global screen_size

ren = window.Renderer()
screen_size=(1200, 900)
show_m = window.ShowManager(ren, size=screen_size)
screen_size = ren.GetSize()


################################################################################
# FA
################################################################################
print("Adding FA")
fa = nib.load("/home/algo/data/selfie/hardi_188/tracting/fa.nii.gz")
# fa_affine = fa.get_affine()
fa_affine = np.eye(4)
fa_data = fa.get_data()

fa_actor = actor.slice(fa_data, fa_affine)
fa_actor.display_extent(0, fa.shape[0] - 1,
                               fa.shape[1]//2, fa.shape[1]//2, 0, fa.shape[2] - 1)
fa_actor.opacity(1.0)
ren.add(fa_actor)

def change_fa_slice(obj, event):
    y = int(np.round(obj.get_value()))
    fa_actor.display_extent(0, fa.shape[0] - 1,
                               y, y, 0, fa.shape[2] - 1)
 
fa_slider = widget.slider(show_m.iren, show_m.ren,
                       callback=change_fa_slice,
                       min_value=0,
                       max_value=fa.shape[2] - 1,
                       value=fa.shape[2] / 2,
                       label="Move FA slice",
                       right_normalized_pos=(.95, 0.7),
                       size=(120, 0), label_format="%0.lf",
                       color=(1., 1., 1.),
                       selected_color=(0.8, 0.8, 0.8))


################################################################################
# RGB
################################################################################
print("Adding RGB")
rgb = nib.load("/home/algo/data/selfie/hardi_188/tracting/rgb.nii.gz")
# rgb_affine = rgb.get_affine()
rgb_affine = np.eye(4)
rgb_data = rgb.get_data()

rgb_actor = actor.slice(rgb_data, rgb_affine)
rgb_actor.opacity(1.0)
ren.add(rgb_actor)

def change_rgb_slice(obj, event):
    z = int(np.round(obj.get_value()))
    rgb_actor.display_extent(0, rgb.shape[0] - 1,
                               0, rgb.shape[1] - 1, z, z)

rgb_slider = widget.slider(show_m.iren, show_m.ren,
                       callback=change_rgb_slice,
                       min_value=0,
                       max_value=rgb.shape[2] - 1,
                       value=rgb.shape[2] / 2,
                       label="Move RGB slice",
                       right_normalized_pos=(.95, 0.5),
                       size=(120, 0), label_format="%0.lf",
                       color=(1., 1., 1.),
                       selected_color=(1.0, 0.0, 0.0))


################################################################################
# T1
################################################################################
print("Adding T1")
t1 = nib.load("/home/algo/data/selfie/hardi_188/tracting/t1.nii.gz")
# t1_affine = t1.get_affine()
t1_affine = np.eye(4)
t1_data = t1.get_data()

t1_actor = actor.slice(t1_data, t1_affine)

t1_actor.opacity(1.0)
t1_actor.display_extent(t1.shape[0]//2, t1.shape[0]//2,
                               0, t1.shape[1] - 1, 0, t1.shape[2] - 1)
ren.add(t1_actor)

def change_t1_slice(obj, event):
    x = int(np.round(obj.get_value()))
    t1_actor.display_extent(x, x,
                               0, t1.shape[1] - 1, 0, t1.shape[2] - 1)

t1_slider = widget.slider(show_m.iren, show_m.ren,
                       callback=change_t1_slice,
                       min_value=0,
                       max_value=t1.shape[2] - 1,
                       value=t1.shape[2] / 2,
                       label="Move T1 slice",
                       right_normalized_pos=(.95, 0.3),
                       size=(120, 0), label_format="%0.lf",
                       color=(1., 1., 1.),
                       selected_color=(1.0, 0.0, 0.0))


################################################################################
# Streamlines
################################################################################
print("Adding Streamlines")
streams, hdr = tv.read("/home/algo/data/selfie/hardi_188/tracting/nice_bundle.trk")
#streams, hdr = tv.read("/home/algo/data/selfie/hardi_188/tracting/tractogram.trk")
global streamlines
streamlines = [s[0] for s in streams[::1]]
global stream_actor
stream_actor = actor.line(streamlines)
ren.add(stream_actor)


################################################################################
# Widgets
################################################################################
# fetch_viz_icons()

button_png_minus = read_viz_icons(fname='minus_i.png')

def button_strml_callback(obj, event):
    global stream_actor
    stream_actor.SetVisibility(1-stream_actor.GetVisibility())

button_strml = widget.button(show_m.iren,
                            show_m.ren,
                            button_strml_callback,
                            button_png_minus, (.15, .7), (120, 50))

button_png_compression = read_viz_icons(fname='compression_i.png')

def button_comp_callback(obj, event):
    global streamlines
    global stream_actor
    streamlines = compress_streamlines(streamlines)
    ren.RemoveActor(stream_actor)
    stream_actor = actor.line(streamlines)
    ren.add(stream_actor)
    obj.GetRepresentation().GetHoveringProperty().SetColor((1.0, 0., 0.))
    obj.ProcessEventsOff()

button_comp = widget.button(show_m.iren,
                            show_m.ren,
                            button_comp_callback,
                            button_png_compression, (.15, .5), (120, 50))

button_png_star = read_viz_icons(fname='star_i.png')

def button_qb_callback(obj, event):
    global streamlines
    global stream_actor
    qb = QuickBundles(threshold=20)
    clusters = qb.cluster(streamlines)
    colors_bundle = np.ones((len(streamlines), 3))
    for cluster, color in zip(clusters, distinguishable_colormap()):
        colors_bundle[cluster.indices] = color
    
    colors = []
    for color, streamline in zip(colors_bundle, streamlines):
        colors += [color] * len(streamline)
        
    vtk_colors = actor.numpy_to_vtk_colors(255 * np.array(colors))
    del colors_bundle
    vtk_colors.SetName("Colors")
    stream_actor.GetMapper().GetInput().GetPointData().SetScalars(vtk_colors)

button_qb = widget.button(show_m.iren,
                            show_m.ren,
                            button_qb_callback,
                            button_png_star, (.15, .3), (120, 50))


################################################################################
# Rendering
################################################################################
print("Rendering")

def win_callback(obj, event):
    global screen_size
    button_comp.place(ren)
    button_strml.place(ren)
    button_qb.place(ren)
    if screen_size != obj.GetSize():
        fa_slider.place(ren)
        rgb_slider.place(ren)
        t1_slider.place(ren)
        screen_size = obj.GetSize()

show_m.initialize()
show_m.add_window_callback(win_callback)

show_m.render()
show_m.start()
