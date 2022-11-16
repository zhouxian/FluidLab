import numpy as np
import os
import os
import cv2
import time
import skimage
import numpy as np
import taichi as ti
from time import time
from scipy import ndimage
from fluidlab.engine.renderers.gl_renderer_src import flex_renderer

import taichi as ti

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch)

res = (320, 320)
window = ti.ui.Window("Color Picker", res, vsync=True)

canvas = window.get_canvas()
canvas.set_background_color((1,1,1))
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
camera.position(0.5, 1.0, 1.95)
camera.lookat(0.2, 0.3, 0.5)
camera.fov(15)

r, g, b, a = 1, 1, 1, 1
def show_options():
    global use_random_colors
    global paused
    global r, g, b, a
    global curr_preset_id
    global g_x, g_y, g_z

    window.GUI.begin("color", 0.05, 0.05, 0.8, 0.8)

    r = window.GUI.slider_float("red ", r, 0.0, 1.0)
    g = window.GUI.slider_float("green ", g, 0.0, 1.0)
    b = window.GUI.slider_float("blue ", b, 0.0, 1.0)
    a = window.GUI.slider_float("alpha ", a, 0.0, 1.0)
    window.GUI.end()

    flex_renderer.set_body_color(0, np.array([r, g, b, a]))


def render():
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.ambient_light((0, 0, 0))

    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.5))

    canvas.scene(scene)


flex_renderer.init(1024, 1024, 0, 0.4)
n = 30
flex_renderer.create_scene(
    True,
    0.01,
    0.01,
    1.0,
    1.0,
    3.5,
    0.1,
    np.array([1, 10, 1.1]),
    np.array([1.  , 1.  , 0.98]),
    60,
    0.0,
    10.0,
    np.array([n**3]),
    np.array([0]),
    np.array([1, 1, 1, 1.0]),
    np.array([False]),
    1,
)
flex_renderer.set_camera_params(np.array([-0.3 ,  5.64,  5.  ]), np.array([-0.13142319 , -0.83442671,  0.        ]), 0.1, 10.0)

pos = np.random.uniform(0.2, 0.8, (n**3, 4))

flex_renderer.set_particles_state(pos.flatten(), np.ones(n**3))

while window.running:

    show_options()

    render()
    window.show()

    img = flex_renderer.render()
    img = np.flip(img.reshape([1024, 1024, 4]), 0)[:, :, :3]

    cv2.imshow('img', img[..., ::-1])
    cv2.waitKey(1)

