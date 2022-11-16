import numpy as np
import os

import taichi as ti

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch)

n = 20
n_particles = n**3

x = ti.Vector.field(3, float, n_particles)
colors = ti.Vector.field(4, float, n_particles)

@ti.kernel
def init_cube_vol():
    for i,j,k in ti.ndrange(n, n, n):
        x[i*n*n+j*n+k] = ti.Vector([i*0.01, j*0.01, k*0.01]) + 0.25

@ti.kernel
def set_color(color: ti.types.ndarray()):
    for i in range(n_particles):
        colors[i] = ti.Vector([
            color[0], color[1],
            color[2], color[3]
        ])
init_cube_vol()
res = (1280, 720)
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


    window.GUI.begin("color", 0.05, 0.05, 0.2, 0.4)

    r = window.GUI.slider_float("red ", r, 0.0, 1.0)
    g = window.GUI.slider_float("green ", g, 0.0, 1.0)
    b = window.GUI.slider_float("blue ", b, 0.0, 1.0)
    a = window.GUI.slider_float("alpha ", a, 0.0, 1.0)
    window.GUI.end()
    set_color(np.array([r, g, b, a]))


def render():
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.ambient_light((0, 0, 0))
    scene.particles(x, per_vertex_color=colors, radius=0.005)

    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.5))

    canvas.scene(scene)

while window.running:
    show_options()

    render()
    window.show()
