import os
import trimesh
import skimage
import numpy as np
import taichi as ti
from time import time
from scipy import ndimage
from fluidlab.configs.macros import *

@ti.data_oriented
class GGUIRenderer:
    def __init__(self, 
        res=(640, 640),
        camera_pos=(0.5, 2.5, 3.5),
        camera_lookat=(0.5, 0.5, 0.5),
        fov=30,
        mode='human',
        particle_radius=0.0075,
        lights=[{'pos': (0.5, 1.5, 0.5), 'color': (0.5, 0.5, 0.5)},
                {'pos': (0.5, 1.5, 1.5), 'color': (0.5, 0.5, 0.5)}],
    ):
        self.res = res
        self.camera_pos = np.array(camera_pos)
        self.camera_lookat = np.array(camera_lookat)
        self.camera_vec = self.camera_pos - self.camera_lookat
        self.camera_init_xz_rad = np.arctan2(self.camera_vec[0], self.camera_vec[2])
        self.lights = []
        self.uninit = True

        for light in lights:
            self.add_light(light['pos'], light['color'])

        self.fov = fov
        self.particle_radius = particle_radius
        self.frame = ti.Vector.field(3, dtype=ti.f32, shape=(9,))
        self.frames = [ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,))]
        self.color_target = None

    def add_light(self, pos, color=(0.5, 0.5, 0.5)):
        light = {
            'pos': pos,
            'color': color
        }
        self.lights.append(light)

    def build(self, sim, particles):
        self.sim = sim
        self.particles_color = ti.Vector.field(4, ti.f32, shape=(len(particles['color']),))
        self.particles_color.from_numpy(particles['color'].astype(np.float32))

        for i in range(200):
            self.frames[0][i] = ti.Vector([0., 0., 0.]) + i/200 * ti.Vector([1., 0., 0.])
            self.frames[1][i] = ti.Vector([0., 0., 0.]) + i/200 * ti.Vector([0., 1., 0.])
            self.frames[2][i] = ti.Vector([0., 0., 0.]) + i/200 * ti.Vector([0., 0., 1.])
            self.frames[3][i] = ti.Vector([1., 1., 1.]) + i/200 * ti.Vector([-1., 0., 0.])
            self.frames[4][i] = ti.Vector([1., 1., 1.]) + i/200 * ti.Vector([0., -1., 0.])
            self.frames[5][i] = ti.Vector([1., 1., 1.]) + i/200 * ti.Vector([0., 0., -1.])
            self.frames[6][i] = ti.Vector([0., 1., 0.]) + i/200 * ti.Vector([1., 0., 0.])
            self.frames[7][i] = ti.Vector([0., 1., 0.]) + i/200 * ti.Vector([0., 0., 1.])
            self.frames[8][i] = ti.Vector([1., 0., 0.]) + i/200 * ti.Vector([0., 1., 0.])
            self.frames[9][i] = ti.Vector([1., 0., 0.]) + i/200 * ti.Vector([0., 0., 1.])
            self.frames[10][i] = ti.Vector([0., 0., 1.]) + i/200 * ti.Vector([1., 0., 0.])
            self.frames[11][i] = ti.Vector([0., 0., 1.]) + i/200 * ti.Vector([0., 1., 0.])

        # timer
        self.t = time()

    def update_fps(self):
        self.fps = 1.0 / (time() - self.t)
        self.t = time()

    def update_camera(self, t=None, rotate=False):
        self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)

        speed = 1e-2
        if self.window.is_pressed(ti.ui.UP):
            camera_dir = np.array(self.camera.curr_lookat - self.camera.curr_position)
            camera_dir[1] = 0
            camera_dir /= np.linalg.norm(camera_dir)
            new_camera_pos = np.array(self.camera.curr_position) + camera_dir * speed
            new_camera_lookat = np.array(self.camera.curr_lookat) + camera_dir * speed
            self.camera.position(*new_camera_pos)
            self.camera.lookat(*new_camera_lookat)
        elif self.window.is_pressed(ti.ui.DOWN):
            camera_dir = np.array(self.camera.curr_lookat - self.camera.curr_position)
            camera_dir[1] = 0
            camera_dir /= np.linalg.norm(camera_dir)
            new_camera_pos = np.array(self.camera.curr_position) - camera_dir * speed
            new_camera_lookat = np.array(self.camera.curr_lookat) - camera_dir * speed
            self.camera.position(*new_camera_pos)
            self.camera.lookat(*new_camera_lookat)
        elif self.window.is_pressed('u'):
            camera_dir = np.array([0, 1, 0])
            new_camera_pos = np.array(self.camera.curr_position) + camera_dir * speed
            new_camera_lookat = np.array(self.camera.curr_lookat) + camera_dir * speed
            self.camera.position(*new_camera_pos)
            self.camera.lookat(*new_camera_lookat)
        elif self.window.is_pressed('i'):
            camera_dir = np.array([0, -1, 0])
            new_camera_pos = np.array(self.camera.curr_position) + camera_dir * speed
            new_camera_lookat = np.array(self.camera.curr_lookat) + camera_dir * speed
            self.camera.position(*new_camera_pos)
            self.camera.lookat(*new_camera_lookat)

        # rotate
        if rotate and t is not None:
            speed = 7.5e-4
            xz_radius = np.linalg.norm([self.camera_vec[0], self.camera_vec[2]])
            rad = speed * np.pi * t + self.camera_init_xz_rad
            x = xz_radius * np.sin(rad)
            z = xz_radius * np.cos(rad)
            new_camera_pos = np.array([
                    x + self.camera_lookat[0],
                    self.camera_pos[1],
                    z + self.camera_lookat[2]]) 
            self.camera.position(*new_camera_pos)

        self.scene.set_camera(self.camera)

    def init_window(self, mode):
        show_window = mode == 'human'
        self.window = ti.ui.Window('FluidLab', self.res, vsync=True, show_window=show_window)

        self.canvas = self.window.get_canvas()
        self.canvas.set_background_color((1,1,1))
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()
        self.camera.position(*self.camera_pos)
        self.camera.lookat(*self.camera_lookat)
        self.camera.fov(self.fov)


    def render_frame(self, mode='human', x_target=None, t=0):
        if self.uninit:
            self.uninit = False
            self.init_window(mode)

        self.update_camera(t)

        # # reference frame
        # for i in range(12):
        #     self.scene.particles(self.frames[i], color=COLOR[FRAME], radius=self.particle_radius*0.5)
            
        # particles
        if self.sim.has_particles:
            state = self.sim.get_state_render(self.sim.cur_substep_local)
            x_particles = state.x
            self.scene.particles(x_particles, per_vertex_color=self.particles_color, radius=self.particle_radius)

            # target particles
            if x_target is not None:
                if self.color_target is None:
                    self.color_target = ti.Vector.field(4, ti.f32, x_target.shape)
                    self.color_target.from_numpy(np.repeat(np.array([COLOR[TARGET]]).astype(np.float32), x_target.shape[0], axis=0))
                self.scene.particles(x_target, per_vertex_color=self.color_target, radius=self.particle_radius)

        # statics
        if len(self.sim.statics) != 0:
            for static in self.sim.statics:
                self.scene.mesh(static.vertices, static.faces, per_vertex_color=static.colors)

        # effectors
        if self.sim.agent is not None and self.sim.agent.n_effectors != 0:
            for effector in self.sim.agent.effectors:
                if effector.mesh is not None:
                    self.scene.mesh(effector.mesh.vertices, effector.mesh.faces, per_vertex_color=effector.mesh.colors)
                # self.scene.particles(effector.latest_pos, color=COLOR[EFFECTOR], radius=self.particle_radius*2)

        # smoke
        if self.sim.smoke_field is not None:
            self.scene.particles(self.sim.smoke_field.vis_particles, per_vertex_color=self.sim.smoke_field.vis_particles_c, radius=1.0/128)

        for light in self.lights:
            self.scene.point_light(pos=light['pos'], color=light['color'])

        self.canvas.scene(self.scene)

        # camera gui
        # if True:
        #     self.window.GUI.begin("Camera", 0.05, 0.1, 0.2, 0.15)
        #     self.window.GUI.text(f'pos:    {self.camera.curr_position[0]:.2f}, {self.camera.curr_position[1]:.2f}, {self.camera.curr_position[2]:.2f}')
        #     self.window.GUI.text(f'lookat: {self.camera.curr_lookat[0]:.2f}, {self.camera.curr_lookat[1]:.2f}, {self.camera.curr_lookat[2]:.2f}')
        #     self.window.GUI.end()

        if mode == 'human':
            self.window.show()
            return None

        elif mode == 'rgb_array':
            img = np.rot90(self.window.get_image_buffer_as_numpy())[:, :, :3]
            img = (img * 255).astype(np.uint8)
            self.update_fps()
            # print(f'===> GGUIRenderer: {self.fps:.2f} FPS')
            return img





