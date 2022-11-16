import os
import cv2
import time
import skimage
import numpy as np
import taichi as ti
from time import time
from scipy import ndimage
from fluidlab.configs.macros import *
from .gl_renderer_src import flex_renderer
import fluidlab.utils.geom as geom_utils
import fluidlab.utils.misc as misc_utils

class GLRenderer:
    def __init__(self, 
        res             = (1280, 1280),
        camera_pos      = (0.5, 2.5, 3.5),
        camera_lookat   = (0.5, 0.5, 0.5),
        camera_near     = 0.1,
        camera_far      = 10.0,
        fov             = 30,
        particle_radius = 0.01,
        smoke_radius    = 0.01,
        render_particle = False,
        light_pos       = (0.5, 5.0, 0.5),
        light_lookat    = (0.5, 0.5, 0.49),
        light_fov       = 50,
        floor_height    = 0.0,
        scene_radius    = 10.0,
        cam_rotate_v    = 0.0,
        _smoothing      = 0.5,
    ):
        self.res                = res
        self.camera_pos         = np.array(camera_pos)
        self.camera_lookat      = np.array(camera_lookat)
        self.camera_vec         = self.camera_pos - self.camera_lookat
        self.camera_init_xz_rad = np.arctan2(self.camera_vec[0], self.camera_vec[2])
        self.camera_angle       = geom_utils.compute_camera_angle(camera_pos, camera_lookat)
        self.camera_near        = camera_near
        self.camera_far         = camera_far
        self.lights             = []
        self.render_particle    = render_particle
        self.fov                = fov / 180.0 * np.pi
        self.particle_radius    = particle_radius
        self.smoke_radius       = smoke_radius
        self.light_pos          = np.array(light_pos)
        self.light_lookat       = np.array(light_lookat)
        self.light_fov          = light_fov
        self.floor_height       = floor_height
        self.scene_radius       = scene_radius
        self.cam_rotate_v       = cam_rotate_v

        self._msaaSamples = 8
        self._anisotropy_scale = 1.0
        self._smoothing = _smoothing
        self._rendering_scale = 2.0
        self._fluid_rest_distance = 0.0125
        self._gl_color_gamma = 3.5

        self.aa = []

        flex_renderer.init(self.res[0], self.res[1], self._msaaSamples, self.fov)


    def build(self, sim, particles):
        self.sim = sim
        self.n_particles = sim.n_particles

        # compute bodies info
        bodies_color = []
        bodies_needs_smoothing = []
        bodies_particle_offset = []
        for body_particle_ids in particles['bodies']['particle_ids']:
            body_particle_offset = body_particle_ids[0]
            bodies_particle_offset.append(body_particle_offset)
            bodies_color.append(misc_utils.alpha_to_transparency(particles['color'][body_particle_offset]))
            bodies_needs_smoothing.append(MAT_CLASS[particles['mat'][body_particle_offset]] == MAT_LIQUID)

        bodies_n_particles     = np.array(particles['bodies']['n_particles'])
        bodies_color           = np.array(bodies_color).flatten()
        bodies_needs_smoothing = np.array(bodies_needs_smoothing)
        bodies_particle_offset = np.array(bodies_particle_offset)

        n_bodies = particles['bodies']['n']
        assert n_bodies == sim.n_bodies
        assert np.sum(bodies_n_particles) == self.n_particles

        # create scene
        flex_renderer.create_scene(
            self.render_particle,
            self.particle_radius,
            self.smoke_radius,
            self._anisotropy_scale,
            self._smoothing,
            self._gl_color_gamma,
            self.scaled(self._fluid_rest_distance),
            self.scaled(self.light_pos),
            self.scaled(self.light_lookat),
            self.light_fov,
            self.scaled(self.floor_height),
            self.scaled(self.scene_radius),
            bodies_n_particles,
            bodies_particle_offset,
            bodies_color,
            bodies_needs_smoothing,
            n_bodies,
        )

        # add statics
        if len(self.sim.statics) != 0:
            for static in self.sim.statics:
                colors         = static.colors_np.flatten()
                faces          = static.faces_np
                static.gl_renderer_id = flex_renderer.add_mesh(static.n_vertices, static.n_faces, colors, faces)

        # add effectors
        if self.sim.agent is not None and self.sim.agent.n_effectors != 0:
            for effector in self.sim.agent.effectors:
                if effector.mesh is not None:
                    colors         = effector.mesh.colors_np.flatten()
                    faces          = effector.mesh.faces_np
                    effector.mesh.gl_renderer_id = flex_renderer.add_mesh(effector.mesh.n_vertices, effector.mesh.n_faces, colors, faces)

        # add smoke particles
        if self.sim.smoke_field is not None:
            num_smoke_particles = self.sim.smoke_field.vis_particles.shape[0]
            positions           = self.sim.smoke_field.vis_particles.to_numpy().flatten()
            colors              = self.sim.smoke_field.vis_particles_c.to_numpy().flatten()
            positions_scaled    = self.scaled(positions)
            flex_renderer.add_smoke_particles(num_smoke_particles, positions_scaled, colors)

        # camera
        flex_renderer.set_camera_params(
            self.scaled(self.camera_pos),
            self.camera_angle,
            self.scaled(self.camera_near),
            self.scaled(self.camera_far),
        )

        # timer
        self.t = time()

    def rotate_camera(self, t=None):
        if t is not None:
            xz_radius = np.linalg.norm([self.camera_vec[0], self.camera_vec[2]])
            rad = self.cam_rotate_v * np.pi * t + self.camera_init_xz_rad
            x = xz_radius * np.sin(rad)
            z = xz_radius * np.cos(rad)
            new_camera_pos = np.array([
                    x + self.camera_lookat[0],
                    self.camera_pos[1],
                    z + self.camera_lookat[2]]) 
            new_camera_angle = geom_utils.compute_camera_angle(new_camera_pos, self.camera_lookat)

            flex_renderer.set_camera_params(
                self.scaled(new_camera_pos),
                new_camera_angle,
                self.scaled(self.camera_near),
                self.scaled(self.camera_far),
            )

    def scaled(self, x):
        return x * self._rendering_scale

    def update_fps(self):
        self.fps = 1.0 / (time() - self.t)
        self.t = time()

    def render_frame(self, mode='human', x_target=None, t=0):
        self.rotate_camera(t)

        # particles
        if self.sim.has_particles:
            state_render = self.sim.get_state_render(self.sim.cur_substep_local)
            positions    = state_render.x.to_numpy()
            used         = state_render.used.to_numpy()
            flex_renderer.set_particles_state(self.scaled(positions.flatten()), used)

        # statics
        if len(self.sim.statics) != 0:
            for static in self.sim.statics:
                mesh_render_id = static.gl_renderer_id
                vertices       = static.init_vertices_np_flattened
                vertex_normals = static.init_vertex_normals_np_flattened
                flex_renderer.update_mesh(
                    mesh_render_id,
                    self.scaled(vertices),
                    vertex_normals,
                )

        # effectors
        if self.sim.agent is not None and self.sim.agent.n_effectors != 0:
            for effector in self.sim.agent.effectors:
                if effector.mesh is not None:
                    mesh_render_id = effector.mesh.gl_renderer_id
                    vertices       = effector.mesh.vertices.to_numpy().flatten()
                    vertex_normals = effector.mesh.vertex_normals.to_numpy().flatten()
                    flex_renderer.update_mesh(
                        mesh_render_id,
                        self.scaled(vertices),
                        vertex_normals,
                    )

        # smoke field
        if self.sim.smoke_field is not None:
            colors = self.sim.smoke_field.vis_particles_c.to_numpy()
            flex_renderer.update_smoke_particles(colors)

        # render
        img = flex_renderer.render()
        img = np.flip(img.reshape([self.res[1], self.res[0], 4]), 0)[:, :, :3]

        self.update_fps()

        if mode == 'human':
            cv2.imshow('FluidLab', img[..., ::-1])
            cv2.setWindowTitle('FluidLab', f'FluidLab [FPS: {self.fps:.2f}]')
            cv2.waitKey(1)
            return None

        elif mode == 'rgb_array':
            print(f'===> GLRenderer: {self.fps:.2f} FPS')
            cv2.imshow('FluidLab', img[..., ::-1])
            cv2.setWindowTitle('FluidLab', f'FluidLab [FPS: {self.fps:.2f}]')
            cv2.waitKey(1)
            return img






