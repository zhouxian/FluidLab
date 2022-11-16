from gl_renderer_src import flex_renderer
from time import time
import numpy as np
import cv2

cam_cfg = {
 'camera_pos': [0, 5, 0],
 'camera_angle': [0, -1.5707963267948966, 0],
}
W = 1280
H = 720
msaaSamples = 8
fov = 0.7

def main():
    n_particles = 100000
    flex_renderer.init(W, H, msaaSamples, fov)
    render_particle = False
    flex_renderer.create_scene(render_particle, n_particles)
    flex_renderer.set_camera_params(cam_cfg)

    t = time()
    p = np.random.uniform([0,0,0,1.0], [1,1,1,1.0], (n_particles, 4))
    while True:
        p[:, 0] += 0.002
        flex_renderer.set_positions(p.flatten())
        img = get_observation()

        cv2.imshow('img', img[..., ::-1])
        cv2.waitKey(1)

        print(f'{1/(time()-t):.2f} fps')
        t = time()



def get_observation(particle_view=False):
    color_img = flex_renderer.render()
    color_img = np.flip(color_img.reshape([H, W, 4]), 0)[:, :, :3]
    return color_img


if __name__=='__main__':
    main()