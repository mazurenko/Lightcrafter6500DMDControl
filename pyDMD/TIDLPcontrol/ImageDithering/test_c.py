import numpy as np
import pyximport
pyximport.install(setup_args={"script_args":["--compiler=mingw32"],
                              "include_dirs":np.get_include()},
                  reload_support=True)

import dither_c
import skimage
from skimage import data
import imgDither
import matplotlib.pyplot as plt
import time

#print dither_c.error_diffuse()

def _get_diffusion_lists(propagator_list):
    return (np.array(map(lambda x: x[0][0], propagator_list), dtype=int),
            np.array(map(lambda x: x[0][1], propagator_list), dtype=int),
            np.array(map(lambda x: x[1], propagator_list), dtype=float))

if __name__ == "__main__":
    image = skimage.img_as_float(np.vstack((np.hstack((data.camera(), data.moon())),
                                            np.hstack((data.moon(), data.moon())))))

    propagator = imgDither.ErrorDiffusion.diffusion_algorithm_dict['floyd_steinberg']
    dx, dy, diff_err = _get_diffusion_lists(propagator)
    threshold = 0.5 * np.ones_like(image)

    t_i = time.time()
    diffused = dither_c.error_diffuse(image, threshold, dx, dy, diff_err)
    t_f = time.time()
    print np.shape(image)
    print diffused
    print t_f - t_i

    plt.gray()
    plt.imshow(diffused, interpolation='none')
    plt.show()
