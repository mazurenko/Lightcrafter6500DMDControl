import numpy as np
cimport numpy as cnp


cnp.import_array()
cimport cython


def error_diffuse(cnp.ndarray[double, ndim=2] img,
                  cnp.ndarray[double, ndim=2] threshold,
                  cnp.ndarray[int, ndim=1] dx,
                  cnp.ndarray[int, ndim=1] dy,
                  cnp.ndarray[double] error_frac):
    cdef:
        ssize_t img_x, img_y
        ssize_t rows_ahead, cols_ahead
        cnp.ndarray[double, ndim = 2] out

    img_x = img.shape[0]
    img_y = img.shape[1]
    out = np.zeros((img_x, img_y))
    for i in xrange(img_x):
        for j in xrange(img_y):
            rows_ahead = img_x - 1 - i
            cols_ahead = img_y - 1 - j
            out[i, j] = 1.0 if img[i, j] > 0.5 else 0.0 #threshold[i, j] else 0
            err = img[i, j] - out[i, j]
            for idx in range(error_frac.shape[0]):
                if dx[idx] <= cols_ahead and dx[idx] + j >= 0 and dy[idx] <= rows_ahead:
                    img[i + dy[idx], j + dx[idx]] -= error_frac[idx] * err
    return out

    '''
            out[i, j] = 1.0 if img[i, j] > threshold[i, j] else 0.0
            err = out[i, j] - 1.0 * img[i, j]
            for idx in range(error_frac.shape[0]):
                if dx[idx] <= cols_ahead and dx[idx] + j >= 0 and dy[idx] <= rows_ahead:
                    img[i + dy[idx], j + dx[idx]] -= error_frac[idx] * err
    return img
'''
def convolve2d(cnp.ndarray[float, ndim=2] f,
               cnp.ndarray[float, ndim=2] g):
    cdef:
        ssize_t vmax, wmax, smax, tmax, smid, tmid, xmax, ymax
        ssize_t s_from, s_to, t_from, t_to
        ssize_t x, y, s, t, v, w
        float value
        cnp.ndarray[float, ndim=2] h
    # f is an image and is indexed by (v, w)
    # g is a filter kernel and is indexed by (s, t),
    #   it needs odd dimensions
    # h is the output image and is indexed by (x, y),
    #   it is not cropped
    if g.shape[0] % 2 != 1 or g.shape[1] % 2 != 1:
        raise ValueError("Only odd dimensions on filter supported")
    # smid and tmid are number of pixels between the center pixel
    # and the edge, ie for a 5x5 filter they will be 2.
    #
    # The output size is calculated by adding smid, tmid to each
    # side of the dimensions of the input image.
    vmax = f.shape[0]
    wmax = f.shape[1]
    smax = g.shape[0]
    tmax = g.shape[1]
    smid = smax // 2
    tmid = tmax // 2
    xmax = vmax + 2 * smid
    ymax = wmax + 2 * tmid
    # Allocate result image.
    h = np.zeros([xmax, ymax], dtype=f.dtype)
    # Do convolution
    for x in range(xmax):
        for y in range(ymax):
            # Calculate pixel value for h at (x,y). Sum one component
            # for each pixel (s, t) of the filter g.
            s_from = max(smid - x, -smid)
            s_to = min((xmax - x) - smid, smid + 1)
            t_from = max(tmid - y, -tmid)
            t_to = min((ymax - y) - tmid, tmid + 1)
            value = 0
            for s in range(s_from, s_to):
                for t in range(t_from, t_to):
                    v = x - smid + s
                    w = y - tmid + t
                    value += g[smid - s, tmid - t] * f[v, w]
            h[x, y] = value
    return h

