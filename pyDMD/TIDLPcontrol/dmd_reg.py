import itertools
import numpy as np
from scipy.ndimage import label, find_objects
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import re
from skimage import transform as tf
from random import random

class DMDBeamCorrector(object):

    @staticmethod
    def gauss(X,a,x0,y0,sx,sy):
        x = X.T[0]
        y = X.T[1]
        return a*np.exp(-((x-x0)**2)/(2*sx)-((y-y0)**2)/(2*sy))

    @staticmethod
    def gaussth(X,a,x0,y0,sx,sy,th):
        xt = X.T[0]
        yt = X.T[1]
        x = xt * np.cos(th) - yt * np.sin(th)
        y = yt * np.cos(th) + xt * np.sin(th)
        return a*np.exp(-((x-x0)**2)/(2*sx)-((y-y0)**2)/(2*sy))

    @staticmethod
    def d2((x0,y0), (x1,y1)):
        return (x1-x0)**2+(y1-y0)**2

    @staticmethod
    def read_pgm(filename, byteorder='>'):
        """Return image data from a raw PGM file as numpy array.

        Format specification: http://netpbm.sourceforge.net/doc/pgm.html

        """
        with open(filename, 'rb') as f:
            buffer = f.read()
        try:
            header, width, height, maxval = re.search(
                b"(^P5\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
        except AttributeError:
            raise ValueError("Not a raw PGM file: '%s'" % filename)
        return np.frombuffer(buffer,
                                dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                                count=int(width)*int(height),
                                offset=len(header)
                                ).reshape((int(height), int(width)))

    def __init__(self, calib_fname, beam_fname, calib_thresh=10, center=(1920/2,1080/2)):
        self.calib = self.read_pgm(calib_fname)
        self.beam = self.read_pgm(beam_fname)
        self.calib_thresh = calib_thresh
        self.transform = None
        self.beam_fixed = None
        self.center = center

    def calc_transform(self):
        points = self.identify_points()
        threes, fours = self.identify_axes_points(points)


        dst = np.array((
            fours[1],
            fours[3],
            threes[1],
            threes[2],
        ))
        src = np.array((
            (100, -100),
            (-100, 100),
            (-100, -100),
            (100, 100)
        ))+self.center

        tform3 = tf.ProjectiveTransform()
        tform3.estimate(src, dst)
        self.transform = tform3

    def homogenize(self, arr, w_center=(1920/2, 1080/2), w_size=(1920/2, 1080/2)):
        if self.transform is None:
            self.calc_transform()
            self.beam_fixed = tf.warp(self.beam, self.transform, output_shape=(1080, 1920))
            self.beam_fixed /= np.float(np.max(self.beam_fixed))


        small_beam = self.beam_fixed[::4,::4]

        fit = curve_fit(self.gaussth,
                  np.array(
                    np.meshgrid(range(small_beam.shape[0]),
                                range(small_beam.shape[1]))).T.reshape(-1,2),
                  small_beam.flatten(),
                  (1, 758/4, 699/4, 500/4, 500/4, 0))

        fit[0][1] -= 25
        fit[0][2] -= 25
        fit[0][3] /= 2

        fitres = self.gaussth(np.array(np.meshgrid(range(small_beam.shape[0]),
                                range(small_beam.shape[1]))).T.reshape(-1,2),
                        *fit[0]).reshape((small_beam.shape[0], small_beam.shape[1]))

        fitres = fitres.repeat(4, axis=0).repeat(4, axis=1)
        # plt.pcolormesh(self.beam_fixed-fitres)
        # plt.show()

        res = arr/(fitres+0.0001)**0.8
        res /= np.max(res[
            w_center[0]-w_size[0]:w_center[0]+w_size[0],
            w_center[1]-w_size[1]:w_center[1]+w_size[1]])
        res[res > 1] = 1
        f = self.read_pgm('res.pgm')
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, sharey=True, figsize=(3.5, 8))
        ax1.imshow(arr, interpolation='none')
        # plt.imshow(tf.warp(f, self.transform, output_shape=(1080, 1920)), interpolation='none')
        # plt.figure()
        ax2.imshow(fitres)
        ax3.imshow(res, interpolation='none')
        ax4.imshow(res*fitres**0.8, interpolation='none')
        plt.show()
        return res

    def identify_points(self):
        thresh = np.where(self.calib > self.calib_thresh, 1, 0)
        thresh[0:5,0:5] = 0
        f = find_objects(label(thresh)[0])

        print len(f)

        points = []
        print len(f)
        for obj in f:
            x1 = obj[0].start-2
            x2 = obj[0].stop+2
            y1 = obj[1].start-2
            y2 = obj[1].stop+2

            cx = (x1+x2)/2
            cy = (y1+y2)/2
            sx = (x2-x1)/2
            sy = (y2-y1)/2

            coords = np.array(np.meshgrid(range(x1, x2), range(y1, y2))).T.reshape(-1,2)

            res, cov = curve_fit(self.gauss,
                                 coords,
                                 self.calib[x1:x2, y1:y2].flatten(),
                                 (self.calib[cx,cy], cx, cy, sx, sy))
            points.append((res[2], res[1]))

        return points

    def identify_axes_points(self, points):
        for ((ai,a),(bi,b),(ci,c),(di,d)) in itertools.combinations(enumerate(points), 4):
            slope = (b[1]-a[1])/(b[0]-a[0])
            yp1 = slope*(c[0]-a[0])+a[1]
            yp2 = slope*(d[0]-a[0])+a[1]

            if abs(float(yp1-c[1])/yp1) < 0.01 and abs(float(yp2-d[1])/yp2) < 0.01:
                threes = sorted(map(lambda i: points[i], filter(lambda x: x not in [ai,bi,ci,di], [0,1,2,3,4,5,6])))
                if self.d2(threes[0], threes[1]) > self.d2(threes[1], threes[2]):
                    threes = list(reversed(threes))

                fours = sorted([a,b,c,d])
                if self.d2(a, b) > self.d2(c, d):
                    fours = list(reversed(fours))

                return threes, fours

        raise Exception("Cannot find the required collinear points")

if __name__ == '__main__':
    corrector = DMDBeamCorrector('multipoint.pgm', 'beam.pgm')
    arr = np.zeros((1080,1920))
    y, x = np.ogrid[0:1080, 0:1920]
    m1 = ((y-1080/2)**2+(x-1920/2)**2<=30**2)
    arr[m1] = 1
    arr = corrector.homogenize(arr)

    for y in range(arr.shape[1]):
        for x in range(arr.shape[0]):
            # custom floyd-steinberg
            op = arr[x][y]
            if random() < 0.5:
                newp = 1 if random() < op else 0
            else:
                newp= 1 if 0.5 < op else 0
            arr[x][y] = newp
            err = op - newp
            if x<arr.shape[0]-1:
                arr[x+1][y] = arr[x+1][y] + err * 7/16
                if y<999:
                    arr[x+1][y+1] = arr[x+1][y+1] + err * 1/16
            if y<arr.shape[1]-1:
                arr[x-1][y+1] = arr[x-1][y+1] + err * 3/16
                arr[x][y+1] = arr[x][y+1] + err * 5/16