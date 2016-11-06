import abc
import numpy as np
import numpy.random as rand
import numpy.fft as fft
import matplotlib.pyplot as plt
import pickle as pkl
import skimage
from skimage import data
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import *


__author__ = 'anton'


class DitheringError(Exception):
    def __init__(self, error_msg):
        self.error_msg = error_msg

    def __str__(self):
        return repr(self.error_msg)

class FourierMask(object):

    type_dict = {'low-pass': lambda X, Y, r: np.sqrt(np.square(X/r[0]) + np.square(Y/r[1])) < 1,
                 'hi-pass': lambda X, Y, r: np.sqrt(np.square(X/r[0]) + np.square(Y/r[1])) > 1}

    def __init__(self, shape, **kwargs):
        self.center = (shape[0]/2.0, shape[1]/2.0)
        self.df = (1.0/shape[0], 1.0/shape[1])
        self.f_cutoff = kwargs.pop('f_cutoff', 0)
        self.mask_type = kwargs.pop('type', 'low-pass')
        self.custom_filter = kwargs.pop('custom_filter', None)
        X, Y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
        if self.mask_type in FourierMask.type_dict.keys():
            self.mask = FourierMask.type_dict[self.mask_type](X-self.center[0], Y-self.center[1], (self.f_cutoff/self.df[0],
                                                                                               self.f_cutoff/self.df[1]))
        elif self.mask_type == "custom" and (self.custom_filter is not None):
            self.mask = self.custom_filter(X-self.center[0], Y-self.center[0])
        else:
            raise Exception("No filter defined!!!")

    def apply_fourier_mask(self, img):
        masked = np.multiply(self.mask, fft.fftshift(fft.fft2(img)))
        return np.abs(fft.ifft2(fft.fftshift(masked)))

class DiffractionLimitedImaging(FourierMask):
    def __init__(self, img, **kwargs):
        kwargs.update('type', 'low-pass')
        self.img = img
        self.NA = kwargs.pop('NA', 0.86/160) # note, this is NA looking along this system .86/160
        self.pix_size = kwargs.pop('pixel_size_um', 7.6)  # um
        self.wav = kwargs.pop('wavelength_um', 0.671)  # um
        super(DiffractionLimitedImaging).__init__(np.shape(self.image), kwargs)
        self.processed_img = self.apply_fourier_mask(self.img)

class DitheringAlgorithmABC(object):

    @abc.abstractmethod
    def process_img(self, img, **kwargs):
        return

    @abc.abstractmethod
    def show_img(self, directory, filename, is_save_pkl=False):
        return

    @abc.abstractmethod
    def fft_img(self):
        return

    @abc.abstractmethod
    def __str__(self):
        return


class ImageHandler(DitheringAlgorithmABC):

    def __init__(self, img, **kwargs):
        self.name = 'identity'
        self.settings = kwargs
        self.is_rescale = kwargs.get('is_rescale', False)
        if self.is_rescale:
            tmp_img = ImageHandler.rescale_image(img.astype(float))
        else:
            tmp_img = img.astype(float)
        eps = 1e-6
        assert np.max(tmp_img) <= 1.0+eps
        assert np.min(tmp_img) >= 0.0-eps
        self.img = tmp_img

        self.processed_img = self.process_img(self.img)

    def __str__(self):
        return 'OPERATION = %s \n SETTINGS = %s'% (self.name, self.settings)

    def process_img(self, img):
        return img

    @staticmethod
    def _plt_save(directory, filename):
        plt.savefig('%s\\%s.png'% (directory, filename))
        pp = PdfPages('%s\\%s.pdf' % (directory, filename))
        pp.savefig(transparent=True, dpi=300)
        pp.close()

    def show_img(self, **kwargs):
        is_save_pkl = kwargs.get('is_save_pkl', False)
        is_save_img = kwargs.get('is_save_img', False)
        directory = kwargs.get('directory', 'output')
        filename = kwargs.get('filename', self.name)

        if is_save_pkl:
            image_dict = {'settings': self.settings,
                          'original': self.img,
                          'processed': self.processed_img}
            pkl.dump(image_dict, open('%s\\%s.pkl'% (directory, filename), 'rb'))
            print "saved pickle file"

        plt.gray()
        plt.imshow(self.img)
        plt.axis('off')
        if is_save_img:
            ImageHandler._plt_save(directory, filename)

    def show_dither_full(self, **kwargs):
        is_save_img = kwargs.get('is_save_img', False)
        directory = kwargs.get('directory', 'output')
        filename = kwargs.get('filename', self.name)
        is_show = kwargs.get('is_show', True)
        zoom_range = kwargs.get('zoom_range', None)

        img_fft, processed_fft = self.fft_img()

        plt.figure(figsize=(10,8))
        plt.subplots_adjust(top=.85)
        plt.gray()
        plt.subplot(2, 2, 1)
        if zoom_range is not None:
            plt.imshow(self.img[zoom_range[0][0]:zoom_range[0][1],
                                zoom_range[1][0]:zoom_range[1][1]],
                       interpolation='none')
        else:
            plt.imshow(self.img, interpolation='none')
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(2, 2, 2)
        if zoom_range is not None:
            plt.imshow(self.processed_img[zoom_range[0][0]:zoom_range[0][1],
                                zoom_range[1][0]:zoom_range[1][1]],
                       interpolation='none')
        else:
            plt.imshow(self.processed_img, interpolation='none')

        plt.title('Processed Image')
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.imshow(self.fft_intensity_db(img_fft))
        plt.title('Original Image FFT')
        plt.axis('off')
        plt.subplot(2, 2, 4)
        plt.imshow(self.fft_intensity_db(processed_fft))
        plt.title('Processed Image FFT')

        plt.suptitle(self.__str__(), fontsize='14')
        plt.axis('off')

        if is_save_img:
            ImageHandler._plt_save(directory, filename)
        if is_show:
            plt.show()
        plt.close()

    def fft_img(self):
        return fft.fftshift(fft.fft2(self.img)), fft.fftshift(fft.fft2(self.processed_img))

    @staticmethod
    def rescale_image(img):
        """
        :param img: float array image
        :return: same array, but normalized to 0 - 1 range
        """
        img_max = np.max(img)
        img_min = np.min(img)
        zeroed = img - img_min
        if img_max - img_min < 1e-6:
            return zeroed
        else:
            return np.divide(zeroed, img_max - img_min)

    @staticmethod
    def fft_intensity_db(fft_img):
        return 10*np.log10(np.square(np.abs(fft_img)))


class ThresholdDithering(ImageHandler):
    """
    This class captures binarization by threshold and random noise - the random noise is optionally
    added to the threshold
    """
    randomization_function_dict = {'uniform':lambda randomness, size: rand.uniform(low = -1.0*randomness,
                                                                                   high= randomness,
                                                                                   size=size),
                                   'gaussian':lambda randomness, size: rand.normal(loc=0,
                                                                                   scale = randomness,
                                                                                   size=size)}
    #all functions must be (a) cenerted about 0, (b) take the randomness paramete

    def __init__(self, img, **kwargs):
        self.threshold_value = kwargs.get('threshold_value', 0.5)
        self.randomness = kwargs.get('randomness', 0.0)
        self.random_statistics = kwargs.get('random_statistics', 'uniform')
        assert self.randomness>=0-1e-6
        assert self.random_statistics in ThresholdDithering.randomization_function_dict.keys()
        super(ThresholdDithering, self).__init__(img, **kwargs)
        self.name = 'randomized'

    def process_img(self, img):
        return ThresholdDithering.threshold_image(img, threshold_value=self.threshold_value,
                                                randomness = self.randomness,
                                                random_statistics=self.random_statistics)

    @staticmethod
    def threshold_image(img, threshold_value = 0.5, randomness = 0.0, random_statistics = 'uniform'):
        if randomness > 1e-6:
            dither_matrix = ThresholdDithering.randomization_function_dict[random_statistics](randomness, np.shape(img))
        else:
            dither_matrix = np.zeros_like(img)
        return 1.0*dither_matrix + threshold_value * np.ones_like(img) < img


class ErrorDiffusion(ImageHandler):
    # Useful reference: http://www.tannerhelland.com/4660/dithering-eleven-algorithms-source-code/
    diffusion_algorithm_dict = {'floyd_steinberg': [((1, 0), 7.0/16.0),
                                                ((-1, 1), 3.0/16.0),
                                                ((0, 1), 5.0/16.0),
                                                ((1, 1), 1.0/16.0)],
                                'jarvis_judice_ninke':  [((1, 0), 7.0/48.0),
                                                ((2, 0), 5.0/48.0),
                                                ((-2, 1), 3.0/48.0),
                                                ((-1, 1), 5.0/48.0),
                                                ((0, 1), 7.0/48.0),
                                                ((1, 1), 5.0/48.0),
                                                ((2, 1), 3.0/48.0),
                                                ((-2, 2), 1.0/48.0),
                                                ((-1, 2), 3.0/48.0),
                                                ((0, 2), 5.0/48.0),
                                                ((1, 2), 3.0/48.0),
                                                ((2, 2), 1.0/48.0),
                                                ],
                                'sierra': [((1, 0), 5.0/32.0),
                                                ((2, 0), 3.0/32.0),
                                                ((-2, 1), 2.0/32.0),
                                                ((-1, 1), 4.0/32.0),
                                                ((0, 1), 5.0/32.0),
                                                ((1, 1), 4.0/32.0),
                                                ((2, 1), 2.0/32.0),
                                                ((-1, 2), 2.0/32.0),
                                                ((0, 2), 3.0/32.0),
                                                ((1, 2), 2.0/32.0)],
                                'two_row_sierra': [((1, 0), 4.0/16.0),
                                                ((2, 0), 3.0/16.0),
                                                ((-2, 1), 1.0/16.0),
                                                ((-1, 1), 2.0/16.0),
                                                ((0, 1), 3.0/16.0),
                                                ((1, 1), 2.0/16.0),
                                                ((2, 1), 1.0/16.0)
                                                   ]}

    def __init__(self, img, **kwargs):
        for key in ErrorDiffusion.diffusion_algorithm_dict:
            is_ok, bad_props = ErrorDiffusion.verify_error_diffusion_list(ErrorDiffusion.diffusion_algorithm_dict[key])
            if not is_ok:
                raise DitheringError('%s algorithm is invalid, bad propagators indexes: %s' % (key, bad_props))

        self.error_diffusion_type = kwargs.get('error_diffusion_type', 'floyd_steinberg')
        assert self.error_diffusion_type in ErrorDiffusion.diffusion_algorithm_dict.keys()
        self.diffusion_list = ErrorDiffusion.diffusion_algorithm_dict[self.error_diffusion_type]
        self.randomness = kwargs.get('randomness', 0.0)
        assert np.abs(self.randomness) < 1
        super(ErrorDiffusion, self).__init__(img, **kwargs)
        self.name = 'error_diffusion/%s' % self.error_diffusion_type

    def process_img(self, img):
        print "Applying %s error diffusion, this may take up to 1 minute:" % self.error_diffusion_type
        return ErrorDiffusion.apply_error_diffusion(img, self.diffusion_list, randomness=self.randomness)

    @staticmethod
    def verify_error_diffusion_list(list):
        """
        :param list: list of ((col, row), offset)
        :return: (True if valid error diffusion, False if not, None if previous is true, list of failed entries if yes)
        error diffusion only works if error is propagated forwards. This verifies that that is the case
        """
        failed_entries = []
        for idx, propagator in enumerate(list):
            if propagator[0][0] <=0 and propagator[0][1]<=0:
                failed_entries.append(idx)
                print "ERR: prop backward, idx: %s, propagator %s" %(idx, propagator)
            if propagator[1] <-1e-6:
                failed_entries.append(propagator)
                print "ERR: negative propagator, idx: %s, propagator %s" %(idx, propagator)
        if not failed_entries:
            return True, None
        else:
            return False, failed_entries

    @staticmethod
    def apply_error_diffusion(img, diffusion_list, randomness=0):
        img_shape = np.shape(img)
        tmp_img = np.copy(img)
        for i in tqdm(xrange(img_shape[0])):
            for j in xrange(img_shape[1]):
                rows_ahead = img_shape[0] - 1 - i
                cols_ahead = img_shape[1] - 1 - j
                threshold = .5 if randomness <1e-6 else rand.normal(0.5, randomness)
                bin = 1.0 * tmp_img[i, j] > threshold
                err = bin - 1.0 * tmp_img[i, j]
                tmp_img[i, j] = bin
                for propagator in diffusion_list:
                    if propagator[0][0] <= cols_ahead and propagator[0][0] + j >= 0 and propagator[0][1] <= rows_ahead:
                        tmp_img[i + propagator[0][1], j + propagator[0][0]] -= propagator[1] * err
                        pass
        return tmp_img


class BayerDither(ImageHandler):
    """
    Useful reference on bayer dithering:
    https://en.wikipedia.org/wiki/Ordered_dithering
    http://caca.zoy.org/study/part2.html
    """

    def __init__(self, img, **kwargs):
        self.bayer_order = kwargs.get('bayer_order', 4)
        self.randomness = kwargs.get('randomness', 0)
        super(BayerDither, self).__init__(img, **kwargs)
        assert np.abs(self.randomness) < 1
        self.name = 'Bayer'

    def process_img(self, img):
        return BayerDither.bayer_threshold(img, bayer_order=self.bayer_order, randomness=self.randomness)

    @staticmethod
    def __increase_bayer_order__(initial_bayer, seed_matrix):
        next_seed_mat = np.repeat(seed_matrix, 2, axis=0)
        next_seed_mat = np.repeat(next_seed_mat, 2, axis=1)

        bayer = 4*np.tile(initial_bayer, (2, 2))+next_seed_mat
        return bayer, next_seed_mat

    @staticmethod
    def generate_bayer_matrix(n, seed_matrix = np.array([[0, 3], [2, 1]])):
        """
        :param n: int - the size of the bayer matrix is 2^n x 2^n
        :param seed_matrix: 1st order seed matrix
        :return: numpy matrix bayer matrix
        """
        bayer = np.array([[0, 3],[2,1]])
        for order in range(n-1):
            bayer, seed_matrix = BayerDither.__increase_bayer_order__(bayer, seed_matrix)
        return 1.0/4**n*(np.array(bayer, dtype=float)+np.ones_like(bayer,dtype=float))

    @staticmethod
    def show_bayer_matrix(n, seed_matrix =  np.array([[0, 3], [2, 1]])):
        plt.gray()
        plt.imshow(BayerDither.generate_bayer_matrix(n, seed_matrix=seed_matrix))


    @staticmethod
    def bayer_threshold(img, bayer_order=1, randomness=0):
        """
        :param img: image to be thresholded
        :param bayer_order: order of matrix to use
        :param randomness: sigma of the gaussian noise on top of filter
        :return:
        """
        assert np.max(img) < 1.0+1e-6
        assert np.min(img) > 0.0-1e-6
        bayer = BayerDither.generate_bayer_matrix(bayer_order)

        image_shape = np.shape(img)
        bayer_side = 2 ** bayer_order
        tiled_bayer = np.tile(bayer, (image_shape[0]/bayer_side + 1, image_shape[1]/bayer_side + 1))

        cropped_bayer = tiled_bayer[0:image_shape[0], 0:image_shape[1]]
        if randomness > 1e-6:
            cropped_bayer += rand.normal(0, randomness, size=image_shape)
        thresholding_matrix = np.divide(img, tiled_bayer[0:image_shape[0], 0:image_shape[1]])
        return np.array(thresholding_matrix > 1, dtype=float)


def test_image_handling():
    image = skimage.img_as_float(data.camera())
    handler = ImageHandler(image, **{'is_rescale': True})
    ffted_img, processed_img = handler.fft_img()
    print ffted_img
    print handler.fft_intensity_db(ffted_img)

    plt.subplot(1, 2, 1)
    plt.imshow(handler.img)
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(handler.fft_intensity_db(ffted_img))
    plt.colorbar()
    plt.show()


def demo_error_diffusion():
    image = skimage.img_as_float(data.camera())
    for algorithm_name in ErrorDiffusion.diffusion_algorithm_dict.keys():
        print 'Diffusing using %s' % algorithm_name
        for rand in np.linspace(0, .5, 6):
            print "Analyzing randomness %s level" % rand
            handler = ErrorDiffusion(image, **{'is_rescale': True,
                                               'error_diffusion_type': algorithm_name,
                                               'randomness': rand})
            handler.show_dither_full(**{'zoom_range':((70, 200), (50, 350)),
                                        'is_save_img': True,
                                        'filename': 'diffusion_%s_rand_%s' % (algorithm_name, rand),
                                        'is_show': False})


def demo_threshold():
    image = skimage.img_as_float(data.camera())
    for algorithm_name in ThresholdDithering.randomization_function_dict.keys():
        print "binarizing using %s" % algorithm_name
        for sigma in np.linspace(0, .5, num=11):
            print "randomness = %s" % sigma
            handler = ThresholdDithering(image, **{'is_rescale': True,
                                                    'random_statistics':algorithm_name,
                                                    'randomness': sigma})
            handler.show_dither_full(**{'is_save_img': True,
                                        'filename': 'threshold_stats_%s_randomness_%s'% (handler.random_statistics,
                                                                                handler.randomness),
                                        'is_show':False})


def demo_bayer_filter():
    image = skimage.img_as_float(data.camera())
    max_order = 6

    order = 3
    handler = BayerDither(image, **{'is_rescale': True,
                                    'bayer_order': order,
                                    'randomness': 0.05})

    handler.show_dither_full(**{'is_save_img': False,
                              'filename': 'bayer%s'% order,
                              #'zoom_range':((400, 500), (200, 300)),
                              'is_show':True})
    print "dithered with order %s"% order


def demo_bayer_filter_randomized():
    image = skimage.img_as_float(data.camera())
    max_order = 6

    for order in range(1, max_order):
        handler = BayerDither(image, **{'is_rescale': True,
                                    'bayer_order': order,
                                    'randomness': .05})
        handler.show_dither_full(**{'is_save_img': True,
                              'filename': 'bayer%s_weak_random'% order,
                              'zoom_range':((400, 500), (200, 300)),
                              'is_show':False})
        print "dithered with order %s"% order

def demo_fft_filter():
    '''
    pixel  = 7.56 um
    NA = .86/160
    max spatial freq = NA/lambda
    :return:
    '''
    NA = .86/160
    wav = .671/7.56 # express in pixel units
    f_cutoff = NA/wav
    image = skimage.img_as_float(data.camera())
    image = np.zeros((1000, 1000))
    image[500, 500] = 1


    mask = FourierMask(np.shape(image), **{'f_cutoff': f_cutoff})
    plt.subplot(2, 1, 1)
    plt.imshow(1.0*mask.apply_fourier_mask(image))

    mask = FourierMask(np.shape(image), **{'mask_type': 'custom',
                                           'custom_filter': lambda X, Y, r: np.sqrt(np.square(X/r[0]) + np.square(Y/r[1])) < 1})
    plt.subplot(2,1,2)
    plt.imshow(1.0*mask.apply_fourier_mask(image))
    plt.show()

    plt.show()


if __name__ == '__main__':
    demo_fft_filter()
    #demo_threshold()
    #demo_error_diffusion()
    #test_error_diffusion()
    #test_bayer_filter()
    #test_bayer_filter_randomized()

    #plot_bayer()

