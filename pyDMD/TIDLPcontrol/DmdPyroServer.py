__author__ = 'anton'

import platform

import Pyro4

from UsbControl import LightCrafter6500Device as lc
import UsbControl.PatternCreation as pc
import PotetialGeneration.ImagePlanePotentialGenerator as ippg
import threading
import time
import numpy as np
import scipy.misc
import uuid
import re
from PotetialGeneration.SavableObject import ThisComputer
import os.path
import pickle as pkl
from screeninfo import get_monitors

Pyro4.config.SERIALIZER = 'pickle'
Pyro4.config.SERIALIZERS_ACCEPTED = ['pickle', 'serpent']

dmd_ip_dictionary = {'dmd0': '192.168.0.102',
                     'dmd1': '172.23.6.81'}

dmd_resolution_dict = {'lc6500': (1080,1920)}


class VideoImage(object):

    def __init__(self, img, **kwargs):
        self.img_uuid = uuid.uuid4()
        self.is_saved = False
        self.temp_dir = kwargs.pop('temp_dir',  os.path.join('VideoDisplayer', 'resources'))
        self.temp_filename = kwargs.pop('temp_filename','temp_file')
        self.temp_format = kwargs.pop('temp_format', 'png')
        assert self.temp_format in ['png', 'bmp']
        self.dmd_model = kwargs.pop('dmd_model', 'lc6500')
        self.img = self.verify_img(img)
        self.filename = '%s_%s.%s'%(self.temp_filename, self.img_uuid, self.temp_format)
        self.img_path = os.path.join(self.temp_dir, self.filename)
        self.save_img()
        time.sleep(.5)
        self.is_saved = True


    def save_img(self):
        scipy.misc.toimage(self.img,cmin=0,cmax=1).save(self.img_path)
        print "saving image %s" % self.img_path

    def verify_img(self, img):
        conditions = [np.max(img) < 1+1e-6,
                      np.min(img) > 0-1e-6,
                      np.shape(img) == dmd_resolution_dict[self.dmd_model]]
        if all(conditions):
            return img
        else:
            error_info = [np.max(img), np.min(img), np.shape(img)]
            raise Exception("Invalid Image: conditions passed - %s" % zip(conditions,error_info))


class DmdServer(object):
    #TODO: HANDLE DMD BEING TURNED OFF better.

    def __init__(self, **kwargs):
        self.dmd_settings = {}
        self.dmd_model = kwargs.pop('dmd_model', 'lc6500')
        self.dmd_image_directory = kwargs.pop('dmd_image_directory', os.path.join("/media/lithium/DMD"))
        self.comp =  ThisComputer()
        self.ip_addr = dmd_ip_dictionary[self.comp.hostname]

        # Pattern on the fly operation
        self.potential_settings = {}
        self.dmd_name = platform.uname()[1]

        # Video mode operation
        self.monitor_resolutions = self.comp.screen_resolution()

        # USB initialization
        self.lc_dmd = lc.LC6500Device()

        #TODO: FINAL THING - uncomment next line
        #assert self.monitor_resolutions == [(1920,1080)] # allow only the dlp to be connected to the device
        self.video_image = VideoImage(np.zeros(dmd_resolution_dict[self.dmd_model]))
        self.mode = None

    @Pyro4.expose
    def dmd_idle_mode(self, mode):
        """
        puts the dmd into random-flipping mode or takes it out of random flipping mode
        :param mode: enable or disable idle mode
        :return: 0
        """
        assert mode in ['enable', 'disable']
        self.lc_dmd.idle_mode(mode)
        return 0

    # video mode operation
    def _set_dmd_to_video_mode(self, video_source = 'hdmi'):
        """
        Sets dmd video mode
        :param video_source: 'hdmi' or 'displayport'
        :return: 0
        """
        #self.lc_dmd = lc.LC6500Device()
        self.lc_dmd.set_video_mode(video_source=video_source)
        self.mode = 'normal_video'
        return 0


    @Pyro4.expose
    def clean_temp_directory(self, exceptions = []):
        """
        removes all temporary files in thre resources directory, unless they are found in "exceptions"
        :param exceptions:
        :return: 0
        """
        temp_directory = os.path.join('VideoDisplayer', 'resources')
        files = os.listdir(temp_directory)
        regex = re.compile('temp_file_.{8}-.{4}-.{4}-.{4}-.{12}.png')
        counter = 0
        for file in files:
            if regex.match(file) and not file == self.video_image.filename:
                print "removing {}".format(file)
                os.remove(os.path.join(temp_directory, file))
                counter += 1
        print "Cleaned temp directory, removed {} files".format(counter)
        return 0

    @Pyro4.expose
    def upload_img_video_definition(self, potential_settings_list, **kwargs):
        """
        Uploads the definition of the image (computed on dmd server)
        :param potential_settings_list: list of length 1, to be computed on the server
        :param kwargs:
        :return: 0
        """
        if not self.mode == 'normal_video':
            self._set_dmd_to_video_mode()

        if not len(potential_settings_list) == 1:
            raise Exception("The video mode cannot play movies (ironic), since the device is not synced to the experiment")

        potential = ippg.ProjectedPotential(**potential_settings_list[0])

        self.video_image = VideoImage(potential.binarized_potential, **kwargs)
        print "uploaded image with UUID %s" % self.video_image.img_uuid
        return 0

    @Pyro4.expose
    def upload_defined_video_image(self, bin_img, **kwargs):
        """
        Uploads raw image - whatever it is, goes.
        :param bin_img: correctly sized matrix to be put on the atoms
        :return: 0 if everything worked fine, while uploading the img to the server
        """
        if not self.mode == 'normal_video':
            self._set_dmd_to_video_mode()

        self.video_image = VideoImage(bin_img, **kwargs)
        print "uploaded image with UUID %s" % self.video_image.img_uuid
        return 0

    @Pyro4.expose
    def upload_video_image_name(self, name):
        """
        upload the image that lives in the default dmd directory and has name name
        :param name: name of file to look for under self.dmd_image_directory
        :param kwargs:
        :return: 0
        """

        if not self.mode == 'normal_video':
            self._set_dmd_to_video_mode()

        print "Uploading by name"
        if name in os.listdir(self.dmd_image_directory):
            print "found file {}".format(name)
            self.video_image = VideoImage(pkl.load(open(os.path.join(self.dmd_image_directory, name), 'rb'))['pattern'])
        else:
            print "file {} not found".format(name)
        return 0

    @Pyro4.expose
    def check_for_updates(self,uuid):
        """
        Method for the video source to check if the dmd has new source.
        :param uuid: UUID currently stored by video service
        :return: False if no new image, true if new image available
        """
        return self.video_image.is_saved and not uuid == self.video_image.img_uuid

    @Pyro4.expose
    def download_img(self):
        self.clean_temp_directory()
        return self.video_image.filename, self.video_image.img_uuid

    # ===============================================================
    # pattern-on-the-fly-mode

    @Pyro4.expose
    def send_defined_pattern_to_dmd(self, dmd_patterns):
        """
        :param dmd_patterns: pre-computed dmd_patterns
        :return: 0 if everything worked
        """
        dmd_patterns = {'patterns': dmd_patterns}

        self.lc_dmd = lc.LC6500Device()
        self.lc_dmd.upload_image_sequence(dmd_patterns)
        self.mode = 'pattern_on_the_fly'
        return 0

    @Pyro4.expose
    def send_pattern_to_dmd(self, dmd_settings_list, potential_settings_list):
        '''
        :param dmd_settings: list of dict of settings for dmd
        :param potential_settings_list: list of dict of settings for various patterns
        :return:
        '''

        self.dmd_settings_list = dmd_settings_list
        self.potential_settings_list = potential_settings_list

        dmd_patterns = []
        for idx, pot_set in enumerate(potential_settings_list):
            dmd_pattern = pc.DMDPattern(**dmd_settings_list[idx])
            potential = ippg.ProjectedPotential(**pot_set)
            #potential.evaluate_potential()
            #binary_matrix = potential.binarize()
            dmd_pattern.pattern = potential.binarized_potential
            dmd_patterns.append(dmd_pattern)

        dmd_patterns = {'patterns': dmd_patterns}

        self.lc_dmd = lc.LC6500Device()
        self.lc_dmd.upload_image_sequence(dmd_patterns)
        self.mode = 'pattern_on_the_fly'

    def run(self):
        def listen():
            while True:
                time.sleep(10)
                print 'listening'
        thread = threading.Thread(target=listen)
        thread.setDaemon(True)
        thread.start()

    #Debugging tools
    @Pyro4.expose
    def test_connection(self):
        return True

    @Pyro4.expose
    def ping(self, test_int = 0):
        print "This server has just been pinged"
        return "This is dmd %s, pinged with test_int %s"% (self.dmd_name, test_int)


def start_dmd_pyro_daemon():
    dmd = DmdServer()
    print "Starting Pyro Server %s"%dmd.dmd_name

    daemon = Pyro4.Daemon(dmd_ip_dictionary[dmd.dmd_name])

    dmd_uri = daemon.register(dmd)
    ns = Pyro4.locateNS()
    ns.register("LiLab.dmdserver.%s"%dmd.dmd_name, dmd_uri)

    daemon.requestLoop()


if __name__ == "__main__":
    start_dmd_pyro_daemon()
