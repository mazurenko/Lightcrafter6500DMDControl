__author__ = 'anton'
import Pyro4
import PotetialGeneration.ImagePlanePotentialGenerator as ippg
import PatternRepository.SavedPattern
import UsbControl.PatternCreation as pc

Pyro4.config.SERIALIZER = 'pickle'

class DmdClient(object):

    def __init__(self, dmd_index, potential_settings_list, dmd_settings_list=[]):
        self.dmd_settings_list = dmd_settings_list
        self.potential_settings_list = potential_settings_list
        self.name = "PYRONAME:LiLab.dmdserver.dmd%s"%dmd_index
        self.dmd_server = Pyro4.Proxy(self.name)

    def ping_dmd_channel(self):
        print "trying to ping %s"%self.name
        print self.dmd_server.ping()

    def upload_to_dmd(self):
        print "trying to upload to %s"%self.name
        self.dmd_server.send_pattern_to_dmd(self.dmd_settings_list, self.potential_settings_list)
        print "Sent!"

    def upload_preloaded_to_dmd(self):
        print "trying to upload to %s"%self.name
        self.dmd_server.send_pattern_to_dmd(self.dmd_settings_list, self.potential_settings_list)
        print "Sent!"


    def upload_to_dmd_video_definition(self):
        """
        :return:
        """
        self.dmd_server.upload_img_video_definition(self.potential_settings_list)


def upload_video_pattern(dmd_idx):
    sites = [
        ((0, 0), ippg.GaussFunction(radii=[100, 500], x_angle=1))
    ]

    pattern_settings = [
                {'zero_pixel': (762, 429),
                'sites': sites,
                'binarization_algorithm': 'randomize'}
                ]
    dmd_client = DmdClient(dmd_idx, pattern_settings)
    #dmd_client.ping_dmd_channel(1)
    dmd_client.upload_to_dmd_video_definition()


def upload_manual_pattern_on_the_fly(dmd_idx):
    dmd_settings = [
                    {
                    'compression':'rle',
                    'exposure_time': 5000000  # in us
                    }
                    ]

    sites_dict = {
        ((0,0), ippg.BoxFunction(radii=[100, 100], x_angle=1))
        }

    pattern_settings = [
                {'zero_pixel': (762, 429),
                'sites': sites_dict,
                'binarization_algorithm':'error_diffusion'}
                ]

    dmd_client = DmdClient(dmd_idx, pattern_settings, dmd_settings_list=dmd_settings)
    dmd_client.upload_to_dmd()


def upload_saved_pattern_video(dmd_idx, name):

    dmd_settings = [
                    {
                    'compression':'rle',
                    'exposure_time': 5000000  # in us
                    }
                    ]

    dmd_name = "PYRONAME:LiLab.dmdserver.dmd%s" % dmd_idx
    dmd_server = Pyro4.Proxy(dmd_name)

    print "trying to ping %s" % dmd_name
    print dmd_server.ping()

    print "trying to upload to %s" % dmd_name
    pattern = PatternRepository.SavedPattern.load_pattern(name)
    dmd_server.upload_defined_video_image(pattern)


def upload_saved_pattern_on_the_fly(dmd_idx, name):

    dmd_settings_list = [
                   {
                    'compression':'rle',
                    'exposure_time': 5000000  # in us
                    }
                    ]

    potential_settings_list = [PatternRepository.SavedPattern.load_pattern(name)]

    dmd_patterns = []
    for idx, pot_set in enumerate(potential_settings_list):
        print dmd_settings_list[idx]
        dmd_pattern = pc.DMDPattern(**dmd_settings_list[idx])
        dmd_pattern.pattern = potential_settings_list[idx]
        dmd_patterns.append(dmd_pattern)

    dmd_name = "PYRONAME:LiLab.dmdserver.dmd%s" % dmd_idx
    dmd_server = Pyro4.Proxy(dmd_name)

    print "trying to ping %s" % dmd_name
    print dmd_server.ping()

    print "trying to upload to %s" % dmd_name

    dmd_server.send_defined_pattern_to_dmd(dmd_patterns)


def ping(idx):
    dmd_client = DmdClient(idx, [])
    dmd_client.ping_dmd_channel()


if __name__ == "__main__":
    upload_saved_pattern_on_the_fly(1, 'redist_20161017_1109.pkl')

