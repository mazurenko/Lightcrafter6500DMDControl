import os
import pickle as pkl
import uuid
from datetime import datetime
import re
try:
    from screeninfo import get_monitors
except:
    print "Warning: screeninfo not installed, computer will be unable to recognize its screen resolution"
    print "screeninfo package can be instlled using pip"

__author__ = 'anton'


class FileHandler(object):
    @staticmethod
    def ensure_dir(f):
        """
        :param f: filepath to desired directory,
        if the directory does not exist, this directory is made
        :return: 0
        """
        d = os.path.dirname(f)
        if not os.path.exists(d):
            os.makedirs(d)
        return 0

    @staticmethod
    def list_directory_files(path):
        os.listdir(path)

    @staticmethod
    def find_file(path, name_string):
        """
        :param path: directory where to search
        :param name_string: string to recognize in filenames
        :return: list of filenames in this dir containing the string
        """
        files = os.listdir(path)
        accepted_files = []
        for file_name in files:
            if name_string in file_name:
                accepted_files.append(file_name)
        return accepted_files

    @staticmethod
    def find_file_regex(path, regex):
        """
        :param path: directory where to search
        :param regex: regular expression
        :return: a list of all filenames matching the regex
        """
        filenames = os.listdir(path)
        matcher = re.compile(regex)
        is_match = lambda x: matcher.match(x) is not None
        return [x for i, x in enumerate(filenames) if is_match(x)]


class ThisComputer(object):
    os_dict = {'nt': 'windows', 'posix': 'linux'}
    get_system_name_fun_dict = {'windows': lambda:  os.environ['COMPUTERNAME'].lower(),
                                'linux': lambda: os.uname()[1] .lower()}

    """SETTINGS"""
    #TODO: populate this
    dmd_data_dir_dict = {'anton_tp2': 'V:\\dmd_patterns'}
    #TODO: Fix the next line
    default_dmd_data_dir_os_dict = {'windows': 'V:\\dmd_patterns',
                                    'linux': 'NOT IMPLEMENTED!!!'}

    def __init__(self):
        if os.name not in ThisComputer.os_dict.keys():
            raise Exception('Operating system not recognized')
        self.os_type = ThisComputer.os_dict[os.name]

        self.hostname = ThisComputer.get_system_name_fun_dict[self.os_type]()

        self.dmd_data_dir = ThisComputer.dmd_data_dir_dict.get(self.hostname,
                                                               ThisComputer.default_dmd_data_dir_os_dict[self.os_type])

    @staticmethod
    def screen_resolution():
        monitors = get_monitors()
        return map(lambda monitor : (monitor.width, monitor.height), monitors)

    def __str__(self):
        return "Computer: os = %s, hostname = %s" % (self.os_type, self.hostname)


class SavableThing(object):

    def __init__(self, name, **kwargs):
        self.comp = ThisComputer()
        self.name = name
        self.save_directory = kwargs.get('save_directory', self.comp.dmd_data_dir)
        _dat = datetime.now()
        if kwargs.pop('skip_uuid', False):
            self.identifier = "{0:0>4}{1:0>2}{2:0>2}_{3:0>2}{4:0>2}".format(_dat.year, _dat.month, _dat.day,
                                                                            _dat.hour, _dat.minute)
        else:
            self.identifier = "{0:0>4}{1:0>2}{2:0>2}_{3:0>2}{4:0>2}_{5}".format(_dat.year, _dat.month, _dat.day,
                                                                                _dat.hour, _dat.minute, uuid.uuid4())

        self.metadata = kwargs.get('metadata', {})

    @classmethod
    def load_from_pkl(cls, filepath):
        saved_dict = pkl.load(open(filepath, 'rb'))
        tmp = SavableThing('tmp')
        tmp.__dict__.update(saved_dict)
        tmp.comp = ThisComputer()
        return tmp

    def save_pkl(self, **kwargs):
        protocol = kwargs.get('protocol', 2)
        assert protocol in [0, 1, 2]
        pkl.dump(self.__dict__, open("%s\\%s_%s.pkl" % (self.save_directory, self.name, self.identifier), 'wb'),
                 protocol=protocol)


        # for lazy people who don't want to edit scans and rerun
        pkl.dump(self.__dict__, open("%s\\%s.pkl" % (self.save_directory, self.name), 'wb'),
                 protocol=protocol)

    def __str__(self):
        return "%s:: %s" % (self.name, self.__dict__)

if __name__ == "__main__":
    pass
    #test_save = SavableObject('test', metadata={"author":"ant"})
    #test_save.save_pkl()

    #test_load = SavableObject.load_from_pkl('V:\\dmd_patterns\\test.pkl')
    #print test_load

