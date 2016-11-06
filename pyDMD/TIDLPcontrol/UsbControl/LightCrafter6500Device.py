'''
Created on Feb 27, 2016

@author: anton
'''
import os
import numpy as np
import HexHelper as hh
from UsbDevice import USBhidDevice
import time

if os.name == 'nt':
    OS_TYPE = 'windows'
    import pywinusb.hid as pyhid
elif os.name == 'posix':
    OS_TYPE = 'linux'

vhex = np.vectorize(hex)

class DmdError(Exception):
    def __init__(self, value):
        self.error_code_dict = {0:"no error, all is well",
                           1:"batch file checksum error - oopf",
                           2:"device failure - that sounds ominous",
                           3:"invalid command number",
                           4:"incompatible controller/dmd",
                           5:"command not allowed in current mode",
                           6:"invalid command parameter",
                           7:"item referred by the parameter is not present",
                           8:"out of resource/RAM",
                           9:"invalid bmp compression type",
                           10:"pattern bit number out of range",
                           11:"pattern bmp not present in flash",
                           12:"pattern dark time out of range",
                           13:"signal delay parameter out of range",
                           14:"pattern exposure time is out of range",
                           15:"pattern number is out of range",
                           16:"invalid pattern definition",
                           17:"pattern image memory address is out of range",
                           255:"Internal error"}
        for i in range(18,255):
            self.error_code_dict[i] = "undefined error - if this is thrown, good luck :<"

        self.value = value
    def __str__(self):
        return repr("%s: %s"%(self.value, self.error_code_dict[self.value]))


class LC6500Device(USBhidDevice):
    
    def __init__(self,**kwargs):
        super(LC6500Device, self).__init__()
        self.is_debug = kwargs.pop('is_debug', False)

    def dmd_read_response(self):
        return self.dev.read(0x81, 64)

    def dmd_command(self, mode, reply, sequencebyte, com1, com2, data=[], is_read=True, readlen=None):
        if self.os == 'windows':
            buffer = [0x00]*65
            i=1
        else:
            buffer = [0x00]*64
            i=0
        flagstring=''
        if mode=='r':
            flagstring+='1'
        else:
            flagstring+='0'
        if reply:
            flagstring+='1'
        else:
            flagstring+='0'
        flagstring+='000000'
        #buffer[0]=0x00
        buffer[i]=int(flagstring,2)
        buffer[i+1]=sequencebyte
        len1=2+len(data)
        if len1>254:
            len2=(2+len(data))/255
            len1=len1-len2
        else:
            len2=0
        buffer[i+2]=len1
        buffer[i+3]=len2
        buffer[i+4]=com2
        buffer[i+5]=com1
        for j in range(len(data)):
            buffer[i+6+j]=data[j]

        if self.os == 'windows':
            super(LC6500Device,self).send_packet(buffer)
        elif self.os == 'linux':
            super(LC6500Device,self).raw_command(buffer)
            if is_read:
                response = self.dmd_read_response()
                if self.is_debug:
                    print response
            else:
                response = None
        else:
            print "unrecognized os, no comm sent"
        return response

    def __slave_command__(self,buff):
        buffer = [0x00] + buff
        super(LC6500Device,self).send_packet(buffer)
        
    '''USEFUL COMMANDS'''

    def check_for_errors(self):
        error_code = self.dmd_command('w',True,0xAB,0x01,0x00,[])[6]

        if error_code != 0:
            #print "Error %s: %s"%(error_code, self.error_code_dict[error_code])
            raise DmdError(error_code)
        return error_code

    def set_pattern_on_the_fly(self):
        """
        Set the DMD into pattern on the fly mode
        """
        self.dmd_command('w',True,0x22,0x1a,0x1b, data=[0x03])
        self.check_for_errors()
        print "Setting pattern-on-the-fly mode"

    def __set_mode(self, mode):
        mode_dict = {'normal_video':0x00, 'prestored_pattern':0x01, 'video_pattern':0x02, 'pattern_on_the_fly':0x03}
        assert mode in mode_dict.keys()
        self.dmd_command('w',True,0x91,0x1a,0x1b,data=[mode_dict[mode]])
        print "Set %s mode" % mode

    def __set_video_source_mode(self, video_source):
        video_source_dict = {'none': 0x00, 'hdmi': 0x01, 'displayport': 0x02}
        assert video_source in video_source_dict.keys()
        self.dmd_command('w',True,0xcb,0x1a,0x01,data = [video_source_dict[video_source]])
        print "set video source to %s" % video_source

    def set_video_pattern_mode(self, video_source='hdmi'):
        self.__set_mode('video_pattern')
        self.__set_video_source_mode(video_source)

    def set_video_mode(self, video_source='hdmi'):
        self.__set_mode('normal_video')
        self.__set_video_source_mode(video_source)

    def start_stop_sequence(self,cmd):
        """
        :param cmd - string either 'start', 'stop' or 'pause'
        """
        if cmd == 'start':
            data = [0x02]
        elif cmd == 'stop':
            data = [0x00]
        elif cmd == 'pause':
            data = [0x01]
        else:
            raise Exception("can only have start, stop, or pause")
        self.dmd_command('w',True,0x22,0x1a,0x24, data=data)
        self.check_for_errors()
        print "%sing sequence"%cmd

    def bmp_load(self, length, index = 0):
        data = hh.int2reversed_hex_array(index, n_bytes=2)
        data+= hh.int2reversed_hex_array(length,n_bytes=4)
        self.dmd_command('w',True,0x11,0x1a,0x2a, data=data)
        self.check_for_errors()
    
    def idle_mode(self, mode):
        mode_dict = {'enable':0x01, 'disable':0x00}
        print mode in mode_dict.keys()

        assert mode in mode_dict.keys()
        if mode == 'enable':
            self.start_stop_sequence('stop')
        self.dmd_command('w', True, 0x42, 0x02, 0x01, [mode_dict[mode]])
        print "idle mode %sd" % mode
        if mode == 'disable':
            self.start_stop_sequence('start')
        print self.check_for_errors()

    def pattern_display_lut_definition(self, num_patterns, num_repeat = 0):
        assert num_patterns < 256
        print "configuring lookup table"
        num_patterns_bytes = hh.int2reversed_hex_array(num_patterns, n_bytes=2)
        num_repeats_bytes = hh.int2reversed_hex_array(num_repeat, n_bytes=4)
        data = num_patterns_bytes + num_repeats_bytes
        #self.dmd_command('w', True, 0x78, 0x1a, 0x31, data)
        self.dmd_command('w', False, 0x0d, 0x1a, 0x31, data, is_read=False) #NOTE: THE IS READ = FALSE IS ESSENTIAL
        self.check_for_errors()

    def trigger_1_config(self, trigger_delay, is_trigger_rising_edge = True):
        assert trigger_delay > 104
        print "configuring trigger"
        trigger_delay_bytes = hh.int2reversed_hex_array(trigger_delay, n_bytes = 2)
        rising_edge_bit = [0x00] if is_trigger_rising_edge else [0x01]
        self.dmd_command('w', True, 0x43, 0x1a, 0x35, trigger_delay_bytes + rising_edge_bit)
        self.check_for_errors()

    def pattern_def_cmd(self, pattern_index, exposure_time = 105, dark_time = 0, is_wait_for_trig = True):
        assert pattern_index < 256 and pattern_index >= 0 

        pattern_index_bytes = hh.int2reversed_hex_array(pattern_index, 2)
        exposure_bytes = hh.int2reversed_hex_array(exposure_time, 3)
        trigger_settings_bytes = [0xf1] if is_wait_for_trig else [0x01]
        dark_time_bytes = hh.int2reversed_hex_array(dark_time, 3)
        trig2_output_bytes = [0x00]
        
        data = pattern_index_bytes + exposure_bytes + trigger_settings_bytes + \
            dark_time_bytes + trig2_output_bytes + pattern_index_bytes
        self.dmd_command('w', True, 0x65, 0x1a, 0x34, data)
        self.check_for_errors()

    def upload_image_sequence(self, dmd_pattern_sequence):
        '''
        :param dmd_pattern_sequence: dict with 'patterns' giving a list of patterns, and 'num_repeats' giving the number of repeats
        '''
        assert dmd_pattern_sequence['patterns'] is not None
        
        self.start_stop_sequence('stop')
        self.set_pattern_on_the_fly()
        
        for idx, dmd_pattern in enumerate(dmd_pattern_sequence['patterns']):
            self.pattern_def_cmd(idx, exposure_time=dmd_pattern.exposure_time, dark_time=dmd_pattern.dark_time )

        for idx, dmd_pattern in reversed(list(enumerate(dmd_pattern_sequence['patterns']))):
            image_bits = dmd_pattern.compress_pattern()
            self.bmp_load(np.size(image_bits), index=idx)
            self.send_image_stream(image_bits)
            print "sent pattern %s"%idx
            
        print "The number of patterns is %s"%len(dmd_pattern_sequence['patterns'])
        
        self.pattern_display_lut_definition(len(dmd_pattern_sequence['patterns']),dmd_pattern_sequence.pop('num_repeats', 0))

        self.trigger_1_config(105)
        self.start_stop_sequence('start')

        print "Finished Upload Process"

    def send_image_stream(self, image_bits):

        '''
        images get transmitted as 1a2b commands, followed by 7 slave commands,
        
        (flag byte)(sequence byte)(length lsb)(length(msb)(command lsb)(command msb)(num bytes packet lsb)(num bytes packet msb [1 bit only!])
        that is: 512-8 = 504 max!
        '''
        
        '''
        max real payload number of bytes in the first packet 56
        max real payload number of bytes in the slave 64
        '''
        max_cmd_payload = 504

        start_time = time.clock()
        im_size = np.size(image_bits)

        num_full_cmd_groups = im_size/max_cmd_payload
        remainder_group_length = im_size % max_cmd_payload

        print "bytes to be transmitted: %s"% np.size(image_bits)
        print "split into %s command groups"% num_full_cmd_groups
        print "with remainder %s bytes"% remainder_group_length

        print "starting image upload"

        # Send full command groups
        for cmd_group_index in range(num_full_cmd_groups):  
            if cmd_group_index % 50 == 0:
                print "sending packet group %s" % cmd_group_index
            self.send_image_command_with_slaves(image_bits[0+cmd_group_index*max_cmd_payload:max_cmd_payload+cmd_group_index*max_cmd_payload],
                                                cmd_group_index)
        # Send remainder
        self.send_image_command_with_slaves(image_bits[num_full_cmd_groups*max_cmd_payload:num_full_cmd_groups*max_cmd_payload+remainder_group_length],
                                            0xab)

        end_time = time.clock()
        time_elapsed = end_time-start_time

        print "the programming took %s seconds" % time_elapsed
        print "the average data rate is %s B/s" % (1.0*im_size/time_elapsed)

        self.check_for_errors()

    def send_image_command_with_slaves(self, image_bits, sequence_byte):
        """
        :param image_bits is a numpy uint array of the current payload
        :param sequence_byte is there for bookkeeping, gets placed as sequence byte of first command
        """
        
        payload_length = np.size(image_bits) 
        assert payload_length <= 504

        command  = [0x00]*8
        command[1] = np.uint8(sequence_byte % 256)
        data_length_with_data_header = hh.int2reversed_hex_array(2+2+payload_length)
        # 2 for the command bytes, 2 for the length bytes in the payload, then payload
        data_length = hh.int2reversed_hex_array(np.size(image_bits))
        
        # compute number of slaves
        if payload_length <= 56:
            print "preparing last packet of image"
            n_slaves = 0
            pad_width = 56 - payload_length
            image_bits_padded = np.pad(image_bits, (0, pad_width), 'constant', constant_values=(0,0))

        else:
            n_slaves = (payload_length - 56) / 64
            if (payload_length - 56) % 64 != 0:
                n_slaves += 1
            pad_width = (64 - (payload_length - 56) % 64)% 64
            image_bits_padded = np.pad(image_bits, (0, pad_width), 'constant', constant_values=(0,0))

        for i in range(2):
            command[2+i] = data_length_with_data_header[i]
        command[4] = 0x2B
        command[5] = 0x1A
        for j in range(2):
            command[6+j] = data_length[j]
        
        if n_slaves == 0:
            command += image_bits_padded.tolist()
            self.raw_command(command)
        else:
            command += image_bits_padded[0:56].tolist()
            self.raw_command(command)
            
            slave_bits = image_bits_padded[56:]
            for k in range(n_slaves):

                if OS_TYPE == 'windows':
                    print "slave rng %s, %s"%(64*k, 64*(k+1))
                    print "length %s"%len(slave_bits[64*k:64*k+64].tolist())
                self.raw_command(slave_bits[64*k:64*k+64].tolist())

if __name__ == '__main__':
    lc_dmd = LC6500Device()
    lc_dmd.start_stop_sequence('stop')
