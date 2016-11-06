'''
Created on Feb 27, 2016

@author: anton
'''
import os

if os.name == 'nt':
    OS_TYPE = 'windows'
    import pywinusb.hid as pyhid
elif os.name == 'posix':
    OS_TYPE = 'linux'
    import usb.core
    import usb.util
else:
    print "Unrecognized OS: %s"%os.name

config ={}
config['is_debug'] = True

class USBError(Exception):
    def __init__(self, value, (vid, pid) = (None, None)):
        self.value = value
        self.vid = vid
        self.pid = pid

    def __str__(self):
        return repr("USB ERROR: %s" % self.value) if self.vid is None and self.pid is None else repr("USB ERROR: vid %s, pid %s: %s"%(self.vid, self.pid, self.value))


class USBhidDevice(object):
    
    def __init__(self, **kwargs):
        self.vendor_id = kwargs.pop('vendor_id',0x0451)
        self.product_id = kwargs.pop('product_id', 0xc900)
        self.os = OS_TYPE
        self.hid_timeout = kwargs.pop('hid_timeout', 100)

        if OS_TYPE== 'windows':
            filter = pyhid.HidDeviceFilter(vendor_id = 0x0451, product_id = 0xc900)
            devices = filter.get_devices()

            if devices:
                device = devices[0]
                print "success"
            self.dev = device
            self.dev.open()


        if OS_TYPE=='linux':
            self.dev=usb.core.find(idVendor=self.vendor_id,idProduct=self.product_id)
            if self.dev is None:
                raise USBError('USB device not found!!!', (self.vendor_id, self.product_id))

            if self.dev.is_kernel_driver_active(0):
                reattach = True
                self.dev.detach_kernel_driver(0)
            self.dev.set_configuration()
            cfg = self.dev.get_active_configuration()
            intf = cfg[(0,0)]
            self.dev.reset()

    def send_packet(self,packet):
        assert 65 == len(packet)

        if OS_TYPE == 'linux':
            self.dev.write(1, packet)
            #print self.dev.read(0x81,64)


        elif OS_TYPE == 'windows':
            reports = self.dev.find_output_reports()
            #print reports[0]
            reports[0].send(packet)

   
    def raw_command(self,buff,readlen=None,endpoint=1):
        if endpoint == 1:
            assert 64 == len(buff)

        if OS_TYPE == 'linux':
            self.dev.write(endpoint, buff)
            if readlen is not None:
                return self.dev.read(0x81,readlen)

        elif OS_TYPE == 'windows':
            buff = [0x00] + buff
            reports = self.dev.find_output_reports()
            #print reports[0]
            reports[0].send(buff)

    def release(self):
        usb.util.dispose_resources(self.dev)




if __name__ == '__main__':
    test_dev = USBhidDevice()
    print test_dev
