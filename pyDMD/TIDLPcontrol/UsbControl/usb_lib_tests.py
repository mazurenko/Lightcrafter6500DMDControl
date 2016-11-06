import usb.core
import usb.util


class dmd():
    def __init__(self):
        self.dev=usb.core.find(idVendor=0x0451 ,idProduct=0xc900 )

        if self.dev.is_kernel_driver_active(0):
            reattach = True
            self.dev.detach_kernel_driver(0)

        self.dev.set_configuration()

    def command(self,mode,reply,sequencebyte,com1,com2,data=None):
        buffer = [0x00]*64

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
        buffer[0]=int(flagstring,2)
        buffer[1]=sequencebyte
        len1=2+len(data)
        if len1>254:
            len2=(2+len(data))/255
            len1=len1-len2
        else:
            len2=0
        buffer[2]=len1
        buffer[3]=len2
        buffer[4]=com2
        buffer[5]=com1
        for i in range(len(data)):
            buffer[6+i]=data[i]


        self.dev.write(1, buffer)

    def readreply(self):
        a=self.dev.read(0x81,64)
        for i in a:
            print hex(i)

    def idle_on(self):
        self.command('w',False,0x00,0x02,0x01,[int('00000001',2)])

    def idle_off(self):
        self.command('w',False,0x00,0x02,0x01,[int('00000000',2)])

    def standby(self):
        self.command('w',False,0x00,0x02,0x00,[int('00000001',2)])

    def wakeup(self):
        self.command('w',False,0x00,0x02,0x00,[int('00000000',2)])

    def testread(self):
        self.command('r',True,0xff,0x11,0x00,[])
        self.readreply()

    def testwrite(self):
        self.command('w',True,0x22,0x11,0x00,[0xff,0x01,0xff,0x01,0xff,0x01])


dlp6500=dmd()


dlp6500.wakeup()

dlp6500.testwrite()

dlp6500.testread()

dlp6500.standby()