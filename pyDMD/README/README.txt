To make the USB permissions play nice, please add 50-dmd.rules to /etc/udev/rules.d


Software to install
git
Anaconda
IDE of your choice
if you plan to talk to the fileserver:
apt-get install cifs-utils

Note: depending on ubuntu version, you may need to run: 
sudo apt-get update
sudo apt-get upgrade 

Python packages to install
screeninfo
Pyro4
tqdm
pyusb


If you plan to use video mode, you will need to install the python package pySDL2
pip install pySDL2
apt-get install libsdl2-dev
apt-get install libsdl2-image-dbg

Configuration
Set "turn screen off" to "never".

Add the fileserver permanently to the path:
sudo mkdir /media/fileserver
sudo mount -t cifs //fileserver1.greinerlab.com/FileServer /media/fileserver -o username=mazurenko,domain=greinerlab.com,iocharset=utf8,file_mode=0777,dirmode=0777,cache=none

sudo mkdir /media/lithium
sudo mount -t cifs //fileserver1.greinerlab.com/Lithium /media/fileserver -o username=mazurenko,domain=greinerlab.com,iocharset=utf8,file_mode=0777,dirmode=0777,cache=none

TODO:
-make mouse disappear in video mode.
