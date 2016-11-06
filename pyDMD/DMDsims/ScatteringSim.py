'''
Created on Jan 30, 2016

@author: anton
'''

import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
from numpy import meshgrid
from matplotlib.backends.backend_pdf import PdfPages
import brewer2mpl as cb
from math import isnan

def square_boxcar(X,Y,center_x, center_y, width_x, width_y):
    return np.multiply(np.abs(X-center_x) < width_x/2.0, np.abs(Y-center_y) < width_y/2.0)

def radius(X,Y):
    return np.sqrt(np.add(np.square(X),np.square(Y)))

def azimuthalAverage(image, center=None, stddev=False, returnradii=False, return_nr=False, 
        binsize=0.5, weights=None, steps=False, interpnan=False, left=None, right=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fractional pixels).
    stddev - if specified, return the azimuthal standard deviation instead of the average
    returnradii - if specified, return (radii_array,radial_profile)
    return_nr   - if specified, return number of pixels per radius *and* radius
    binsize - size of the averaging bin.  Can lead to strange results if
        non-binsize factors are used to specify the center and the binsize is
        too large
    weights - can do a weighted average instead of a simple average if this keyword parameter
        is set.  weights.shape must = image.shape.  weighted stddev is undefined, so don't
        set weights and stddev.
    steps - if specified, will return a double-length bin array and radial
        profile so you can plot a step-form radial profile (which more accurately
        represents what's going on)
    interpnan - Interpolate over NAN values, i.e. bins where there is no data?
        left,right - passed to interpnan; they set the extrapolated values

    If a bin contains NO DATA, it will have a NAN value because of the
    divide-by-sum-of-weights component.  I think this is a useful way to denote
    lack of data, but users let me know if an alternative is prefered...
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if center is None:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    if weights is None:
        weights = np.ones(image.shape)
    elif stddev:
        raise ValueError("Weighted standard deviation is not defined.")

    # the 'bins' as initially defined are lower/upper bounds for each bin
    # so that values will be in [lower,upper)  
    nbins = int((np.round(r.max() / binsize)+1))
    maxbin = nbins * binsize
    bins = np.linspace(0,maxbin,nbins+1)
    # but we're probably more interested in the bin centers than their left or right sides...
    bin_centers = (bins[1:]+bins[:-1])/2.0

    # Find out which radial bin each point in the map belongs to
    whichbin = np.digitize(r.flat,bins)

    # how many per bin (i.e., histogram)?
    # there are never any in bin 0, because the lowest index returned by digitize is 1
    nr = np.bincount(whichbin)[1:]

    # recall that bins are from 1 to nbins (which is expressed in array terms by arange(nbins)+1 or xrange(1,nbins+1) )
    # radial_prof.shape = bin_centers.shape
    if stddev:
        radial_prof = np.array([image.flat[whichbin==b].std() for b in xrange(1,nbins+1)])
    else:
        radial_prof = np.array([np.multiply(image,weights).flat[whichbin==b].sum() / weights.flat[whichbin==b].sum() for b in xrange(1,nbins+1)])
        #radial_prof = np.array([np.multiply(image,weights).flat[whichbin==b].sum() / weights.flat[whichbin==b].sum() for b in xrange(1,10)])

    #import pdb; pdb.set_trace()

    if interpnan:
        radial_prof = np.interp(bin_centers,bin_centers[radial_prof==radial_prof],radial_prof[radial_prof==radial_prof],left=left,right=right)

    if steps:
        xarr = np.array(zip(bins[:-1],bins[1:])).ravel() 
        yarr = np.array(zip(radial_prof,radial_prof)).ravel() 
        return xarr,yarr
    elif returnradii: 
        return bin_centers,radial_prof
    elif return_nr:
        return nr,bin_centers,radial_prof
    else:
        return radial_prof



class FourierSim(object):
    
    def __init__(self, 
                 xrng = [-200, 200], 
                 yrng = [-200,200], 
                 N = [400,400],
                 NA = .005, 
                 wavelength = .671,
                 magnification = 160,
                 dmd = None):
        
        self.out_dir = "plots"

        self.xrng = xrng
        self.yrng = yrng
        self.NA = NA
        self.wavelength = wavelength
        self.magnification = 160
        self.N = N
        self.dmd = dmd
         
         
        self.maxf = NA/wavelength
        self.sample_spacing = [(self.xrng[1]-self.xrng[0])/(self.N[0]-1),(self.yrng[1]-self.yrng[0])/(self.N[1]-1)]

        x = np.linspace(self.xrng[0], self.xrng[1], self.N[0], endpoint=True)
        y = np.linspace(self.yrng[0], self.yrng[1], self.N[1], endpoint=True)

        fx = fftpack.fftshift(fftpack.fftfreq(self.N[0], self.sample_spacing[0]))
        fy = fftpack.fftshift(fftpack.fftfreq(self.N[1], self.sample_spacing[1]))

        self.X,self.Y = np.meshgrid(x,y)
        self.R = np.sqrt(np.square(self.X) + np.square(self.Y))
        self.FX,self.FY = np.meshgrid(fx,fy)
        self.Fsum = np.sqrt(np.square(self.FX)+np.square(self.FY))

    def compute_initial_amplitude_distribution(self):
        self.amplitude_distribution = np.zeros_like(self.X)
        for i in range(np.size(dmd.centers_x)):
            self.amplitude_distribution = self.amplitude_distribution + square_boxcar(self.X,self.Y,dmd.centers_x[i],dmd.centers_y[i],dmd.pix_spacing,dmd.pix_spacing)
        
        #plt.imshow(self.amplitude_distribution) 
        #plt.show()
    
    def fourier_filter(self):
        self.fourier_filter_mat = self.Fsum < self.maxf
        self.fourier_plane = fftpack.fftshift(fftpack.fft2(self.amplitude_distribution))
        self.fourier_plane_filtered = np.multiply(self.fourier_filter_mat, self.fourier_plane)
    
        #backtransform
        self.image = np.abs(fftpack.ifft2(self.fourier_plane_filtered))
        #print self.image

    def compute_radial_average(self):
        rads, avg = azimuthalAverage(self.image,returnradii = True)
        
        assert self.sample_spacing[0] == self.sample_spacing[1]
        
        center_amp = next(x for x in avg if not isnan(x))

        return self.sample_spacing[0]*rads/self.magnification, avg*self.magnification, center_amp*self.magnification # Assume that spacing is the same in X, Y
    
    def show_plots(self):
        plt.subplot(221)
        plt.imshow(self.amplitude_distribution)
        plt.title("initial distribution")
        plt.colorbar()
        plt.subplot(222)
        plt.imshow(np.abs(self.fourier_plane))
        plt.title("Fourier Plane")
        plt.subplot(224)
        plt.imshow(self.fourier_filter_mat)
        plt.title("Fourier filter")
        plt.subplot(223)
        plt.imshow(self.image)
        plt.colorbar()
        plt.title("Image")
        
        
        name = self.dmd.turn_on_radius

        
        pp=PdfPages('%s\\%s.pdf'%(self.out_dir,name))
        plt.savefig(pp,format='pdf')
        pp.close()
        plt.savefig('%s\\%s.png'%(self.out_dir,name),bbox_inches='tight')
 
        
        plt.show()
     
 
        

class DMDchip(object):
    
    def __init__(self,turn_on_radius_um, chip_side_length_pixels = 40):
        #all spatial dims in micron
        self.pix_spacing = 7.6
        self.turn_on_radius = turn_on_radius_um
        self.chip_side_length_pixels = chip_side_length_pixels 
        assert chip_side_length_pixels%2 == 0

    def compute_on_pixels(self):
        x = np.arange(-1*self.chip_side_length_pixels/2,self.chip_side_length_pixels/2,1)
        y = x
        X, Y = np.meshgrid(x,y)
        R_pix = radius(X, Y)
        R_um = self.pix_spacing*R_pix
        #R_filtered = np.multiply(R_um,R_um < self.turn_on_radius)
        _on_pixels_x, _on_pixels_y = np.nonzero(R_um < self.turn_on_radius)
        self.on_pixels_x, self.on_pixels_y = x[_on_pixels_x], y[_on_pixels_y]
        #self.on_pixels_x
        #self.on_pixels_y
        self.centers_x,self.centers_y = self.pix_spacing*self.on_pixels_x, self.pix_spacing*self.on_pixels_y
    
   


def manual():
    #settings
    xrng = [-200.0,200.0]
    yrng = [-200.0,200.0]
    NA = .005
    wavelength = .671
    
    maxf = NA/wavelength
    
    #spatial units are micron 
    N = [401,401]

    
    #SCRIPT
    sample_spacing = [(xrng[1]-xrng[0])/(N[0]-1),(yrng[1]-yrng[0])/(N[1]-1)]

    x = np.linspace(xrng[0], xrng[1], N[0], endpoint=True)
    y = np.linspace(yrng[0], yrng[1], N[1], endpoint=True)

    fx = fftpack.fftshift(fftpack.fftfreq(N[0], sample_spacing[0]))
    fy = fftpack.fftshift(fftpack.fftfreq(N[1], sample_spacing[1]))

    X,Y = np.meshgrid(x,y)
    FX,FY = np.meshgrid(fx,fy)
    
    Fsum = np.sqrt(np.square(FX)+np.square(FY))
    
    #inital distribution
    w = 100
    gauss = np.exp(-(np.square(X)+np.square(Y))/ np.square(w))
    initial = gauss
    
    #fourier filter
    filter = Fsum < maxf
    F1 = fftpack.fftshift(fftpack.fft2(gauss))
    filtered = np.multiply(filter, F1)
    
    #backtransform
    image = np.abs(fftpack.ifft2(filtered))
    print image
    
    
    plt.subplot(221)
    plt.imshow(gauss)
    plt.colorbar()
    plt.subplot(222)
    plt.imshow(np.abs(F1)*filter)
    plt.subplot(223)
    plt.imshow(image)
    plt.colorbar()
    plt.subplot(224)
    plt.imshow(square_boxcar(X,Y,100,100,50,200))
    plt.show()
    
if __name__ == '__main__':
    #manual()
    
    #radii = np.array([5,10,20,30,40,60,80,100,150,200,250])
    radii = np.array([5,20,40,60,80,100,150,200,250])
    #radii = np.array([5,60,200])

    #on_pix_um_radius = 50
    
    center_amp_vec = np.zeros_like(radii)
    
    
    #plot_settings
    colors = cb.get_map('YlOrRd','Sequential', 9).mpl_colors
    
    plt.figure(figsize=(10,10))
    plt.subplot(2,1,1)
    for idx,r in enumerate(radii):
        dmd = DMDchip(r,chip_side_length_pixels= 100)
    
        sim = FourierSim(xrng = [-500.0, 500.0], yrng = [-500.0,500.0], N = [500,500], dmd = dmd)
        sim.dmd.compute_on_pixels()
        sim.compute_initial_amplitude_distribution()
        sim.fourier_filter()
        rads, avg, center_amp = sim.compute_radial_average()
        
        print center_amp
        center_amp_vec[idx] = center_amp


        plt.plot(rads,avg,color = colors[idx],linewidth = 4,label = "Active dmd rad = %s um"%r)
    
    plt.title("Azimuthal Averages",fontsize = 20)
    plt.xlabel("Atom Plane Radius (micron)", fontsize = 16)
    plt.ylabel("Amplitude (arb)", fontsize = 16)
    plt.legend(loc = 1)


    plt.subplot(2,1,2)
    plt.plot(radii, center_amp_vec,'o-',linewidth = 3, markersize = 5, color = cb.get_map("PuBu","Sequential",3).mpl_colors[-1])

    plt.title("Central Amplitude",fontsize = 20)
    plt.xlabel("Active DMD Radius (micron)", fontsize = 16)
    plt.ylabel("Amplitude (arb)", fontsize = 16)
    
    
    name = "azimuthal"
    pp=PdfPages('%s\\%s.pdf'%(sim.out_dir,name))
    plt.savefig(pp,format='pdf')
    pp.close()
    plt.savefig('%s\\%s.png'%(sim.out_dir,name),bbox_inches='tight')
 
        

    plt.show()