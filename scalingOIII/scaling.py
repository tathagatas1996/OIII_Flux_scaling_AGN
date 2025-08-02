import os
import warnings
import numpy as np
import scipy.integrate as intg 
from scipy.interpolate import interp1d
import emcee

def gauss_sm(lam,lam0,sd):
    """GAUSSIAN PROFILE
    Parameters:
    ----------
    lam : numpy array
        Wavelengths of the target spectrum.
    lam0 : float
        Central wavelength of the Gaussian.
    sd : float
        Standard deviation of the Gaussian.
    """
    tt = ((lam - lam0)/sd)**2
    ff = np.exp(-tt/2)
    return ff

def estimate_errorbars(x, y):
    """
    Estimate errorbars in case if the provided data does not have errorbars
    """
    dlam = 10.0
    lam_l = min(x)
    lam_h = max(x)
    xsm = np.arange(lam_l-dlam,lam_h+dlam,dlam)
    ysm = []
    for i in range(xsm.size):
        Ns = np.sum( gauss_sm(xsm[i],x,dlam))
        fs = np.sum( gauss_sm(xsm[i],x,dlam) * y )
        ysm.append(fs/Ns) 
    
    ysm = np.array(ysm)
    
    f = interp1d(xsm, ysm)
    y_baseline = f(x)
    e = np.sqrt((y - y_baseline)**2/(y.size-1))*np.ones(y.size)
    return e


def load_spec( ref_spec ):
    """
    LOAD SPECTRA
    Parameters
    ----------
    ref_spec : str
        Path to the reference spectrum file.
    """
    try:
        Spectrum = np.loadtxt( ref_spec)
        columns = Spectrum[0,:].size
        if columns == 3:
            xR = Spectrum[:,0]
            R  = Spectrum[:,1]
            eR = Spectrum[:,2]
            return([xR,R,eR])
        elif columns == 2:

            warnings.warn("Did not find a the third (errorbar) column. " \
            "Calculating one based on the (lamda, flux) values. " \
            "This method might not be accurate for all spectra. " \
            "Users are advised to provide a third column with errorbars.", UserWarning)

            xR = Spectrum[:,0]
            R  = Spectrum[:,1]
            eR = estimate_errorbars(xR,R)
            DataOut = np.column_stack((xR, R, eR))
            np.savetxt(ref_spec+"_with_errorbars_for_scalingOIII.txt", DataOut)
            return([xR, R, eR])
        else:
            return(0,0,0)
    except FileNotFoundError:
        print(f"File not found:{ref_spec}. Make sure the path is correct.")
        return(0,0,0)
    
def require_valid_data(func):
    """ Decorator to ensure that the data attributes are numpy arrays before calling a method.
    Raises RuntimeError"""
    def wrapper(self, *args, **kwargs):
        for attr in (self.xr, self.fr, self.er, self.xt, self.ft, self.et):
            if not isinstance(attr, np.ndarray):
                raise RuntimeError(f"Cannot call {func.__name__}: One or more data attributes are not numpy arrays.")
        return func(self, *args, **kwargs)
    return wrapper
    

class scale:
    def __init__(self, ref_spec, spec_path):
        """
        Load and initialize the reference and target spectra.
        Parameters
        ----------
        ref_spec : str
            Path to the reference spectrum file.
        spec_path : str
            Path to the target spectrum file.
        """
        self.xr, self.fr, self.er = load_spec( ref_spec)
        self.xt, self.ft, self.et = load_spec( spec_path )
        self.window = (5330,5450)
        self.line   = [(5340, 5380)]
    
    @require_valid_data
    def cont(self, x, xcont, ycont):
        """INTERPOLATE THE UNDERLYING CONTINUUM"""
        f0 = interp1d(xcont, ycont, fill_value = "extrapolate")
        return f0(x)
    
    def gauss(self, lam,lam0,sd):
        """GAUSSIAN KERNEL PROFILE"""
        tt = ((lam - lam0)/sd)**2
        ff = np.exp(-tt/2)
        return ff

    @require_valid_data
    def trim_range(self, lam, flux, 
                   window, mask):
        """SEGMENT THE SPECTRUM"""
        flux = np.delete(flux,np.where((lam<window[0]) | (lam>window[1])))
        lam = np.delete(lam,np.where((lam<window[0]) | (lam>window[1])))
        for wm in mask:
            flux = np.delete(flux,np.where((lam>wm[0]) & (lam<wm[1])))
            lam = np.delete(lam,np.where((lam>wm[0]) & (lam<wm[1])))
        return (lam,flux)


    @require_valid_data
    def spectral_segments(self):
        """ CREATE SEGMENTS OF THE REFERENCE SPECTRUM"""
        nomask = [(0,0)]
        self.xc, self.fc = self.trim_range( self.xr, self.fr, self.window, self.line )
        self.xc, self.ec = self.trim_range( self.xr, self.er, self.window, self.line )

        self.xT, self.fT = self.trim_range( self.xr, self.fr, self.window, nomask )
        self.xT, self.eT = self.trim_range( self.xr, self.er, self.window, nomask )
        
        self.fl = self.fT - self.cont( self.xT, self.xc, self.fc) #<<----- Subtract the continuum
        self.R0 = intg.simps(self.fl, self.xT)
        
        """ CREATE SEGMENTS OF THE TARGET SPECTRUM"""
        self.xOIII , self.fOIII  = self.trim_range( self.xt, self.ft, self.window, nomask)
        self.xOIII , self.eOIII  = self.trim_range( self.xt, self.et, self.window, nomask)

        self.xct , self.fct = self.trim_range( self.xOIII, self.fOIII, self.window, self.line)
        self.xct , self.ect = self.trim_range( self.xOIII, self.eOIII, self.window, self.line)

        self.fline = self.fOIII - self.cont( self.xOIII, self.xct, self.fct )


    def fsmooth(self, xO, fO, eO, 
                      sd, shift):
        """ Gaussian smoothing of the observed spectrum """
        li = np.arange( min(xO), max(xO), 1 )
        ff = np.zeros( li.size, dtype=float)
        ee = np.zeros( li.size, dtype=float)
        for i in range(li.size):
            Ni = np.sum( self.gauss( li[i], xO, sd ) )
            ff[i] = (1/Ni) * np.sum( fO * self.gauss( li[i], xO, sd ) )
            ee[i] = (1/Ni) * np.sum( eO * self.gauss( li[i], xO, sd ) )
        li = li + shift
        return([li,ff, ee])


    def calculate_interp_error(self, xref,x,e):
        """ CALCULATE THE ERROR ON THE INTERPOLATED FLUX """
        eref = np.zeros( xref.size, dtype= float)
        for i in range (xref.size):
        
            idx1 = np.where( xref[i] >= x )[0]
            idx2 = np.where( xref[i] <  x )[0]
        
            if idx1.size != 0 and idx2.size != 0 :
                id1 = idx1[0]
                id2 = idx2[idx2.size-1]
                eref[i] = np.sqrt(1/e[id1]**2  + 1/e[id2]**2)**(-1)           
        return eref
    

    def f0(self, x, a, sd, shift, 
           xO, fO, eO):
        """ Smooth, Shift, Scale and Interpolate the spectrum at desired points"""
        a0 = self.fsmooth( xO, fO, eO, sd, shift )
        f01 = interp1d( a0[0], a0[1], fill_value='extrapolate' )
        ff = a * f01(x)
        ee = a * self.calculate_interp_error( x, a0[0], a0[2] )
        return [ff,ee]

    ######################### MCMC FUNCTIONS #########################
    def log_likelihood(self, theta, x, y, yerr):
        """likelihood function"""
        a, sd, shift = theta
        # target being evaluated at the wavelength of the reference spectrum with scale, smoothing, and shift transformation:
        Olambda, eOlambda = self.f0( self.xT, a, sd, shift, x, y, yerr)
        #local variable being used for the reference spectra:
        Rlambda = self.fl
        # Error of reference and error of target added in quadrature:                 
        sigma2  =  eOlambda**2 + self.eT**2
        l =  -0.5 * np.sum( (Olambda - Rlambda)**2/sigma2 ) # Likelihood estimation:
        return l

    def log_prior(self, theta):
        """uniform priors"""
        a, sd, shift = theta
        if 0 < a < 10 and 5e-1 < sd < 10 and -8 < shift < 20:
            return 0.0
        return -np.inf

    def log_probability(self, theta, x, y, yerr):
        """total probalility"""
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta, x, y, yerr)
    
    @require_valid_data
    def run_emcee(self, x, f, e, nchains):
        """
        emcee implementation
        
        Parameters
        ----------
        x : numpy array of the OIII line of the target spectrum
        f : numpy array of the flux of the OIII line of the target spectrum
        e : numpy array of the errorbars of the OIII line of the target spectrum
        nchains : number of mcmc steps
        """
        init = np.array([1,4,2], dtype=float)
        pos = init + 1e-4 * np.random.randn(10, 3) 
        nwalkers, ndim = pos.shape
        sampler = emcee.EnsembleSampler( nwalkers, ndim, self.log_probability, args=(x, f, e) )
        sampler.run_mcmc(pos, nchains, progress=True);
        flat_samples = sampler.get_chain(discard=int(0.1*nchains), thin=15, flat=True)
        return flat_samples
    
    def best_values(self, samples):
        """
        returns the best fit values from the samples
        Parameters
        ----------
        samples : numpy array of the mcmc samples
        """
        a = np.median(samples[:,0])
        sd = np.median(samples[:,1])
        shift = np.median(samples[:,2])
        return a, sd, shift

    @require_valid_data
    def transform_spectrum(self, x, a, sd, s):
        """
        Transform the target spectrum using the best fit parameters.
        Parameters
        ----------
        x : numpy array with the wavelength of the target spectrum
        a : float Scaling factor for the flux.
        sd : float Standard deviation for the Gaussian smoothing.
        s : float Shift in wavelength.
        """
        flam, elam = self.f0(x, a, sd, s, self.xOIII, self.fline, self.eOIII)
        return (x, flam, elam)

