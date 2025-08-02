import matplotlib.pyplot as plt
import numpy as np
import corner

def plot_chain(samples):
    """
    Plot the chain of the MCMC samples.
    Parameters
    ----------
    samples : numpy array of the MCMC samples
    """
    ndim = samples[0,:].size
    fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    labels = [r"A",r"$\sigma$",r"$\Delta \lambda$[$\AA$]"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    return

def plot_contour(samples):
    """
    Plot the corner plots of the posteriors.
    Parameters
    ----------
    samples : numpy array of the mcmc samples
    """
    labels = [r"A",r"$\sigma$",r"$\Delta \lambda$[$\AA$]"]
    a1 = samples[:,0]
    a2 = samples[:,1]
    a3 = samples[:,2]
    data = np.array([a1,a2,a3]).T
    ndim = samples[0,:].size
    #title = lab,quantiles=[0.05, 0.5, 0.95], label_kwargs={"fontsize": 20}, show_titles=True,title_kwargs={"fontsize": 16},)
    fig = corner.corner(data, 
                        labels = labels, 
                        fontsize = 12, 
                        show_titles = True, 
                        label_kwargs={"fontsize": 20},
                        title_kwargs = {"fontsize": 16},
                        levels=(0.683,0.90,0.99));
    
    axes = np.array(fig.axes).reshape((ndim, ndim))
    for i in range(ndim):
        ax = axes[i, i]
        ax.axvline( np.median(samples[:,i]), color="g")

    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(np.median(samples[:,xi]), color="g")
            ax.axhline(np.median(samples[:,yi]), color="g")
            ax.plot(np.median(samples[:,xi]), np.median(samples[:,yi]), "sg")
    return


def plot_OIII( x_ref, y_ref, x_trans, y_trans, Ascale=1,  stat=None):
    """
    Plot the OIII line window for the reference and target spectra.
    x_ref : numpy array with the wavelength of the reference spectrum
    y_ref : numpy array with the flux of the reference spectrum
    x_trans : numpy array with the wavelength of the target spectrum
    y_trans : numpy array with the flux of the target spectrum 
    Ascale : float, optional
        Scaling factor for the target spectrum, default is 1.
    stat : str, optional
        If "scaled", the target spectrum will be scaled by Ascale, otherwise it will remain
    """
    plt.figure( figsize=(7.2,5.2))
    plt.title("The OIII line window", fontsize=20)
    plt.plot( x_ref, y_ref, label="Reference")
    plt.plot( x_trans, y_trans, label="Target: unscaled")

    if stat=="scaled":
        plt.plot( x_trans, y_trans*Ascale, label="Target:scaled")

    plt.legend( loc="best", fontsize=15 )
    plt.xticks( fontsize=12 )
    plt.yticks( fontsize=12 )
    plt.xlabel("Wavelength[Angstrom]", fontsize=15)
    plt.ylabel("Flux [erg s$^{-1}$ cm$^{-2}$ angstrom$^{-1}$]", fontsize=15)