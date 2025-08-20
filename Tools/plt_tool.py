import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import matplotlib.cm as cm
import sca_osbl_tool as sot
from constants import pi, nan, kappa
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def multiline(xs, ys, c, ax=None, lgd_list=None, **kwargs):
    """Plot lines with different colorings
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates, each row is one line
    c : iterable container of numbers mapped to colormap, np.array
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection
    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)
    Returns
    lc : LineCollection instance.
    """
    # find axes
    ax = plt.gca() if ax is None else ax
    
    # create LineCollection
    if xs.ndim < ys.ndim:
        xs = np.broadcast_to(xs, ys.shape)
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    # use norm=plt.Normalize(min,max) to control range of c values
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    c = c.ravel()
    lc.set_array(c)
    
    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscale xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    
    # use 'proxy artists' for the legend
    if lgd_list is not None:
        def make_proxy(cvalue, scalar_mappable, **kwargs):
            color = scalar_mappable.cmap(scalar_mappable.norm(cvalue))
            return Line2D([0, 1], [0, 1], color=color, **kwargs)
        proxies = [make_proxy(item, lc, **kwargs) for item in c]
        ax.legend(proxies, lgd_list)
    return lc


def mcolor_line(xs, ys, cs, crange=None, cmap='viridis', axis=None, cs_type='seq', **kwargs):
    """
    Make a multicolored line
    """
    if axis is None:
        axis = plt.gca()
    
    if np.sum(np.isnan(xs)) > 0:
        _,Iclump = sot.get_clumps(xs)
    
    for iclp in Iclump:
        x, y, c = xs[iclp], ys[iclp], cs[iclp]
        # Create a set of line segments so that we can color them individually
        # This creates the points as an N x 1 x 2 array so that we can stack points
        # together easily to get the segments. The segments array for line collection
        # needs to be (numlines) x (points per line) x 2 (for x and y)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create a continuous norm to map from data points to colors
        if crange is None:
            if cs_type=='div':
                cmax = np.floor(np.abs(c).max())
                crange = (-cmax, cmax)
            elif cs_type=='seq':
                crange = (c.min(), c.max())
            else:
                raise ValueError('cs_type {} not supported. Should be \'div\' or \'seq\'')
        norm = plt.Normalize(crange[0], crange[1])
        ncolor = np.diff(crange)*2
        lc = LineCollection(segments, cmap=cm.get_cmap(cmap,ncolor), norm=norm, **kwargs)
        # Set the values used for colormapping
        lc.set_array(c)
        # lc.set_linewidth(lw)
        line = axis.add_collection(lc)
    return line


def mcolor_ylabel(ax, list_of_strings, list_of_colors, anchorpad=0, **kw):
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
    
    # y-axis label
    boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',rotation=90,**kw)) 
                 for text,color in zip(list_of_strings[::-1],list_of_colors[::-1]) ]
    ybox = VPacker(children=boxes, align='center', pad=0, sep=5)
    anchored_ybox = AnchoredOffsetbox(loc=3, child=ybox, pad=anchorpad, frameon=False, bbox_to_anchor=(-0.15, -0.05), 
                                      bbox_transform=ax.transAxes, borderpad=0.)
    ax.add_artist(anchored_ybox)


def plot_accL_spec(omega, Saa, epsilon, omega0, Lfloat, Snn, omegaM, site, i):
    """
    Plot Lagrangian acceleratiion spectrum.
    """
    from sca_osbl_tool import modelaccspL
    
    SaaM = modelaccspL(epsilon, omega0, omega, Lfloat)
    fit_mask = omega <= omegaM
    omegaL = (epsilon/Lfloat**2)**(1/3)
    
    fig = plt.figure(num=1, figsize=(5,4.5), constrained_layout=True, clear=True)
    plt.plot(omega, Saa, lw=0.5, alpha=0.5, label='_nolegend_')
    plt.plot(omega[fit_mask], Saa[fit_mask], c='C0', lw=1.5)
    plt.plot(omega, Snn, '--k', lw=0.5);
    plt.plot(omega, SaaM, lw=2)
    plt.plot([omega0, omegaL, omegaM], epsilon*np.ones(3), 'k+', ms=10, ls='none', mew=2)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(5e-12, 5e-6)
    plt.xlim(2e-5*2*pi, 2e-1*2*pi)
    plt.ylabel(r'$\Phi_{aa}$ [m$^2$ s$^{-1}$]', fontsize=10)
    plt.xlabel(r'$\omega$ [radian s$^{-1}$]');
    plt.legend(['Filtered', 'Noise', 'Fit'], markerscale=1, fontsize=8)
    fig.savefig(f'./Figures/AccL_spec/{site}_drift_{i:03d}.png')
    plt.close()


def plot_regime_diagram(Lax, hLL, axis=None, xpow=2, condition='Destabilizing'):
    """
    Plot regime diagram following Li et al. 2019
    """
    if xpow == 2:
        xyb = [1, 1]
    elif xpow == 1:
        xyb = [6, 1.5]
    else:
        raise ValueError('xpow {} not supported. Should be \'1\' or \'2\'')
    
    # range of power
    xpr = [-2, 1]
    ypr = [-3, 3]
    # range
    xlims = [j*10**i for i,j in zip(xpr,xyb)]
    ylims = [j*10**i for i,j in zip(ypr,xyb)]
    xbpr = sot.round_away_from_0( np.log10(xlims) )
    ybpr = sot.round_away_from_0( np.log10(ylims) )
    
    if condition == 'Destabilizing':
        # background following Fig. 3 of Belcher et al. 2012
        sign_yc = 1
        cnts = plot_regime_diagram_background_BG12(xlims, ylims, axis=axis, xpow=xpow)
        xystr, xystr_log = None, None
    elif condition == 'Stabilizing':
        # background following Fig. 1 of Li et al. 2019
        sign_yc = -1
        plot_regime_diagram_background_heating(xlims, ylims, axis=axis, xpow=xpow)
        cnts = None
        xystr = [(0.16, 2.6e-3), (0.28, 2e-2), (0.42, 2e-1)]
        xystr_log = [(2.6,3)]
    else:
        raise ValueError('Condition {} not supported. Should be \'Destabilizing\' or \'Stabilizing\'')
    
    # Surface cooling or heating
    Lax_cnd, hLL_cnd = filter_stability(Lax, hLL, condition=condition)
    
    # get the bi-dimensional histogram in log-log space
    # xdata = np.log10(Lax_cnd)
    # ydata = np.log10(hLL_cnd*np.sign(hLL_cnd))
    # hist_cnd, xc_cnd, yc_cnd, dx, dy = sot.loglog_hist2d(xdata, ydata, xpr=xbpr, ypr=ybpr, bins=50)
    # Dx, Dy = np.meshgrid(dx, dy, indexing='ij')
    # dxdy = Dx*Dy
    
    # get the bi-dimensional PDF in linear-log space
    binXe = np.logspace(-2, 1.5)
    binYe = np.concatenate([-np.flip(np.logspace(-3, 3.5)), 
                            #np.linspace(-9e-4, 9e-4, int(2e-3/1e-4)-2), 
                            np.logspace(-3, 3.5)])
    hist, xc, yc = sot.hist2d(Lax, hLL, bins=[binXe,binYe], density=True)
    count, xc, yc = sot.hist2d(Lax, hLL, bins=[binXe,binYe])
    
    # plot bi-dimensional histogram
    plot_pdf_contour(hist, xc, sign_yc*yc, axis=axis, levels=[0.6, 0.95, 0.99], fcolors='xkcd:grey', 
                     xystr=xystr, linestyles='-')
    plot_pdf_contour(count, xc, sign_yc*yc, axis=axis, levels=[0.99], fcolors='xkcd:ocean blue', 
                     xystr=xystr_log, rotstr=-48, linestyles=':', linewidths=1)
    return cnts


def plot_regime_diagram_background_BG12(xlims, ylims, axis=None, xpow=2):
    """Plot the background of the regime diagram following Fig. 3 of Belcher et al. 2012
    axis: (matplotlib.axes, optional) axis to plot figure on
    """
    if axis is None:
        axis = plt.gca()
    
    # size of x and y
    nx = 500
    ny = 500
    x = np.logspace(np.log10(xlims[0]), np.log10(xlims[1]), nx)
    y = np.logspace(np.log10(ylims[0]), np.log10(ylims[1]), ny)
    xx, yy = np.meshgrid(x, y)
    text_val = [str(i) for i in [0, 0.1, 0.25, 0.5, 1, 2, 3, 4]]
    if xpow == 2:
        Lat = np.sqrt(xx)
        xlabel_str = r'La$_t^2 = u_* \left/ u^s_0 \right.$'
        text_x = np.array([0.73, 2.72, 11, 12, 12, 3.3, 3.3e-1, 3.3e-2])
        text_y = [3e-3, 3e-3, 1.6, 53, 2.9e2, 1e3, 1e3, 1e3]
        text_rot = [-90, -90, -60, 25, 29, 32, 33, 34]
        turb_x = [0.03, 0.1, 0.9]
        turb_y = [0.1, 0.8, 0.2]
    elif xpow == 1:
        Lat = xx.copy()
        xlabel_str = 'La$_t$'
        text_x = [0.85, 1.65, 3.8, 6, 0.14, 0.07, 0.07, 0.07]
        text_y = [3e-3, 3e-3, 1e-1, 1.5e2, 1e-2, 9e-1, 1.5e1, 1.5e2]
        text_rot = [-90, -90, -90, 29, -90, 32, 33, 34]
        turb_x = [0.07, 4, 0.1]
        turb_y = [2e-3, 2e-2, 1e2]
    else:
        raise ValueError('xpow {} not supported. Should be \'1\' or \'2\'')
    
    zz1 = 2*(1-np.exp(-0.5*Lat))
    zz2 = 0.22*Lat**(-2)
    zz3 = 0.3*Lat**(-2)*yy
    zz = zz1 + zz2 + zz3
    # axis.contourf(xx, yy, np.log10(zz),
    #               levels=[-0.1, 0, 0.1, 0.25, 0.5, 1, 2, 3, 4],
    #               cmap='YlGnBu_r', extend='both', vmin=-10, vmax=5)
    # axis.contour(xx, yy, np.log10(zz),
    #               levels=[-0.1, 0, 0.1, 0.25, 0.5, 1, 2, 3, 4],
    #               colors='darkgray', linewidths=0.5)
    nodes = [0, 0.2, 1]
    cmap1 = LinearSegmentedColormap.from_list('cmap1', list(zip(nodes, ['w', 'w', mcl.to_rgba('xkcd:pale teal',0.8)])))
    cmap2 = LinearSegmentedColormap.from_list('cmap2', list(zip(nodes, ['w', 'w', mcl.to_rgba('xkcd:terracota',0.8)])))
    cmap3 = LinearSegmentedColormap.from_list('cmap3', list(zip(nodes, ['w', 'w', mcl.to_rgba('xkcd:lavender',0.8)])))
    cvals = np.arange(5,10.5,0.5)/10
    c1 = axis.contourf(xx, yy, zz1/zz, levels=cvals, vmax=1.3, cmap=cmap1, linestyles='none')
    c2 = axis.contourf(xx, yy, zz2/zz, levels=cvals, vmax=1.3, cmap=cmap2, linestyles='none')
    c3 = axis.contourf(xx, yy, zz3/zz, levels=cvals, vmax=1.3, cmap=cmap3, linestyles='none')
    axis.contour(xx, yy, zz1/zz, levels=[0.5], colors='k', linewidths=1.5, linestyles='-')
    axis.contour(xx, yy, zz2/zz, levels=[0.5], colors='k', linewidths=1.5, linestyles='-')
    axis.contour(xx, yy, zz3/zz, levels=[0.5], colors='k', linewidths=1.5, linestyles='-')
    # axis.axhline(5, ls='--', c='gray')
    axis.set_xlim(xlims)
    axis.set_ylim(ylims)
    axis.set_xscale('log')
    axis.set_yscale('log')#'symlog', linthresh=1e-3)
    # axis.set_xlabel(xlabel_str)
    axis.set_ylabel(r"$h \left/ L_L \right. = w_*^3 \left/ u_*^2 u^s_0 \right.$")
    axis.set_aspect(aspect=np.diff(np.log10(xlims))/np.diff(np.log10(ylims)))
    # for i in range(len(text_val)):
    #     axis.text(text_x[i], text_y[i], text_val[i], color='gray', fontsize=8, rotation=text_rot[i])
    axis.text(turb_x[0], turb_y[0], 'Langmuir', bbox=dict(boxstyle='square',ec='k',fc='w',alpha=0.6), 
              va='center',ha='left', transform=axis.transAxes)
    axis.text(turb_x[1], turb_y[1], 'Convection', bbox=dict(boxstyle='square',ec='k',fc='w',alpha=0.6), 
              va='center',ha='left', transform=axis.transAxes)
    axis.text(turb_x[2], turb_y[2], 'Shear', bbox=dict(boxstyle='square',ec='k',fc='w',alpha=0.6), 
              va='center',ha='right', transform=axis.transAxes)
    return (c1, c2, c3)


def plot_regime_diagram_background_heating(xlims, ylims, axis=None, xpow=2):
    """
    """
    if axis is None:
        axis = plt.gca()
    
    if xpow == 2:
        xlabel_str = r'La$_t^2 = u_* \left/ u^s_0 \right.$'
        turb_x = [0.03, 0.1, 0.9]
        turb_y = [0.9, 0.2, 0.8]
    elif xpow == 1:
        xlabel_str = 'La$_t$'
        turb_x = [0.07, 4, 0.1]
        turb_y = [2e-3, 2e-2, 1e2]
    else:
        raise ValueError('xpow {} not supported. Should be \'1\' or \'2\'')
    
    # axis.axhline(5, ls='--', c='gray')
    axis.set_xlim(xlims)
    axis.set_ylim(np.flip(ylims))
    axis.set_xscale('log')
    axis.set_yscale('log')#'symlog', linthresh=1e-3)
    axis.set_xlabel(xlabel_str)
    axis.set_ylabel(r'$-h \left/ L_L \right.$')
    axis.set_aspect(aspect=np.diff(np.log10(xlims))/np.diff(np.log10(ylims)))
    axis.text(turb_x[0], turb_y[0], 'Langmuir', bbox=dict(boxstyle='square',ec='k',fc='w',alpha=0.6), 
              va='center',ha='left', transform=axis.transAxes)
    axis.text(turb_x[1], turb_y[1], 'Decaying Turbulence', bbox=dict(boxstyle='square',ec='k',fc='w',alpha=0.6), 
              va='center',ha='left', transform=axis.transAxes)
    axis.text(turb_x[2], turb_y[2], 'Shear', bbox=dict(boxstyle='square',ec='k',fc='w',alpha=0.6), 
              va='center',ha='right', transform=axis.transAxes)


def plot_Lat2_hLL(ds, opt='surf', axis=None, condition='Destabilizing'):
    """
    Plot Lat2 vs hLL in regime diagram.
    """
    if axis is None:
        axis = plt.gca()
    if opt=='surf':
        mask = (ds.Lat2 > 0) & (ds.ustar > 8e-3)
        xdata = ds.Lat2.where(mask, drop=True)
        hLL = -ds.bld*ds.Lat2/ds.LObukhov/kappa
        ydata = hLL.where(mask, drop=True)
    elif opt=='SL':
        mask = (ds.LaSL2 > 0) & (ds.ustar > 8e-3)
        xdata = ds.LaSL2.where(mask, drop=True)
        hLL = -ds.bld*ds.LaSL2/ds.LObukhov/kappa
        ydata = hLL.where(mask, drop=True)
    if condition == 'Destabilizing':
        ysign = 1
        xcnd, ycnd = filter_stability(xdata, ydata, condition=condition)
        axis.scatter(xcnd, ycnd*ysign, 4, lw=0.5, color=ds.color)
    elif condition == 'Stabilizing':
        ysign = -1
        xcnd, ycnd = filter_stability(xdata, ydata, condition=condition)
        axis.scatter(xcnd, ycnd*ysign, 4, lw=0.5, color=ds.color)
    else:
        raise ValueError('Condition {} not supported. Should be \'Destabilizing\' or \'Stabilizing\'')


def plot_regime_diagram_LaSL(Lax=None, hLL=None, axis=None, xpow=2, condition='Destabilizing'):
    """
    Plot regime diagram using surface layer averaged Stokes drift
    """
    if xpow == 2:
        xyb = [1, 1]
    elif xpow == 1:
        xyb = [6, 1.5]
    else:
        raise ValueError('xpow {} not supported. Should be \'1\' or \'2\'')
    
    # range of power
    xpr = [-1, 2]
    ypr = [-3, 3]
    # range
    xlims = [j*10**i for i,j in zip(xpr,xyb)]
    ylims = [j*10**i for i,j in zip(ypr,xyb)]
    xbpr = sot.round_away_from_0( np.log10(xlims) )
    ybpr = sot.round_away_from_0( np.log10(ylims) )
    
    if condition == 'Destabilizing':
        sign_yc = 1
        cnts = plot_regime_diagram_background_LaSL_convec(xlims, ylims, axis=axis, xpow=xpow)
    elif condition == 'Stabilizing':
        sign_yc = -1
        plot_regime_diagram_background_LaSL_heating(xlims, ylims, axis=axis, xpow=xpow)
        cnts = None
    else:
        raise ValueError('Condition {} not supported. Should be \'Destabilizing\' or \'Stabilizing\'')
    return cnts


def plot_regime_diagram_background_LaSL_convec(xlims, ylims, axis=None, xpow=2):
    """
    """
    if axis is None:
        axis = plt.gca()
    # size of x and y
    nx = 500
    ny = 500
    x = np.logspace(np.log10(xlims[0]), np.log10(xlims[1]), nx)
    y = np.logspace(np.log10(ylims[0]), np.log10(ylims[1]), ny)
    xx, yy = np.meshgrid(x, y)
    text_val = [str(i) for i in [0, 0.1, 0.25, 0.5, 1, 2, 3, 4]]
    if xpow == 2:
        LaSL = np.sqrt(xx)
        xlabel_str = r'La$_{SL}^2 = u_* \left/ u^s_{SL} \right.$'
        text_x = np.array([0.73, 2.72, 11, 12, 12, 3.3, 3.3e-1, 3.3e-2])
        text_y = [3e-3, 3e-3, 1.6, 53, 2.9e2, 1e3, 1e3, 1e3]
        text_rot = [-90, -90, -60, 25, 29, 32, 33, 34]
        turb_x = [0.03, 0.1, 0.9]
        turb_y = [0.1, 0.8, 0.2]
    elif xpow == 1:
        LaSL = xx.copy()
        xlabel_str = 'La$_{SL}$'
        text_x = [0.85, 1.65, 3.8, 6, 0.14, 0.07, 0.07, 0.07]
        text_y = [3e-3, 3e-3, 1e-1, 1.5e2, 1e-2, 9e-1, 1.5e1, 1.5e2]
        text_rot = [-90, -90, -90, 29, -90, 32, 33, 34]
        turb_x = [0.07, 4, 0.1]
        turb_y = [2e-3, 2e-2, 1e2]
    else:
        raise ValueError('xpow {} not supported. Should be \'1\' or \'2\'')
    
    wr3our3 = yy*LaSL**(-2)
    cS = 0.64**(3/2)
    cB = 0.31**(3/2) #0.225**(3/2)
    cL = 0.498
    alpha_B = cS / (cS + cB*wr3our3)
    alpha_L = cS / (cS + cL*LaSL**(-2))
    zz1 = alpha_B*alpha_L*cS
    zz2 = cL*LaSL**(-2)
    zz3 = cB*wr3our3
    zz  = zz1 + zz2 + zz3
    nodes = [0, 0.2, 1]
    cmap1 = LinearSegmentedColormap.from_list('cmap1', list(zip(nodes, ['w', 'w', mcl.to_rgba('xkcd:pale teal',0.8)])))
    cmap2 = LinearSegmentedColormap.from_list('cmap2', list(zip(nodes, ['w', 'w', mcl.to_rgba('xkcd:terracota',0.8)])))
    cmap3 = LinearSegmentedColormap.from_list('cmap3', list(zip(nodes, ['w', 'w', mcl.to_rgba('xkcd:lavender',0.8)])))
    cvals = np.arange(5,10.5,0.5)/10
    c1 = axis.contourf(xx, yy, zz1/zz, levels=cvals, vmax=1.3, cmap=cmap1, linestyles='none')
    c2 = axis.contourf(xx, yy, zz2/zz, levels=cvals, vmax=1.3, cmap=cmap2, linestyles='none')
    c3 = axis.contourf(xx, yy, zz3/zz, levels=cvals, vmax=1.3, cmap=cmap3, linestyles='none')
    axis.contour(xx, yy, zz1/zz, levels=[0.5], colors='k', linewidths=1.5, linestyles='-')
    axis.contour(xx, yy, zz2/zz, levels=[0.5], colors='k', linewidths=1.5, linestyles='-')
    axis.contour(xx, yy, zz3/zz, levels=[0.5], colors='k', linewidths=1.5, linestyles='-')
    axis.contour(xx, yy, np.log10(zz), levels=[0, 0.5, 1, 1.5, 2, 2.5], colors='k', linewidths=0.5, linestyles='--')
    # axis.clabel(clogzz, clogzz.levels, inline=True, fontsize=10)
    axis.set_xlim(xlims)
    axis.set_ylim(ylims)
    axis.set_xscale('log')
    axis.set_yscale('log')#'symlog', linthresh=1e-3)
    axis.set_ylabel(r"$h \left/ L_{SL} \right. = w_*^3 \left/ u_*^2 u^s_{SL} \right.$")
    axis.set_aspect(aspect=np.diff(np.log10(xlims))/np.diff(np.log10(ylims)))
    # for i in range(len(text_val)):
    #     axis.text(text_x[i], text_y[i], text_val[i], color='gray', fontsize=8, rotation=text_rot[i])
    axis.text(turb_x[0], turb_y[0], 'Langmuir', bbox=dict(boxstyle='square',ec='k',fc='w',alpha=0.6), 
              va='center',ha='left', transform=axis.transAxes)
    axis.text(turb_x[1], turb_y[1], 'Convection', bbox=dict(boxstyle='square',ec='k',fc='w',alpha=0.6), 
              va='center',ha='left', transform=axis.transAxes)
    axis.text(turb_x[2], turb_y[2], 'Shear', bbox=dict(boxstyle='square',ec='k',fc='w',alpha=0.6), 
              va='center',ha='right', transform=axis.transAxes)
    return (c1, c2, c3)


def plot_regime_diagram_background_LaSL_heating(xlims, ylims, axis=None, xpow=2):
    """
    """
    if axis is None:
        axis = plt.gca()
    
    # size of x and y
    nx = 500
    ny = 500
    x =  np.logspace(np.log10(xlims[0]), np.log10(xlims[1]), nx)
    y = -np.logspace(np.log10(ylims[0]), -0.5, ny)
    xx, yy = np.meshgrid(x, y)
    
    if xpow == 2:
        LaSL = np.sqrt(xx)
        xlabel_str = r'La$_{SL}^2 = u_* \left/ u^s_{SL} \right.$'
        turb_x = [0.03, 0.1, 0.9]
        turb_y = [0.9, 0.2, 0.8]
    elif xpow == 1:
        LaSL = xx.copy()
        xlabel_str = 'La$_{SL}$'
        turb_x = [0.07, 4, 0.1]
        turb_y = [2e-3, 2e-2, 1e2]
    else:
        raise ValueError('xpow {} not supported. Should be \'1\' or \'2\'')
    
    wr3our3 = yy*LaSL**(-2)
    cS = 0.64**(3/2)
    cB = 0.31**(3/2) #0.225**(3/2)
    cL = 0.498
    alpha_B = cS / (cS + cB*wr3our3)
    alpha_L = cS / (cS + cL*LaSL**(-2))
    zz1 = alpha_B*alpha_L*cS
    zz2 = cL*LaSL**(-2)
    zz3 = cB*wr3our3
    zz  = zz1 + zz2 + zz3
    axis.contour(xx, -yy, np.log10(zz), levels=[0, 0.5, 1, 1.5, 2, 2.5], colors='k', linewidths=0.5, linestyles='--')
    # axis.clabel(clogzz, clogzz.levels, inline=True, fontsize=10)
    axis.plot()
    axis.set_xlim(xlims)
    axis.set_ylim(np.flip(ylims))
    axis.set_xscale('log')
    axis.set_yscale('log')#'symlog', linthresh=1e-3)
    axis.set_xlabel(xlabel_str)
    axis.set_ylabel(r'$-h \left/ L_{SL} \right.$')
    axis.set_aspect(aspect=np.diff(np.log10(xlims))/np.diff(np.log10(ylims)))
    axis.text(turb_x[0], turb_y[0], 'Langmuir', bbox=dict(boxstyle='square',ec='k',fc='w',alpha=0.6), 
              va='center',ha='left', transform=axis.transAxes)
    axis.text(turb_x[1], turb_y[1], 'Decaying Turbulence', bbox=dict(boxstyle='square',ec='k',fc='w',alpha=0.6), 
              va='center',ha='left', transform=axis.transAxes)
    axis.text(turb_x[2], turb_y[2], 'Shear', bbox=dict(boxstyle='square',ec='k',fc='w',alpha=0.6), 
              va='center',ha='right', transform=axis.transAxes)


def filter_stability(x, y, condition=None, drop=True):
    xdata = np.copy(x)
    ydata = np.copy(y)
    if condition is None:
        return xdata, ydata
    elif condition == 'Destabilizing':
        # remove data points where y<0 
        inds = ydata < 0
    elif condition == 'Stabilizing':
        # remove data points where y>0
        inds = ydata > 0
    else:
        raise ValueError('Condition {} not supported. Should be \'Destabilizing\' or \'Stabilizing\'')
    # ncon = np.sum(inds)
    # print(condition+' h/L_L: {:6.2f}%'.format(100-ncon/ydata.size*100))
    xdata[inds] = np.nan
    ydata[inds] = np.nan
    if drop==True:
        mask = ~np.isnan(xdata) & ~np.isnan(ydata)
        xdata = xdata[mask]
        ydata = ydata[mask]
    return xdata, ydata

def fmt(x):
    """
    This custom formatter removes trailing zeros, e.g. "1.0" becomes "1", and
    then adds a percent sign.
    """
    xp = x*100
    s = f"{xp:.1f}"
    if s.endswith("0"):
        s = f"{xp:.0f}"
    return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s} %"


def plot_pdf_contour(hst, xc, yc, dxdy=None, nsa=None, axis=None, xystr=None, rotstr=-90, 
                     levels=[0.3, 0.6, 0.9, 0.99], filled=False, fcolors=None, **kwargs):
    """Plot bi-dimensional histogram. Show the contours of the
       histogram which enclose the highest p1%, p2%, ... and pN%
       centered distribution.
    :hst: (2D numpy array) bi-dimensional histogram
    :xc: (1D numpy array) centers of x dimension
    :yc: (1D numpy array) centers of y dimension
    :dxdy: (2D numpy array) bin area
    :axis: (matplotlib.axes, optional) axis to plot figure on
    :levels: (list of float, optional) contour levels, 0.0-1.0
    :filled: (bool) filled contour if True
    :fcolors: (list, optional) color string or sequence of colors
    :return: (matplotlib figure object) figure
    """
    # use current axis if not specified
    if dxdy is None:
        dxdy = np.ones_like(hst)
    if axis is None:
        axis = plt.gca()
    if nsa is None:
        nsa = np.sum(hst)
    # pdfData = hst/nsa
    pdfData = hst/nsa/dxdy
    pdfData_1d = pdfData.flatten()
    # hlist = -np.sort(-pdfData_1d)
    isort = np.argsort(-pdfData_1d)
    hlist = pdfData_1d[isort]
    alist = dxdy.flatten()[isort]
    # hcum = np.cumsum(hlist)
    hcum = np.cumsum(hlist*alist)
    vl = levels
    nv = len(vl)
    vlev = np.zeros(nv)
    for i in np.arange(nv):
        ind = np.argmin(abs(hcum-vl[i]))
        vlev[i] = hlist[ind]
    # pdfData[pdfData==0] = 1e-12
    
    if not filled:
        fig = axis.contour(xc, yc, np.transpose(pdfData),
                           levels=vlev[::-1], colors=fcolors, **kwargs)
        if xystr is not None:
            strs = [f'{100*v:.0f}%' for v in vl]
            for i,str in enumerate(strs):
                axis.text(xystr[i][0], xystr[i][1], str, fontsize=9, color=fcolors, 
                          rotation=rotstr, ha='center', va='center', 
                          bbox=dict(boxstyle='round',ec='w',fc='w',pad=0.05,alpha=0.8))
    else:
        if fcolors is None:
            cmap = cm.get_cmap('bone')
            fcolors = cmap(np.linspace(1.0, 0.0, 11)[0:nv+1])
        else:
            nfc = len(fcolors)
            if nfc != nv+1:
                raise ValueError('Length of fcolors should equal to number of levels + 1.')
        fig = axis.contourf(xc, yc, np.log10(np.transpose(pdfData)),
                            levels=np.log10(vlev[::-1]),
                            colors=fcolors, extend='both', **kwargs)
    return fig


def plot_SL_regime_diagram(axis=None, condition='Destabilizing'):
    """
    Plot regime diagram following Li et al. 2019
    """
    xyb = [1, 1]
    # range of power
    xpr = [-4, 6]
    ypr = [-4, 6]
    # range
    xlims = [j*10**i for i,j in zip(xpr,xyb)]
    ylims = [j*10**i for i,j in zip(ypr,xyb)]
    xbpr = sot.round_away_from_0( np.log10(xlims) )
    ybpr = sot.round_away_from_0( np.log10(ylims) )
    nx, ny = 120, 120
    x = np.logspace(xbpr[0], xbpr[1], nx)[:,None]
    y = np.logspace(ybpr[0], ybpr[1], nx)[:,None]
    xx, yy = np.meshgrid(x, y)
    vfunc = np.vectorize(sot.get_iterative_chim)
    cc = vfunc(xx, yy)
    pp = sot.get_emp_phi(-yy/xx,'mom')
    xxe = cc*xx*pp
    zz = xxe + 1 + yy
    mu = 1/np.sqrt(xx**2 + yy**2)
    zeta_abs = np.logspace(-4,4,5)
    
    nodes = [0, 0.05, 1]
    cmap1 = LinearSegmentedColormap.from_list('cmap1', list(zip(nodes, ['w', 'w', 'xkcd:pale teal'])))
    cmap2 = LinearSegmentedColormap.from_list('cmap2', list(zip(nodes, ['w', 'w', 'xkcd:terracota'])))
    cmap3 = LinearSegmentedColormap.from_list('cmap3', list(zip(nodes, ['w', 'w', 'xkcd:lavender'])))
    cmap4 = LinearSegmentedColormap.from_list('cmap4', list(zip(nodes, ['w', 'w', 'xkcd:grey'])))
    cvals = np.arange(5,11,1)/10
    text_x = [7e5,7e5,7e5,7e3,7e1]
    text_y = [3e1,3e3,3e5,3e5,3e5]
    if condition == 'Destabilizing':
        ysign = 1
        zeta = np.copysign(zeta_abs, -ysign)
        yzeta = -zeta*x
        axis.plot(x, yzeta*ysign, lw=0.5, c='k', ls='--')
        # axis.contour(xx, yy, mu, levels=np.logspace(-1,0,2), colors='k', linewidths=1, linestyles='--')
        axis.contour(xx, yy, xxe/zz, levels=[0.5], colors='k', linewidths=1.5, linestyles='-')
        axis.contour(xx, yy,   1/zz, levels=[0.5], colors='k', linewidths=1.5, linestyles='-')
        axis.contour(xx, yy,  yy/zz, levels=[0.5], colors='k', linewidths=1.5, linestyles='-')
        c1 = axis.contourf(xx, yy, np.ma.masked_where(xxe/zz<0.5, xxe/zz), levels=cvals, vmax=1.5, cmap=cmap1, linestyles='none')
        c2 = axis.contourf(xx, yy, np.ma.masked_where(  1/zz<0.5,   1/zz), levels=cvals, vmax=1.5, cmap=cmap2, linestyles='none')
        c3 = axis.contourf(xx, yy, np.ma.masked_where( yy/zz<0.5,  yy/zz), levels=cvals, vmax=1.5, cmap=cmap3, linestyles='none')
        axis.set_ylim(ylims)
        axis.set_ylabel(r'BF / SSP')
        # axis.text(10, 0.1, f'$\mu = 10^{{{-1}}}$', color='k', fontsize=8, rotation=-90, ha='left', va='top')
        # axis.text(0.9, 3e-3, f'$\mu = 10^{{{0}}}$', color='k', fontsize=8, rotation=-90, ha='right', va='top')
        text_x = [8e5,8e5,8e5,8e3,8e1]
        text_val = [rf'$\zeta = -10^{{{i:n}}}$' for i in np.log10(zeta_abs)]
        for i in range(len(text_val)):
            axis.text(text_x[i], text_y[i], text_val[i], color='k', fontsize=8, rotation=45, ha='right', va='top')
        axis.text(0.03, 0.1, 'Langmuir', bbox=dict(boxstyle='square',ec='k',fc='w',alpha=0.6), 
                  va='center',ha='left', transform=axis.transAxes)
        axis.text(0.05, 0.75, 'Convection', bbox=dict(boxstyle='square',ec='k',fc='w',alpha=0.6), 
                  va='center',ha='left', transform=axis.transAxes)
        axis.text(0.9, 0.2, 'Shear', bbox=dict(boxstyle='square',ec='k',fc='w',alpha=0.6), 
                  va='center',ha='right', transform=axis.transAxes)
    elif condition == 'Stabilizing':
        ysign = -1
        zeta = np.copysign(zeta_abs, -ysign)
        yzeta = -zeta*x
        axis.plot(x, yzeta*ysign, lw=0.5, c='k', ls='--')
        # zze = zz - yy
        # axis.contour(xx, yy, mu, levels=np.logspace(-1,0,2), colors='k', linewidths=1, linestyles='--')
        # axis.contour(xx, yy, np.ma.masked_where(zze<yy, xxe/zze), levels=[0.5], colors='k', linewidths=1.5, linestyles='-')
        # axis.contour(xx, yy, np.ma.masked_where(zze<yy,   1/zze), levels=[0.5], colors='k', linewidths=1.5, linestyles='-')
        axis.contour(xx, yy, xxe/zz, levels=[0.5], colors='k', linewidths=1.5, linestyles='-')
        axis.contour(xx, yy,   1/zz, levels=[0.5], colors='k', linewidths=1.5, linestyles='-')
        axis.contour(xx, yy,  yy/zz, levels=[0.5], colors='gray', linewidths=2, linestyles='-')
        c1 = axis.contourf(xx, yy, np.ma.masked_where(xxe/zz<0.5, xxe/zz), levels=cvals, vmax=1.5, cmap=cmap1, linestyles='none')
        c2 = axis.contourf(xx, yy, np.ma.masked_where(  1/zz<0.5,   1/zz), levels=cvals, vmax=1.5, cmap=cmap2, linestyles='none')
        c3 = axis.contourf(xx, yy, np.ma.masked_where( yy/zz<0.5,  yy/zz), levels=cvals, vmax=1.5, cmap=cmap4, linestyles='none')
        axis.set_xlabel(r'1 / $\eta$ = ESP$_{n}$ / SSP')
        axis.set_ylabel(r'$-$BF / SSP')
        axis.set_ylim(np.flip(ylims))
        # axis.text(10, 0.1, f'$\mu = 10^{{{-1}}}$', color='k', fontsize=8, rotation=-90, ha='left', va='bottom')
        # axis.text(0.9, 3e-3, f'$\mu = 10^{{{0}}}$', color='k', fontsize=8, rotation=-90, ha='right', va='bottom')
        text_val = [rf'$\zeta = 10^{{{i:n}}}$' for i in np.log10(zeta_abs)]
        for i in range(len(text_val)):
            axis.text(text_x[i], text_y[i], text_val[i], color='k', fontsize=8, rotation=-45, ha='right', va='bottom')
        axis.text(8, 2e5, 'production = suppression', color='gray', fontsize=10, rotation=-45, fontweight='bold')
        axis.text(0.03, 0.9, 'Langmuir', bbox=dict(boxstyle='square',ec='k',fc='w',alpha=0.6), 
                  va='center',ha='left', transform=axis.transAxes)
        axis.text(0.05, 0.25, 'Decaying Turbulence', bbox=dict(boxstyle='square',ec='k',fc='w',alpha=0.6), 
                  va='center',ha='left', transform=axis.transAxes)
        axis.text(0.9, 0.8, 'Shear', bbox=dict(boxstyle='square',ec='k',fc='w',alpha=0.6), 
                  va='center',ha='right', transform=axis.transAxes)
    else:
        raise ValueError('Condition {} not supported. Should be \'Destabilizing\' or \'Stabilizing\'')
    axis.set_xlim(xlims)
    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.set_aspect(aspect=np.diff(np.log10(xlims))/np.diff(np.log10(ylims)))
    return (c1, c2, c3)


def plot_SL_prod(ds, use_chim=False, axis=None, condition='Destabilizing'):
    """
    Plot ratios of local production terms
    """
    if axis is None:
        axis = plt.gca()
    mask = ds.maskSL & (ds.SSP>0)
    if use_chim:
        xdata = (ds.ESP*ds.chim/ds.SSP).where(mask, drop=True)#.mean('z')
    else:
        xdata = (ds.ESP/ds.SSP).where(mask, drop=True)#.mean('z')
    ydata = (ds.BFsl/ds.SSP).where(mask, drop=True)#.mean('z')
    
    if condition == 'Destabilizing':
        ysign = 1
        xcnd, ycnd = filter_stability(xdata, ydata, condition=condition)
        axis.scatter(xcnd, ycnd*ysign, 4, lw=0.5, color=ds.color)
    elif condition == 'Stabilizing':
        ysign = -1
        xcnd, ycnd = filter_stability(xdata, ydata, condition=condition)
        axis.scatter(xcnd, ycnd*ysign, 4, lw=0.5, color=ds.color)
    else:
        raise ValueError('Condition {} not supported. Should be \'Destabilizing\' or \'Stabilizing\'')


def add_legend(axis, xloc, yloc, color, text, dx=0.025, fontsize=12, **kwargs):
    axis.scatter(xloc, yloc, color=color, transform=axis.transAxes, clip_on=False, **kwargs)
    axis.text(xloc+dx, yloc, text, transform=axis.transAxes, color='k', clip_on=False, 
              va='center_baseline', fontsize=fontsize)


def add_colorbar(axis, mappable, labelstr, labelc='k', labelpad=-12.5, ticklabel=None, 
                 pos=(0.975, 0, 0.025, 1), labelL=True, **kwargs):
    axins = inset_axes(axis, width='100%', height='100%', loc='lower left', bbox_to_anchor=pos,
                       bbox_transform=axis.transAxes, borderpad=0)
    cbar = plt.colorbar(mappable, cax=axins, **kwargs)
    if ticklabel is not None:
        axins.set_yticklabels(ticklabel) 
    axins.tick_params(axis='y', direction='in', length=2, labelleft=labelL, labelright=False, left=True, right=True)
    cbar.set_label(labelstr, color=labelc, labelpad=labelpad, fontsize=10, fontweight='bold')


def plot_float_traj(yd, zf, axis=None, **kwargs):
    """
    Plot float vertical trajectory time series
    """
    del_tmin = np.diff(yd)*24*60 # in minute
    idx_gap = np.where(del_tmin >= 10)[0]
    yd_ragged = np.insert(yd, idx_gap+1, yd[idx_gap]+5/60/24)
    zf_ragged = np.insert(zf, idx_gap+1, nan)
    if axis is None:
        axis = plt.gca()
    ltraj = axis.plot(yd_ragged, zf_ragged, **kwargs)
    return ltraj


def plot_LObukhov(t, LObukhov, axis=None, **kwargs):
    # note sign(hLL) = -sign(LObukhov)
    tp, Lp = filter_stability(t, -LObukhov, condition='Stabilizing', drop=False)
    LObp = -Lp
    tn, Ln = filter_stability(t, -LObukhov, condition='Destabilizing', drop=False)
    LObn = -Ln
    
    if axis is None:
        axis = plt.gca()
    # plot in ocean vertical coordinate (z<0)
    lobj1, = axis.plot(tp, -LObp, ls=':', **kwargs)
    lobj2, = axis.plot(tn, LObn, ls='-', **kwargs)
    return (lobj1,lobj2)


def plot_sbf(t, sbf, axis=None, showline=True, **kwargs):
    """
    Plot timeseries of surface buoyancy flux [sbf], positive into the ocean [heating or rain]
    """
    if axis is None:
        axis = plt.gca()
    axis.fill_between(t, sbf, 0, where=sbf>0, color='C1', alpha=0.5, lw=0.3, interpolate=True) # positive heating
    axis.fill_between(t, sbf, 0, where=sbf<0, color='C0', alpha=0.5, lw=0.3, interpolate=True)
    if showline is True:
        axis.plot(t, sbf, 'k', lw=0.3)


def plot_cmap_examples(colormaps):
    """
    Helper function to plot random data with associated colormap.
    """
    np.random.seed(19680801)
    data = np.random.randn(30, 30)
    n = len(colormaps)
    plt.close()
    fig, axs = plt.subplots(1, n, figsize=(n * 2 + 2, 3),
                            constrained_layout=True, squeeze=False)
    for [ax, cmap] in zip(axs.flat, colormaps):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)


def plot_binstat(df, c=None, axis=None, condition=None, **kwargs):
    """
    Plot binned statistics.
    """
    if axis is None:
        axis = plt.gca()
    ycnd, xcnd = filter_stability(df.y.s, -df.x.m, condition=condition)
    mask = np.isin(df.y.s, ycnd)
    if condition=='Stabilizing':
        xsign = -1
        xerr = df.xmerr.values[mask,:].T
    elif condition=='Destabilizing':
        xsign = 1
        xerr = np.flipud(df.xmerr.values[mask,:].T)
    if xcnd.size<3:
        hdof = df.n.values[mask] >= 5
        handle = axis.errorbar(xsign*xcnd[hdof], ycnd[hdof], xerr=xerr[:,hdof], yerr=df.yserr.values[mask,:].T[:,hdof],
                 ls='none', mfc=mcl.to_rgba(c,0.5), c=c, marker='o',
                 ms=8, elinewidth=1.5, mew=1.5, ecolor=c, lw=1.5)
        axis.errorbar(xsign*xcnd[~hdof], ycnd[~hdof], xerr=xerr[:,~hdof], yerr=df.yserr.values[mask,:].T[:,~hdof],
                 ls='none', mfc='none', c=c, marker='o',
                 ms=8, elinewidth=1.5, mew=1.5, ecolor=c, lw=1.5)
    else:
        line, = axis.plot(xsign*xcnd, ycnd, c=c, lw=2, **kwargs)
        # xshade = -np.vstack([df.x.ml[mask], df.x.mu[mask]]).ravel(order='F')
        # yshadel = np.vstack([df.y.sl[mask], df.y.sl[mask]]).ravel(order='F')
        # yshadeu = np.vstack([df.y.su[mask], df.y.su[mask]]).ravel(order='F')
        # patch = axis.fill_between(xsign*xshade, yshadel, yshadeu, alpha=0.15, color=c)
        patch = axis.fill_between(xsign*xcnd, df.y.sl[mask], df.y.su[mask], alpha=0.15, color=c)
        handle = (line, patch)
    return handle


def plot_binprof(df, c=None, axis=None, lw=2, alpha=0.15, n_min=5, **kwargs):
    """
    Plot binned statistics as a profile.
    """
    if axis is None:
        axis = plt.gca()
    if df.x.m.size < 5:
        mask = df.n.values >= n_min
        handle = axis.errorbar(df.y.s[mask], df.x.m[mask], xerr=df.yserr.values[mask].T, yerr=df.xmerr.values[mask].T,
                 ls='none', mfc=mcl.to_rgba(c,0.5), c=c, marker='o',
                 ms=8, elinewidth=1.5, mew=1.5, ecolor=c, lw=1.5)
        axis.errorbar(df.y.s[~mask], df.x.m[~mask], xerr=df.yserr.values[~mask].T, yerr=df.xmerr.values[~mask].T,
                 ls='none', mfc='none', c=c, marker='o',
                 ms=8, elinewidth=1.5, mew=1.5, ecolor=c, lw=1.5)
    else:
        line, = axis.plot(df.y.s, df.x.m, c=c, lw=lw, **kwargs)
        patch = axis.fill_betweenx(df.x.m, df.y.sl, df.y.su, color=c, alpha=alpha)
        handle = (line, patch)
    return handle