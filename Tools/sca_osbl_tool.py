import numpy as np
import xarray as xr
import pre_proc_data as ppd
import matplotlib.pyplot as plt
# import pykrige.kriging_tools as kt
import gsw
import warnings
import collections
# from pykrige.ok import OrdinaryKriging
from numba import jit, prange
from scipy import signal, ndimage, stats, optimize, special
from scipy.interpolate import griddata, Rbf
from scipy.integrate import cumulative_trapezoid
from scipy import fft as sp_fft
from tqdm.notebook import tqdm
from plt_tool import plot_accL_spec
from constants import g, cp, rho0, kappa, pi, nan, nat
from pandas import DataFrame, Series, cut, to_datetime, IntervalIndex
from xarray import DataArray, Dataset
from datetime import datetime, timedelta


def xrpd2np(x):
    """
    Convert xarray dataarray and pandas series object to numpy array.
    """
    if type(x) is DataArray or type(x) is Series:
        x = x.values
    return x


def pc_grid_c2e(Xc, Yc):
    """
    Interpolate center values of pcolormesh coordinates to edge vlaues.
    """
    def _interp_pc_grid(X):
        if X.ndim > 1:
            dX = np.diff(X, axis=1)/2
            X = np.hstack((X[:, [0]] - dX[:, [0]],
                           X[:, :-1] + dX,
                           X[:,[-1]] + dX[:, [-1]]))
        else:
            raise ValueError('2D array is required!')
        return X
    
    def _fill_na(X):
        df = DataFrame(X)
        kw0 = dict(method='slinear', fill_value='extrapolate', limit_direction='both', axis=0)
        kw1 = dict(method='linear', fill_value='extrapolate', limit_direction='both', axis=1)
        Xf = df.interpolate(**kw1).interpolate(**kw0)
        return Xf.to_numpy()
    
    def _reshape(X, Y):
        X = xrpd2np(X)
        Y = xrpd2np(Y)
        if X.ndim < Y.ndim:
            X = np.broadcast_to(X, Y.shape)
        elif X.ndim > Y.ndim:
            Y = np.broadcast_to(Y, X.shape)
        return X, Y
    
    Xc, Yc = _reshape(Xc, Yc)
    if np.isnan(Xc).sum() > 0:
        Xc = _fill_na(Xc)
    if np.isnan(Yc).sum() > 0:
        Yc = _fill_na(Yc)
    Xe = _interp_pc_grid(Xc)
    Xe = _interp_pc_grid(Xe.T).T
    Ye = _interp_pc_grid(Yc)
    Ye = _interp_pc_grid(Ye.T).T
    return Xe, Ye


def mtime2pytime(mtime):
    """
    Convert Matlab datenum to Python datetime object.
    Source: https://gist.github.com/conradlee/4366296
            https://sociograph.blogspot.com/2011/04/how-to-avoid-gotcha-when-converting.html
    """
    def _mtime2pytime(x):
        # Matlab day one is 1 Jan 0000, python day one is 1 Jan 0001,
        # hence an increase of 366 days, for year AD 0 was a leap year 
        day = datetime.fromordinal(int(x)-366)
        dfrac = timedelta(days=x%1)
        return day + dfrac
    vfunc = np.vectorize(_mtime2pytime)
    return vfunc(mtime)


def pytime2yd(pytime):
    """
    Convert Python/Numpy datetime to decimal yearday referenced to the start of the year.
    Jan-01 is day one.
    """
    pytime = xrpd2np(pytime).astype('datetime64[s]').tolist()
    def _pytime2yd(x):
        if x is not None:
            epoch = datetime(x.year-1, 12, 31)
            yd = (x - epoch) / timedelta(days=1)
        else:
            yd = nan
        return float(yd)
    vfunc = np.vectorize(_pytime2yd)
    return vfunc(pytime)


def yd2pytime(yd, year):
    """
    Convert decimal yearday (referenced to the start of the year) to Python/Numpy datetime.
    Jan-01 is day one.
    """
    def _yd2pytime(yd, year):
        if np.isnan(yd):
            x = nat
        else:
            epoch = datetime(year-1, 12, 31)
            x = yd*timedelta(days=1) + epoch
        return x
    vfunc = np.vectorize(_yd2pytime)
    return vfunc(xrpd2np(yd), year)


def tcomps2pytime(Y, M, D, h, m, s):
    """
    Assemble datetime from components of datetime.
    """
    Y,M,D,h,m,s = xrpd2np(Y), xrpd2np(M), xrpd2np(D), xrpd2np(h), xrpd2np(m), xrpd2np(s)
    
    # sanity check
    full_sec = s==60
    s[full_sec], m[full_sec] = 0, m[full_sec]+1
    full_min = m==60
    m[full_min], h[full_min] = 0, h[full_min]+1
    
    # convert to 2D arrays
    if Y.ndim == 1: 
        Y,M,D,h,m,s = Y[None,:], M[None,:], D[None,:], h[None,:], m[None,:], s[None,:]
    
    nD = Y.shape[0]
    pytime = [None]*nD
    for i in range(nD):
        df = DataFrame({'year': Y[i,:], 'month': M[i,:], 'day': D[i,:], 
                        'hour': h[i,:], 'minute': m[i,:], 'second': s[i,:]})
        pytime[i] = to_datetime(df).values
    if nD == 1:
        pytime = pytime[0]
    else:
        pytime = np.vstack(pytime)
    return pytime


def get_R2(x, y):
    """
    Compute the coefficient of determination R^2
    x (data) and y (model) are array-like.
    """
    # SS: sum of squares
    SS_res = np.sum((x - y)**2) # residual
    SS_tot = np.sum((x - x.mean())**2)
    R2 = 1 - SS_res/SS_tot
    # slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return R2


def list2matrix(v):
    """
    Convert list of 1D arrays to a 2D array.
    Each 1D array is treated as a column
    Missing data flled with NaN
    Source: https://stackoverflow.com/a/38619350
    """
    lens = np.array([len(item) for item in v])
    mask = lens[:,None] > np.arange(lens.max())
    out = np.full(mask.shape, nan)
    out[mask] = np.concatenate(v)
    return out.T


def shift(arr, num, fill_value=nan, axis=0):
    """
    Shift np array
    Source: https://stackoverflow.com/a/42642326/11959649
    """
    # result = np.empty_like(arr)
    # if num > 0:
    #     result[:num] = fill_value
    #     result[num:] = arr[:-num]
    # elif num < 0:
    #     result[num:] = fill_value
    #     result[:num] = arr[-num:]
    # else:
    #     result[:] = arr
    result = np.roll(arr, num, axis=axis)
    if axis==0:
        result[:num,:] = np.ones(result[:num,:].shape)*fill_value
    elif axis==1:
        result[:,:num] = np.ones(result[:,:num].shape)*fill_value
    return result


def sa_index(sa, a):
    """
    Find the index of subarray sa in array a.
    Source: https://stackoverflow.com/a/69920497/11959649
    """
    sorter = a.argsort()
    i = sorter[np.searchsorted(a, sa, sorter=sorter)]
    return i


def extrap(x, xp, yp):
    """
    Numpy interp function with linear extrapolation.
    Source: https://stackoverflow.com/a/6933490/11959649
    """
    y = np.interp(x, xp, yp)
    y[x<xp[0]] = yp[0] + (x[x<xp[0]]-xp[0]) * (yp[0]-yp[1]) / (xp[0]-xp[1])
    y[x>xp[-1]] = yp[-1] + (x[x>xp[-1]]-xp[-1]) * (yp[-1]-yp[-2]) / (xp[-1]-xp[-2])
    return y


def get_clumps(a, min_len=3, min_da=0):
    """
    Extract contiguous clumps from an array with NaNs.
    Source: https://stackoverflow.com/a/14606271/11959649
    min_len: minimum length of segments
    min_da: minimum range of a values
    """
    a = xrpd2np(a)
    clp_idx = np.ma.clump_unmasked(np.ma.masked_invalid(a))
    clp_idx = [i for i in clp_idx if (i.stop-i.start)>=min_len]
    if min_da!=0:
        clp_idx = [i for i in clp_idx if np.ptp(a[i])>=min_da]
    aclumps = [a[i] for i in clp_idx]
    return aclumps, clp_idx


def get_float_buoyancy(Mass, V0, T0, alpha, gamma, Air, P, Tw, rhow, B):
    """
    Calculate the weight (or buoyancy) of a Lagrangian float in seawater according to Eq.(1) in D'Asaro 2003.
    This version takes trapped air volume as an input.
    Adapted from the Matlab function written by Eric A. D'Asaro.
    Mass, float mass [kg]
    V0, reference float volume [m^3]
    T0, reference temperature [C]
    alpha, float thermal expansion coefficient [1/C]
    gamma, float compressibility [1/dbar]
    Air, trapped air volume [m^3]
    P, water pressure [dbar]
    Tw, in-situ water temperature [C]
    rhow, in-situ water density [kg/m^3]
    B, bocha, piston volume [m^3]
    """
    P[P<0] = 0
    Vf = V0 + B + Air*10/(10+P) + V0*( alpha*(Tw-T0) - gamma*P )
    b = -(Mass - rhow*Vf)*1000 # [g]
    return b

    
def first_turn(a):
    """
    Find the index of first turning point in a time series.
    Source: https://stackoverflow.com/a/3843124/11959649
    """
    da_sign = np.sign(np.diff(a))
    
    if len(np.unique(da_sign)) == 1:
        iturn1 = 0 # no turning poinnt
    else:
        iturn1 = np.where( np.diff(da_sign) )[0][0] + 1
    return iturn1


def last_turn(a):
    """
    Find the index of last turning point in a time series.
    Source: https://stackoverflow.com/a/3843124/11959649
    """
    da_sign = np.sign(np.diff(a))
    
    if len(np.unique(da_sign)) == 1:
        iturn1m = len(a)-1 # no turning poinnt
    else:
        iturn1m = np.where( np.diff(da_sign) )[0][-1] + 1
    return iturn1m


def re_slice(s, dstart, dstop=0):
    """
    Update single slice object with a shift of start index by dstart, 
    and a shift of stop index by dstop.
    Source: https://stackoverflow.com/a/43933590/11959649
    """
    return np.s_[(s.start + dstart):(s.stop + dstop)]


def weight_average(quantity, weight):
    """
    """
    return sum(q*w for q,w in zip(quantity,weight)) / sum(weight)


def get_mean_prof(q, z, dz):
    """
    """
    q_p = q.copy()
    q_p[z>=0] = nan
    zi = np.arange(-200,dz,dz)
    zm = (zi[:-1] + zi[1:])/2
    qm = np.full(len(zm), nan)
    dmq = np.full(len(q), nan)
    
    for i,zbot in enumerate(zi[:-1]):
        ztop = zbot+dz
        is_in_bin = (z>=zbot) & (z<ztop)
        
        if (is_in_bin).sum() > 0:
            qm[i] = np.nanmean(q[is_in_bin])
            dmq[is_in_bin] = q[is_in_bin]-qm[i]
    return qm, zm, dmq


def nanmean(a):
    """
    Avoid RuntimeWarning: 'Mean of empty slice'
    Source: https://stackoverflow.com/a/29688253/11959649
    """
    return nan if np.all(a!=a) or len(a)==0 else np.nanmean(a, keepdims=True)


def move_mean(a, n):
    """
    Compute moving mean of the sequencce (a) with odd window length (n)
    Source: https://stackoverflow.com/a/57897124/11959649
            Pandas rolling average, NaN values ignored,
            edge handled with window length n//2+1
    """
    assert n%2==1, 'please use odd number for window length.'
#     win = signal.windows.hann(n)
    return DataFrame(a).rolling(n, center=True, min_periods=1).mean().to_numpy().ravel()


def wt_sum(var, weights):
    """
    Calculates the weighted sum
    """
    return np.sum(var*weights)/np.sum(weights)


def wt_std(var, weights, wm=None):
    """
    Calculates the weighted standard deviation
    """
    x, w = xrpd2np(var), xrpd2np(weights)
    if wm is None:
        wm = np.average(x, weights=w)
    return np.sqrt(np.average((x-wm)**2, weights=w))


def wt_skew(y, weights=None, qmc=1):
    """
    Calculates the weighted skewness from cubic moments, with optional correction for quadratic moments (qmc)
    """
    # - wt_mean(var, weights)
    # unpack zipped columns and convert tuples to np arrays
    cm, qm = zip(*y)
    cm, qm = np.array(cm), np.array(qm)
    if weights is None:
        weights = np.ones_like(cm)
    return np.average(cm, weights=weights) / np.average(qm*qmc, weights=weights)**(3/2)


def wt_quantiles(values, weights=None, quantiles=0.5, **kwargs):
    """
    Calculates the weighted quantiles.
    """
    values = xrpd2np(values)
    if weights is None:
        weights = np.ones_like(values)
    else:
        weights = xrpd2np(weights)
    i = np.argsort(values, **kwargs)
    c = np.cumsum(weights[i], **kwargs)
    if c.ndim==1:
        quants = values[i[np.searchsorted(c, np.array(quantiles)*c[-1])]]
    # else:
    #     quants = values[i[np.searchsorted(c, np.array(quantiles)*c[-1])]]
    return quants


def bstrap(q, n=None, confidence=0.95, iterations=300,
              sample_size=1.0, statistic=np.average):
    """
    Compute confidence intervals of statistics using bootstrap method.
    Source: https://github.com/mvanga/pybootstrap
    #TODO: complex numbers may cause bug in sorting!
    """
    stats = []
    nsamp = int(len(q)*sample_size)
    if n is None:
        n = np.ones_like(q)
    q, n = xrpd2np(q), xrpd2np(n)
    pool = np.stack([q,n])
    
    for _ in range(iterations):
        # Sample (with replacement) from the given dataset
        sample = np.random.choice(len(q), nsamp)#, p=n/n.sum()) # random sample of index for pairs
        # Calculate user-defined statistic and store it
        stat = statistic(pool[0,sample], weights=pool[1,sample])
        stats.append(stat)
    
    # Sort the array of per-sample statistics and confidence intervals
    ostats = sorted(stats)
    lalp = (1-confidence)/2
    ualp = 1-lalp
    lbnd = np.percentile(ostats, 100*lalp)
    ubnd = np.percentile(ostats, 100*ualp)
    return lbnd, ubnd


def grpf(g, ns_min=1, stat=np.average, xerr='CI'):
    d = []
    ns = g.n.sum()
    nc = len(g.columns)
    lc = g.columns[-1] # last column
    if ns >= ns_min:
        d.append(ns)
        for j in g.columns[1:-1]:
            if xerr=='quat':
                d.append(wt_quantiles(g[j], weights=g.n, quantiles=0.5))
                d.extend(wt_quantiles(g[j], weights=g.n, quantiles=[0.25, 0.75]))
            else:
                wm = np.average(g[j], weights=g.n)
                d.append(wm)
                if xerr=='CI':
                    d.extend(bstrap(g[j], n=g.n, statistic=np.average))
                elif xerr=='std':
                    wstd = wt_std(g[j], g.n)
                    d.extend([wm-wstd, wm+wstd])
                else:
                    raise ValueError(f'Coordinate uncertainty option {xerr} not supprted!')
        d.append(stat(g[lc], weights=g.n))
        d.extend(bstrap(g[lc], n=g.n, statistic=stat))
    else:
        d.append(ns)
        dnull = np.full(3*(nc-1), nan).tolist()
        d.extend(dnull)
    idn = [['n'], 
           ['']]
    idx = []
    idl = [[lc,  lc,   lc], 
           ['s', 'sl', 'su']]
    for j in g.columns[1:-1]:
        idx.append(np.vstack(([j]*3, ['m', 'ml', 'mu'])))
    idall = np.hstack([idn, np.hstack(idx), idl]).tolist()
    return Series(d, index=idall)


def bin_stat_1d(x, y, xbins=None, n=None, mask=None, ns_min=None, dof=None, 
                ystat=np.average, **kwargs):
    """
    Compute binned statistics for y in 1D coordinate x
    n: number of data, or degree of freedom
    mask: boolean array for data used for calculation
    """
    x = xrpd2np(x).T
    if (type(y) is list) and (len(y)==2):
        y, y1 = xrpd2np(y[0]).T, xrpd2np(y[1]).T
    else:
        y  = xrpd2np(y).T
        y1 = np.full_like(y, nan)
    if y.ndim < x.ndim:
        # print('broadcast y to have the same shape as x')
        y  = np.broadcast_to(y, x.shape)
        y1 = np.broadcast_to(y1, x.shape)
    elif y.ndim > x.ndim:
        # print('broadcast x to have the same shape as y')
        x = np.broadcast_to(x, y.shape)
    
    if n is None:
        n = np.ones_like(x)
    if mask is None:
        mask = np.ones_like(x, dtype=bool)
    x1 = kwargs.get('x1', np.full_like(x, nan))
    x2 = kwargs.get('x2', np.full_like(x, nan))
    x1, x2, n, mask = xrpd2np(x1).T, xrpd2np(x2).T, xrpd2np(n).T, xrpd2np(mask).T
    if x1.ndim < x.ndim:
        # print('broadcast x1 to have the same shape as x')
        x1 = np.broadcast_to(x1, x.shape)
    if x2.ndim < x.ndim:
        # print('broadcast x2 to have the same shape as x')
        x2 = np.broadcast_to(x2, x.shape)
    if mask.ndim < x.ndim:
        # print('broadcast mask to have the same shape as x')
        mask = np.broadcast_to(mask, x.shape)
    
    x, x1, x2, n, mask, y, y1 = x.ravel(), x1.ravel(), x2.ravel(), n.ravel(), mask.ravel(), y.ravel(), y1.ravel()
    if len(n[mask])==0:
        print('No data to bin!')
        return None
    else:
        df = DataFrame(np.column_stack((n[mask], x[mask], x1[mask], x2[mask], y[mask], y1[mask])), 
                       columns=['n', 'x', 'x1', 'x2', 'y', 'y1']) \
                      .dropna(axis=1, how='all').dropna(how='any', subset=['n','x','y'])
        if 'y1' in df.columns:
            df['y'] = list(zip(df.y, df.y1))
            df = df.drop('y1', axis=1)
        if xbins is None:
            xbins = bins_from_qcut(df.x, df.n, dof)
            ns_min = 0.1
        grpd = df.groupby(cut(df.x, xbins))
        if ns_min is None:
            ns_min = max(grpd.n.sum().sum()*0.001, 100)
        # using apply, the entire group as a DataFrame gets passed into the function
        bnd = grpd.apply(grpf, ns_min=ns_min, stat=ystat, xerr='quat')
        for j in df.columns[1:-1]:
            bnd[j+'merr','l'] = bnd[j].m  - bnd[j].ml
            bnd[j+'merr','u'] = bnd[j].mu - bnd[j].m
        bnd['yserr','l'] = bnd.y.s  - bnd.y.sl
        bnd['yserr','u'] = bnd.y.su - bnd.y.s
        return bnd


def bins_from_qcut(x, n, dof):
    """
    Discretizes x into equal-sized buckets.
    dof: number of data points in each quantile bin
    """
    Nb = np.clip(int(n.sum()/dof), 1, 45)
    xq = np.linspace(0, 1, Nb+1)[1:-1]
    x_quants = wt_quantiles(x, n, xq)
    Ledges = np.concatenate([[x.min()-1e-3], x_quants])
    Redges = np.concatenate([x_quants, [x.max()+1e-3]])
    xbins = IntervalIndex.from_arrays(Ledges, Redges)
    return xbins


def bin_stat_2d(xlist, ylist, dlist, nlist, masklist, bins=None, dof=None, Nby=3, coord_err='CI', **kwargs):
    """
    bins: (xbin, ybin)
    Nby >= 3 at least 3 bins per column
    """
    x_all, y_all, d_all, n_all, x1_all, x2_all, x3_all = [],[],[],[],[],[],[]
    if type(xlist) is not list:
        x1list = [kwargs.get('x1', np.full_like(xlist, nan))]
        x2list = [kwargs.get('x2', np.full_like(xlist, nan))]
        x3list = [kwargs.get('x3', np.full_like(xlist, nan))]
        xlist, ylist, dlist, nlist, masklist = [xlist], [ylist], [dlist], [nlist], [masklist]
    else:
        x1list = kwargs.get('x1', [np.full_like(e, nan) for e in xlist])
        x2list = kwargs.get('x2', [np.full_like(e, nan) for e in xlist])
        x3list = kwargs.get('x3', [np.full_like(e, nan) for e in xlist])
    for i in range(len(xlist)):
        x = xrpd2np(xlist[i]).T
        y = xrpd2np(ylist[i]).T
        d = xrpd2np(dlist[i]).T
        n = xrpd2np(nlist[i]).T
        x1, x2, x3 = x1list[i].T, x2list[i].T, x3list[i].T
        if y.ndim < x.ndim:
            y = np.broadcast_to(y, x.shape)
        elif y.ndim > x.ndim:
            x  = np.broadcast_to(x, y.shape)
            x1 = np.broadcast_to(x1, y.shape)
            x2 = np.broadcast_to(x2, y.shape)
            x3 = np.broadcast_to(x3, y.shape)
        if d.ndim < x.ndim:
            d = np.broadcast_to(d, x.shape)
        #igd = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(d) & ~np.isnan(n) & ~np.isnan(x1) & ~np.isnan(x2)
        mask = xrpd2np(masklist[i].T)# & igd
        x_all.append(x[mask].ravel())
        y_all.append(y[mask].ravel())
        d_all.append(d[mask].ravel())
        n_all.append(n[mask].ravel())
        x1_all.append(x1[mask].ravel())
        x2_all.append(x2[mask].ravel())
        x3_all.append(x3[mask].ravel())
    xall = np.concatenate(x_all)
    yall = np.concatenate(y_all)
    dall = np.concatenate(d_all)
    nall = np.concatenate(n_all)
    x1all = np.concatenate(x1_all)
    x2all = np.concatenate(x2_all)
    x3all = np.concatenate(x3_all)
    
    if bins is None:
        dfall = DataFrame(np.column_stack((nall, xall, x1all, x2all, x3all, yall, dall)), 
                          columns=['n', 'x', 'x1', 'x2', 'x3', 'y', 'd']) \
                         .dropna(axis=1, how='all').dropna(how='any')#, subset=['n','x','y']
        yq = np.linspace(0, 1, Nby+1)[1:-1]
        Nbx = int(nall.sum()/dof/Nby) # at least half of the DOF in 1d binning (dof>=0.5)
        xq = np.linspace(0, 1, Nbx+1)[1:-1]
        x_quants = wt_quantiles(dfall.x, dfall.n, xq)
        x_Ledges = np.concatenate([[dfall.x.min()-1e-3], x_quants])
        x_Redges = np.concatenate([x_quants, [dfall.x.max()+1e-3]])
        bds_list = []
        
        for j in range(Nbx):
            idx = (dfall.x>x_Ledges[j]) & (dfall.x<=x_Redges[j])
            df = dfall[idx]
            y_quants = wt_quantiles(df.y, df.n, yq)
            y_edges = np.concatenate([[df.y.min()-1e-3], y_quants, [df.y.max()+1e-3]])
            grpd = df.groupby(cut(df.y, y_edges))
            bnd = grpd.apply(grpf, ns_min=0.1, stat=np.average, xerr=coord_err)
            # for c in df.columns[1:-1]:
            #     bnd[c+'merr','l'] = bnd[c].m  - bnd[c].ml
            #     bnd[c+'merr','u'] = bnd[c].mu - bnd[c].m
            # bnd['dserr','l'] = bnd.d.s  - bnd.d.sl
            # bnd['dserr','u'] = bnd.d.su - bnd.d.s
            vard = dict([(v, (['yb','xb','stat'], np.column_stack((bnd[v].ml, bnd[v].m, bnd[v].mu))[:,None,:])) for v in df.columns[1:-1]])
            vard.update({'d': (['yb','xb','stat'], np.column_stack((bnd.d.sl, bnd.d.s, bnd.d.su))[:,None,:])})
            vard.update({'ne': (['yb','xb'], bnd.n.values[:,None])})
            vard.update({'xbl': (['xb'], x_Ledges[:,None][j])})
            vard.update({'xbr': (['xb'], x_Redges[:,None][j])})
            bds = Dataset(data_vars=vard, coords=dict(stat=(['stat'], ['CIl','m','CIu'])))
            bds_list.append(bds)
    else:
        counts = stats.binned_statistic_2d(xall, yall, nall, statistic='sum', bins=bins)[0]
        wt_x_sums = stats.binned_statistic_2d(xall, yall, xall*nall, statistic='sum', bins=bins)[0]
        wt_y_sums = stats.binned_statistic_2d(xall, yall, yall*nall, statistic='sum', bins=bins)[0]
        wt_d_sums = stats.binned_statistic_2d(xall, yall, dall*nall, statistic='sum', bins=bins)[0]
        bstrap(g.y, n=g.n, statistic=ystat)
        sparse_bins = counts < 0.5 # less than half of that in 1d bining
        counts[sparse_bins] = nan
        wt_x_means = wt_x_sums/counts
        wt_y_means = wt_y_sums/counts
        wt_d_means = wt_d_sums/counts
    bds = xr.concat(bds_list, dim='xb')
    return bds


@jit(nopython=True, parallel=True)
def Gaussian_interp2d(xyzd, xg, yg, r, cutoff, yox):
    """
    Interpolate irregularly spaced data onto specified space using a 2D Gaussian weighting function.
    Adapted from Matlab function 'Ggrid' by Mark Prater, JBG & Eric A. D'Asaro
    
    xyzd: row corresponds to data location xd, yd, and data value zd in order
    xg: interpolation location x, in 2D matrix
    yg: interpolation location y, in 2D matrix
    r: Gaussian weighting scale, in units of y
    cutoff: maximum radius of interpolation
    yox: converts x units to y units, i.e. dy/dx
    rbf: 0 - use Guassian (usually better)
         1 - use 1/(1+(dist/r)^4)
         2 - Guassian + 5% linear out to cutoff (experimental)
    """
    # screen NaN
    # ib_bad = np.isnan(xyzd).any(axis=0)
    # xyzd = np.delete(xyzd, ib_bad, axis=1)
    ib_good = np.isnan(xyzd).sum(axis=0) == 0
    xyzd = xyzd[:,ib_good]
    xd, yd, zd = xyzd
    
    rbf = 0
    ny, nx = xg.shape
    zg, zerr = xg*nan, xg*nan

    cutoff2 = cutoff**2
    yox2 = yox**2
    r2 = r**2
    
    for i in prange(ny):
        for j in prange(nx):
            
            dist2 = yox2*(xd - xg[i,j])**2 + (yd - yg[i,j])**2
            md2 = np.min(dist2)
            if md2 > cutoff2:
                continue
            else:
                idx_used = dist2 <= cutoff2
                dist2_used = dist2[idx_used]
            
            if rbf == 0.0: #'gaussian'
                weight = np.exp( -dist2_used/r2 )
            elif rbf == 1.0: #'inverse'
                weight = 1/(1 + (dist2_used/r2)**2 )
            elif rbf == 2.0: #'gaussian-linear'
                dist = np.sqrt(dist2_used)
                weight = np.exp( -dist2_used/r2 ) + \
                         0.05*(1 -dist2_used/cutoff)*(dist < cutoff)
                weight = weight/1.5
            
            sw = weight.sum()
            if sw > 0:
                zg[i,j] = (zd[idx_used]*weight).sum() / sw
            else:
                zg[i,j] = nan

            # calculate weighted RMS deviation
            zerr2 = ((zd[idx_used] - zg[i,j])**2 * weight).sum() / sw
            zerr[i,j] = np.sqrt(zerr2)
    return zg, zerr


def KS_maxdiff(x, test_distribution='uniform'):
    """
    Maximum difference of cumulative distribution function (CDF) in Kolmogorov–Smirnov test.
    """
    # exclude edges in distribution
    x05 = np.percentile(x,  5)
    x95 = np.percentile(x, 95)
    idx_interior = np.logical_and(x > x05, x < x95)
    x_sort = np.sort( x[idx_interior] )
    Nsamp  = len(x_sort)
    cdf_x  = np.arange(Nsamp)/Nsamp
    
    if test_distribution == 'uniform':
        slope = 1/np.ptp(x_sort)
        cdf_test = (x_sort-x_sort[0])*slope
    return abs(cdf_x - cdf_test).max()


def select_LDseg(ep):
    """
    Select Lagrangian drift trajectories using various statistics.
    """
    # separate into segments
    _,Iclump = get_clumps(ep['drift'])
    
    # remove non-Lagrangian trajectories carried by inertia at the beginning of segments
    first_idx = [first_turn(ep['Pmid'][n]) for n in Iclump] # index for the beginning of drift
    
    # remove segments less than 3 hours
    Iseg = [re_slice(n,i) for n,i in zip(Iclump, first_idx) \
                          if np.ptp(ep['yd'][re_slice(n, i)])*24 > 3]
    
    # remove segments with erroneous float bouyancy
#     skewP = [stats.skew(ep['Pmid'][n], nan_policy='omit') for n in Iseg]
#     Iseg = [n for i,n in enumerate(Iseg) if abs(skewP[i])<1]
    # 57 tail increases skewness, # 92 trapped in entrainment layer
    
    # remove segments with partial sampling of OSBL
    zozi = [ep['Pmid'][n]/ep['mld'][n] for n in Iseg]
    Dcdf = [KS_maxdiff(n) for n in zozi]
    Iseg = [i for i,d in zip(Iseg, Dcdf) if d < 0.3]
#     zozi_bin = np.arange(0, .92, .02)
#     pdf_zozi = [np.histogram(ep['Pmid'][n]/ep['mld'][n], zozi_bin, density=True)[0] for n in Iseg]
#     varpdf_zozi = [np.var(n) for n in pdf_zozi]
#     Iseg = [n for i,n in enumerate(Iseg) if varpdf_zozi[i]<1.5]
    
    # ID for drifting segments
    if 'drift_slc' not in ep:
        ep['drift_slc'] = np.full(ep['drift'].shape, nan)
        for j,n in enumerate(Iseg):
            ep['drift_slc'][n] = j+1
    return Iseg


def w_from_pfLDseg(pf, ep, SF, ML):
    """
    """
    Iseg = select_LDseg(ep)
    
    # timestamps for surface fluxes and mixed layer depth
    t_sf = SF['time']['yd']
    t_mld = ML['thr']
    
    # Compute vertical velocity and interpolate surface fluxes
    wseg,YDseg,Pseg, = ([None]*len(Iseg) for i in range(3))
    Fseg = collections.defaultdict(list)
    
    for i,n in enumerate(Iseg):
        # get fast data during the selected drift
        is_drift = (pf['ydP'] >= ep['yd'][n.start]) & \
                   (pf['ydP'] < ep['yd'][n.stop])
        yd,P = pf['ydP'][is_drift], pf['Pmid'][is_drift]
        ys = np.around(yd*24*3600)
        nseq = int(np.ptp(ys) + 1)
        ys_seq = np.arange(nseq) + ys[0]
        P_seq = np.full(nseq, nan)
        P_seq[np.isin(ys_seq, ys)] = P

        w[[0, -1]] = nan # bad results at boundaries
        w = np.interp(ys_seq, ys_seq[~np.isnan(w)], w[~np.isnan(w)])
        w_mdf = ndimage.median_filter(w, size=5)
        dt   = 1 # [s]
        f_Ny = 1/2/dt # [Hz]
        omega_c,omega_Ny = 0.1, f_Ny*2*np.pi # [rad/s]
        b,a = signal.butter(2, omega_c/omega_Ny, btype='low')
        w_lp = signal.filtfilt(b, a, w_mdf)
        
        # resample to 30s intervals
        yd_ep = ep['yd'][n]
        wseg[i] = np.interp(ep['yd'][n]*24*3600, ys_seq, w_lp)
        Pseg[i] = np.interp(ep['yd'][n]*24*3600, ys_seq, P_seq)
        YDseg[i] = yd_ep # do I really need these?
        
        # mixing layer depth
        dist = Pseg[i].copy() # shallow copy
        dist[dist<0] = nan
        H = 2*np.nanmean(dist)
        
        # interpolate mixed layer depth (~ inversion height in ABL)
        zi = np.interp(yd_ep, t_mld, ML['mld'])

        # interpolate surface fluxes
        ustar2 = np.interp(yd_ep, t_sf, SF['stress']['kf'])
        Bnet   = np.interp(yd_ep, t_sf, SF['buoyF']['net'])
        LOb    = -(ustar2**(3/2))/Bnet/kappa
        zeta   = dist/LOb  # referenced to the instantaneous sea surface
        zozi   = dist/zi
        zoH    = dist/H
        wstar3 = H*Bnet
        wstar2 = np.sign(wstar3)*abs(wstar3)**(2/3)
        
        Fseg['ustar2'].append(ustar2)
        Fseg['wstar2'].append(wstar2)
        Fseg['Bnet'].append(Bnet)
        Fseg['LOb'].append(LOb)
        Fseg['zeta'].append(zeta)
        Fseg['H'].append(H)
        Fseg['zi'].append(zi)
        Fseg['zoH'].append(zoH)
        Fseg['zozi'].append(zozi)
    return wseg, Pseg, YDseg, Fseg


def w_from_epLDseg(ep, SF):
    """
    Split pressure records into segments of Lagrangian drift and compute vertical velocity 
    using center difference.
    """
    Iseg = select_LDseg(ep)
    
    # data in each segment
    Pseg   = [ep['Pmid'][n] for n in Iseg]
    YDseg  = [ep['yd'][n] for n in Iseg]
    MODseg = [ep['mode'][n] for n in Iseg]
    mLat   = [ep['lat'][n].mean() for n in Iseg]
    
    # timestamps for surface fluxes
    t_sf = SF['time']['yd']
    
    # Compute vertical velocity and interpolate surface fluxes
    wseg = [None]*len(Pseg)
    Fseg = collections.defaultdict(list)
    
    for i,(ti,Pi) in enumerate(zip(YDseg,Pseg)):
        # interpolate pressure to 30s timestep
        half_mins = ti*24*60*2 # number of 30-s windows
        t_s30 = np.arange(np.ceil(half_mins[0]),np.floor(half_mins[-1])+1)/2/60/24
        P_s30 = np.interp(t_s30,ti,Pi)
#         z_s30 = gsw.z_from_p(P_s30,mLat[i]) # positive values not removed for easy spectral calculation
        dist  = P_s30.copy() # shallow copy
        dist[dist<0] = nan
        
        # interpolate drifting mode
        iMod = MODseg[i].astype(np.float) - 4 # -1: safe, 1: active
        mod_s30 = np.interp(t_s30,ti,iMod)
        
        # interpolate mixed layer depth
        zi_s30 = np.interp(t_s30,ep['yd'],ep['mld']) # ~ inversion height in ABL
        
        # mixing layer depth
        H = 2*np.nanmean(dist)

        # interpolate surface fluxes
        ustar2_s30 = np.interp(t_s30,t_sf,SF['stress']['kf'])
        Bnet_s30   = np.interp(t_s30,t_sf,SF['buoyF']['net'])
        LOb_s30    = -(ustar2_s30**(3/2))/Bnet_s30/kappa
        zeta_s30   = dist/LOb_s30  # referenced to the instantaneous sea surface

        zozi_s30 = dist/zi_s30
        zoH_s30  = dist/H
        wstar3_s30 = H*Bnet_s30
        wstar2_s30 = np.sign(wstar3_s30)*abs(wstar3_s30)**(2/3)
        
        dz_dt = -np.gradient(P_s30,t_s30)/24/3600 # [m/s]
        dz_dt[[0,-1]] = nan # boundary values are not center differencing
        wseg[i]  = dz_dt
        Pseg[i]  = P_s30 # contains some negative values
        YDseg[i] = t_s30
        
        Fseg['ustar2'].append(ustar2_s30)
        Fseg['wstar2'].append(wstar2_s30)
        Fseg['Bnet'].append(Bnet_s30)
        Fseg['LOb'].append(LOb_s30)
        Fseg['zeta'].append(zeta_s30)
        Fseg['H'].append(H)
        Fseg['zi'].append(zi_s30)
        Fseg['zoH'].append(zoH_s30)
        Fseg['zozi'].append(zozi_s30)
        Fseg['mod'].append(mod_s30)
    return wseg, Pseg, YDseg, Fseg


def bin_wwn_seg(wwn_seg, Pseg, YDseg, amod):
    """
    """
    # vertical grid
    Pmax = max([np.max(i) for i in Pseg])
    dz = 5
    zi = np.arange(-dz*np.ceil(Pmax/dz),dz,dz)
    z  = (zi[:-1] + zi[1:])/2
    
    # seperate data into depth bins and average
    wwn_bin = np.full((len(z),len(Pseg)),nan)
    z_bin   = np.full((len(z),len(Pseg)),nan)
    
    for j,(x,p,yd) in enumerate(zip(wwn_seg,Pseg,YDseg)):
        for i,zbot in enumerate(zi[:-1]):
            ztop = zbot+dz
            is_in_bin = (-p>=zbot) & (-p<ztop)
            
            if is_in_bin.sum() > 0:
                if amod == 'bulk':    
                    wwn_bin[i,j] = np.nanmean(x[is_in_bin])
                    z_bin[i,j] = np.nanmean(-p[is_in_bin])
                elif amod == 'weight':
                    is_in_bin_mask = is_in_bin.astype(float)
                    is_in_bin_mask[~is_in_bin] = nan
                    
                    # seperate different trajectories crossing the bin
                    Iclumps = get_clumps(is_in_bin_mask)[1]
                    I_clp = [s for s in Iclumps if (s.stop-s.start) >= 2] # 2 data points at least
                    
                    if I_clp: # empty sequences are false
                        x_clp = [x[ic] for ic in I_clp]
                        p_clp = [p[ic] for ic in I_clp]
                        yd_clp = [yd[ic] for ic in I_clp]

                        mx_clp = [np.nanmean(n) for n in x_clp]
                        mp_clp = [np.nanmean(n) for n in p_clp]
                        dt_clp = [np.ptp(n) for n in yd_clp] # TODO: interpolate to bin edges

                        wwn_bin[i,j] = weight_average(mx_clp,dt_clp)
                        z_bin[i,j] = -weight_average(mp_clp,dt_clp)
    return wwn_bin, z_bin


def prof_from_env(P, T, S, yd, mode, mlon, mlat):
    """
    Extract T, S profiles measured in float profiling mode
    """
    Pdown = np.full(P.shape, nan)
    Pup   = np.full(P.shape, nan)
    prof  = np.full(P.shape, nan)
    is_down = mode==0
    is_up   = mode==2
    Pdown[is_down] = P[is_down]
    Pup[is_up]     = P[is_up]
    
    _,Idown_clp = get_clumps(Pdown)
    Idown_clp = extend_profs(Idown_clp, P, mode, 'down')
    Pdown_clp = [P[a] for a in Idown_clp]
    YDdown_clp = [(yd[a][0] + yd[a][-1])/2 for a in Idown_clp]
    Tdown_clp = [T[a] for a in Idown_clp]
    Sdown_clp = [S[a] for a in Idown_clp]
    # sort in increasing pressure
    downPsort = [np.argsort(a) for a in Pdown_clp]
    Pdown_clp = [a[b] for a,b in zip(Pdown_clp, downPsort)]
    Tdown_clp = [a[b] for a,b in zip(Tdown_clp, downPsort)]
    Sdown_clp = [a[b] for a,b in zip(Sdown_clp, downPsort)]
    
    _,Iup_clp = get_clumps(Pup)
    Iup_clp = extend_profs(Iup_clp, P, mode, 'up')
    Pup_clp = [P[a] for a in Iup_clp]
    YDup_clp = [(yd[a][0] + yd[a][-1])/2 for a in Iup_clp]
    Tup_clp = [T[a] for a in Iup_clp]
    Sup_clp = [S[a] for a in Iup_clp]
    # sort in increasing pressure
    upPsort = [np.argsort(a) for a in Pup_clp]
    Pup_clp = [a[b] for a,b in zip(Pup_clp, upPsort)]
    Tup_clp = [a[b] for a,b in zip(Tup_clp, upPsort)]
    Sup_clp = [a[b] for a,b in zip(Sup_clp, upPsort)]

    # combine downward and upward profiles
    Pprof_clp = Pdown_clp + Pup_clp
    YDprof_clp = YDdown_clp + YDup_clp
    Tprof_clp = Tdown_clp + Tup_clp
    Sprof_clp = Sdown_clp + Sup_clp
    Iprof_clp = Idown_clp + Iup_clp
    
    # sort in time
    zipped = list(zip(YDprof_clp, Pprof_clp, Tprof_clp, Sprof_clp, Iprof_clp))
    YDpf,Ppf,Tpf,Spf,Ipf = zip(*sorted(zipped))
    for i,ipf in enumerate(Ipf):
        prof[ipf] = i
    
    ydprof = np.array(YDpf)
    Pprof = list2matrix(Ppf)
    Tprof = list2matrix(Tpf)
    Sprof = list2matrix(Spf)
    
    SAprof = gsw.SA_from_SP(Sprof, Pprof, mlon, mlat)
    CTprof = gsw.CT_from_t(SAprof, Tprof, Pprof)
    # gsdh = gsw.geo_strf_dyn_height(SAprof, CTprof, Pprof, 0)
    Zprof = gsw.z_from_p(Pprof, mlat)#, gsdh)
    return CTprof, SAprof, Zprof, ydprof, prof


def get_emp_phi(zeta, var, opt=None):
    """
    Formulas using kappa=0.35 (Businger et al. 1971) have been addjusted to 0.4, 
    giving slightly higher vertical shear/temperature gradient scaling.
    """
    zeta = xrpd2np(zeta)
    idp = zeta >  0
    idn = zeta <= 0
    phi = np.full(zeta.shape, nan)
    if var == 'wrms':
        if opt == 'WilsonA':
            phi[idn] = 1.00*(1.0 - 4.5*zeta[idn])**(1/3) # Wilson 2008 (type a)
            phi[idp] = 1.00*np.ones_like(phi[idp])
        elif opt == 'WilsonB':
            phi[idn] = 0.80*(1.0 - 9.5*zeta[idn])**(1/3) # Wilson 2008 (type b)
            phi[idp] = 0.80*np.ones_like(phi[idp])
        elif opt == 'Panofsky':
            phi[idn] = 1.30*(1.0 - 3.0*zeta[idn])**(1/3) # Panofsky et al. 1977
            phi[idp] = 1.30*np.ones_like(phi[idp])
        elif opt == 'KF94':
            phi[idn] = 1.25*(1.0 - 3.0*zeta[idn])**(1/3) # Kaimal and Finnigan 1994
            phi[idp] = 1.25*(1 + 0.2*zeta[idp]) # Kaimal and Finnigan 1994
        else:
            raise ValueError('Option not supprted!')
        # phi[idn] = 1.20*(0.7 - 3.0*zeta[idn])**(1/3) # Andreas et al. 1998
        # phi[idp] = 1.1 + 0.9*zeta[idp]**(0.6) # Pahlow et al. 2001
        # phi[idp] = 1.25*(1 + 3*zeta[idp])**(1/3) # Stiperski & Calaf 2018
    elif var == 'wskew':
        if opt == 'Chiba78':
            phi[idn] = -(0.1 - 0.6*zeta[idn]/kappa/get_emp_phi(zeta[idn],'wrms',opt='WilsonA')**3)
            phi[idp] = nan
        else:
            raise ValueError('Option not supprted!')
    elif var == 'mom':
        if opt == 'KPP':
            idns = zeta < -0.2
            idnw = (zeta >= -0.2) & idn
            phi[idns] = (1.26 - 8.38*zeta[idns])**(-1/3)
            phi[idnw] = (1.0 - 16*zeta[idnw])**(-1/4)
            phi[idp] = 1.00 + 5*zeta[idp]
        elif opt == 'L19':
            phi[idn] = (1.00 - 14*zeta[idn])**(-1/3)
            phi[idp] = 1.00 + 5*zeta[idp]
        elif opt == 'B71':
            phi[idn] = 1.00*(1 - 15*zeta[idn])**(-1/4)
            phi[idp] = 1.00 + 4.7*zeta[idp]
        else:
            phi[idn] = 1.1429*(1 - 13.125*zeta[idn])**(-1/4)
            phi[idp] = 1.1429 + 4.7*zeta[idp]
    elif var == 'heat':
        if opt == 'KPP':
            idns = zeta < -1
            idnw = (zeta >= -1) & idn
            phi[idns] = (-28.86 - 98.96*zeta[idns])**(-1/3)
            phi[idnw] = (1.0 - 16*zeta[idnw])**(-1/2)
            phi[idp] = 1.00 + 5*zeta[idp]
        elif opt == 'B71':
            phi[idn] = 0.74*(1 -  9*zeta[idn])**(-1/2)
            phi[idp] = 0.74 + 4.7*zeta[idp]
        else:
            phi[idn] = 0.8457*(1 - 7.875*zeta[idn])**(-1/2)
            phi[idp] = 0.8457 + 4.7*zeta[idp]
    return phi


def get_emp_f(zoH, var):
    """
    """
    if var == 'wrms':
        f = np.sqrt(1.8)*zoH**(1/3)*(1-0.8*zoH)
    elif var == 'w3':
        f = 1*zoH*(1-0.7*zoH)**3
    return f


def get_emp_bkw2(LaSL):
    """
    Compute the prediction of neutral dimensionless bulk averaged w variance w^2/u*^2, 
    based on the scaling of Harcourt & DAsaro (2008)
    """
    def _get_emp_bkw2(x):
        if np.isnan(x):
            w2our2_bk = nan
        elif x > 1:
            w2our2_bk = 0.64 + 3.5*np.exp(-2.69*x)
        elif x > 0:
            w2our2_bk = 0.398 + 0.48*x**(-4/3)
        else:
            raise ValueError(f'LaSL: {x} <= 0.')
        return w2our2_bk
    vfunc = np.vectorize(_get_emp_bkw2)
    return vfunc(LaSL)


def get_emp_chi(xi, var='mom'):
    """
    Compute the Stokes similarity function chi, according to Large et al. 2019a
    """
    if var == 'mom':
        if np.isnan(xi):
            chi = nan
        elif xi >= 0.72:
            chi = 1.05 - 2.43*0.72 + 1.69*0.72**2
        elif xi >= 0.35:
            chi = 1.05 - 2.43*xi + 1.69*xi**2
        elif xi >= 0:
            chim0 = 1.05 - 2.43*0.35 + 1.69*0.35**2
            chi = chim0 + (0.35-xi)*(1-chim0)/0.35
        else:
            chi = 1
    elif var == 'heat':
        if np.isnan(xi):
            chi = nan
        elif xi >= 0.85:
            chi = 0.8 - 1.3*0.85 + 0.77*0.85**2
        elif xi >= 0.35:
            chi = 0.8 - 1.3*xi + 0.77*xi**2
        elif xi >= 0:
            chis0 = 0.8 - 1.3*0.35 + 0.77*0.35**2
            chi = chis0 + (0.35-xi)*(1-chis0)/0.35
        else:
            chi = 1
    else:
        raise ValueError(f'var: {var} not supported.')
    return chi


def get_iterative_chim(x, y):
    def func(xi, x, y):
        chim = get_emp_chi(xi, var='mom')
        return xi*(chim*x + 1 + y) - 1
    root,_,ier,_ = optimize.fsolve(func, 0.2, args=(x, y), full_output=True)
    if ier!=1:
        print('xi not found!')
    else:
        xi = root[0]
    return get_emp_chi(xi, var='mom')


def get_emp_surfwave_spec(U10, f, wave_age=None, memp='ABY03'):
    """
    """
    if memp == 'ABY03':
        fp = 0.123*g/U10 # equivalent Cp/U10 = 1/2pi/0.123 ~ 1.29
        Ef = 6.15e-3*g**2/fp*(2*pi*f)**(-4)*np.exp(-(fp/f)**4)
    elif memp == 'D85': # assume U10 in prarallel to the peak wave mean direction
        Cp = wave_age*U10
        fp = g/Cp/2/pi
        alpha_D85 = 0.006*wave_age**(-0.55)
        sigma_D85 = 0.08*(1 + 4*wave_age**3)
        if wave_age <= 1:
            gamma_D85 = 1.7 - 6*np.log10(wave_age)
        elif wave_age > 1:
            gamma_D85 = 1.7
        efac = np.exp(-(1 - f/fp)**2/(2*sigma_D85**2)) # exponent of peak enhancement factor
        Ef = alpha_D85*g**2/fp*(2*pi*f)**(-4)*np.exp(-(fp/f)**4)*(gamma_D85**efac)
    return Ef


def get_hmix(yd, Pdrift, ustar, wbnet, dt, DOF=10):
    """
    Compute mixing layer depths from pressure during drifts.
    Adapted from Andrey's Matlab code 'MLD_adaptive.m'
    DOF: degree of freedom
    """
    def _get_hmix(Iclp):
        tclp, Pclp, urclp, wbclp, = tsec[Iclp], Pdrift[Iclp], ustar[Iclp], wbnet[Iclp]
        index = np.arange(len(Pclp))
        subs = index[ibe:-ibe:step]
        
        hhmix, w2fs, fwidth = (np.full((NIT, tclp.size), nan) for i in range(3))
        hhmix[0,:] = Pclp*0 + np.nanmean(Pclp)
        BFsign = np.sign(wbclp)
        w2update = 1.1*urclp**2 + 0.3*BFsign*((wbclp*hhmix[0,:]*2)**2)**(1/3)
        w2update[w2update<1e-10] = 1e-10
        w2fs[0,:] = Pclp*0 + np.nanmean(w2update)
        
        for i in range(1,NIT):
            # update filter width
            fwidth[i,:] = 4*hhmix[i-1,:]/np.sqrt(w2fs[i-1,:])*DOF # 1 DOF = 1 round trip in the boundary layer
            # smooth filter width
            fwidth[i,subs] = butter_lpfilt(fwidth[i,subs], 1/4, 1)
            for k in subs:
                fwk = fwidth[i,k]
                inwin = (tclp >= tclp[k]-fwk/2) & (tclp <= tclp[k]+fwk/2) & ~np.isnan(Pclp)
                r = np.interp((tclp[inwin]-tclp[k])/fwk/2, fx0, fw0)
                r = r/r.sum()
                hhmix[i,k] = (1-alpha)*(Pclp[inwin]*r).sum() + alpha*hhmix[i-1,k]
            
            hhmix[i,:] = np.interp(tclp, tclp[subs], hhmix[i,subs])
            w2update = 1.1*urclp**2 + 0.3*BFsign*((wbclp*hhmix[i,:]*2)**2)**(1/3)
            w2update[w2update<1e-10] = 1e-10
            for k in subs:
                fwk = fwidth[i,k]
                inwin = (tclp >= tclp[k]-fwk/2) & (tclp <= tclp[k]+fwk/2) & ~np.isnan(Pclp)
                r = np.interp((tclp[inwin]-tclp[k])/fwk/2, fx0, fw0)
                r = r/r.sum()
                w2fs[i,k] = (1-alpha)*(w2update[inwin]*r).sum() + alpha*w2fs[i-1,k]
        return 2*np.interp(tclp, tclp[subs], hhmix[-1,subs])
    
    yd, Pdrift, ustar, wbnet = xrpd2np(yd), xrpd2np(Pdrift), xrpd2np(ustar), xrpd2np(wbnet)
    NIT = 10 # number of itreation
    step = int(60*30/dt) # use every <step> number of data points, typical float sampling period is 30s
    ibe = int(step/2) # begin/end at half step
    alpha = 0.25 # recursive filter coefficient
    
    # filter window
    fx0 = np.linspace(-1,1,101)
    fw0 = signal.windows.hann(101) #(1 - np.abs(fx0)**3)**3
    
    tsec = (yd-yd[0])*24*3600
    _,Iclps = get_clumps(Pdrift)
    hmix = np.full_like(Pdrift, nan)
    for iclp in Iclps:
        hmix[iclp] = _get_hmix(iclp)
    # extra 1.11 factor account for z/H distribution not uniform, tapering off near the bottom
    # see Harcourt & D'Asaro 2010 (Fig. 3a)
    return 1.11*hmix

    
def get_mld(A, Z, dsigma, jt=-1):
    """
    Compute mixed layer depths from density profiles.
    Use the mean above it as reference density
    A: density profiles
    Z: vertical coordiante [-], from top to bottom
    dsigma: density threshold
            0.03 for monthly climatology per de Boyer Montégut et al. 2004
            0.005 for Papa
    TO-DO: Brandon Reichl's Potential Engergy anomaly approach (OSM2022)
    """
    if A.ndim == 1: A = A[:,None]
    nz,ntm = A.shape
    
    if Z.ndim == 1: Z = np.tile(Z, (ntm,1)).T
    
    # reference density (mean value above it)
    cum_A = -cumulative_trapezoid(A, Z, axis=0, initial=0)
    cum_layer_h = Z[0,:] - Z
    cum_layer_h[0,:] = cum_layer_h[1,:] # avoid singularity
    A_ref = cum_A/cum_layer_h
    A_ref[0,:] = A[0,:]
    # sigma_ref = np.full((nz,ntm), nan)
    # for j,(sig,z) in enumerate(zip(sigma.T, Z.T)):
        # NaN checking!
        # no_nan = (~np.isnan(sigma[:,j])) & (~np.isnan(z[:,j]))
        # if no_nan.sum() > 1:
        #     z_gt_zref = z[no_nan,j] >= zref
        #     if z_gt_zref.sum() > 0:
        #         sigma_ref[j] = np.median( sigma[no_nan,j][ z_gt_zref ] )
    
    A_mld = A_ref + dsigma
    deltaA = A - A_mld
    
    mld = np.full(ntm, nan)
    for j,(dA,z) in enumerate(zip(deltaA.T, Z.T)):
        no_nan = ~np.isnan(dA)
        
        if no_nan.sum() > 1:
            gd_dA = dA[no_nan]
            gd_z = z[no_nan]
            Iexceed = np.where(gd_dA > 0)[0]
            
            if Iexceed.size > 0: # otherwise, MLD is deeper the bottom of the profile
                Imld = Iexceed[0] # index of the depth at which density first exceeds dsigma
                # always: gd_dA[0] < 0
                # MLD maybe shallower than the top of the profile, if its first point is far away from the surface
                # xp in np.interp must be ascending
                mld[j] = -np.interp(0, gd_dA[Imld-1:Imld+1], gd_z[Imld-1:Imld+1])

    if ((ntm<=2) & (jt>=0)) & (sum(~np.isnan(mld))>0):
        print('visualize MLD calculation...')
        plt.ioff()
        if ntm == 1:
            plt.figure(figsize=(3,5), constrained_layout=True)
            plt.plot(sigma, z, '.-')
            plt.scatter(sigma_ref, zref/2, 25, facecolors='none', edgecolors='tab:orange')
            plt.axvline(sigma_ref, c='tab:orange', ls=':')
            plt.axvline(sigma_mld, c='g', ls=':')
            plt.axhline(-mld, c='g', ls=':')
            plt.xlabel(r'$\sigma_{\theta}$')
        elif ntm == 2:
            _,ax = plt.subplots(1,ntm, figsize=(6,5), sharex=True, sharey=True,
                                constrained_layout=True)
            for j in range(ntm):
                ax[j].plot(sigma[:,j], z[:,j], '.-')
                ax[j].scatter(sigma_ref[j], zref, 25, facecolors='none', edgecolors='tab:orange')
                ax[j].axvline(sigma_ref[j], c='tab:orange', ls=':')
                ax[j].axvline(sigma_mld[j], c='g', ls=':')
                ax[j].axhline(-mld[j], c='g', ls=':')
                ax[j].set_xlabel(r'$\sigma_{\theta}$')
        plt.ylim(-1.15*np.nanmax(mld), 0)
        plt.xlim(np.nanmin(sigma_ref)*0.997, np.nanmax(sigma_mld)*1.01)
        plt.savefig(f'../../Shear_scaling/Figures/tmp_mld_SP2/mld_{jt}.png')
        plt.close()
        plt.ion()
    return mld


def get_Ribs(Rho, Z, ustar):
    """
    Compute surface bulk Richardson number, using neutral logarithimic shear.
    """
    b = -g*Rho/rho0
    Ribs = kappa**2*(b[0,:] - b[1:,:])*(Z[0,:] - Z[1:,:])/(ustar*np.log(Z[1:,:]/Z[0,:]))**2
    return Ribs


def get_Ribw(Rho, Z, w2fs):
    """
    Compute bulk Richardson number, using boundary layer-averaged vertical TKE
    """
    B = -g*Rho/rho0
    Ribw = (B[0,:] - B[1:,:])*(Z[0,:] - Z[1:,:])/w2fs
    return Ribw

                                          
def get_bld(A, Z, ustar, Ribsc):
    """
    Compute boundary layer depth based on surface bulk Richardson number.
    """
    if A.ndim == 1: A = A[:,None]
    nz, ntm = A.shape
    if Z.ndim == 1: Z = np.tile(Z, (ntm,1)).T
    ustar = ustar[None,:]
    
    if np.nanmean(A) < 1000:
        Rho = A + 1000
    else:
        Rho = A.copy()
    
    Ribs = get_Ribs(Rho, Z, ustar)
    deltaR = Ribs - Ribsc
    
    bld = np.full(ntm, nan)
    for j,(dR,z) in enumerate(zip(deltaR.T, Z[1:,:].T)):
        no_nan = ~np.isnan(dR)
        
        if no_nan.sum() > 1:
            gd_dR = dR[no_nan]
            gd_z = z[no_nan]
            Iexceed = np.where(gd_dR > 0)[0]
            
            if Iexceed.size > 0: # otherwise, BLD is deeper the bottom of the profil
                Ibld = Iexceed[0] # index of the depth at which Ribs first exceeds critical value
                if Ibld == 0:
                    # BLD maybe shallower than the top of the profile, if its first point is far away from the surface
                    bld[j] = -np.interp(0, [-1, gd_dR[Ibld]], [Z[0,j], gd_z[Ibld]])
                else:
                    bld[j] = -np.interp(0, gd_dR[Ibld-1:Ibld+1], gd_z[Ibld-1:Ibld+1])
    return bld, Ribs


def get_tbwc(zz, zref):
    """
    Compute weight coefficient for MLDs etsimated from top and bottom CTD.
    zz: the shallowed depth [-] of the [bottom, top] CTD profile.
    zref: refernce layer depth [-]
    """
    return (zz - zref) / (zz-zref).sum()


def extend_profs(Iclp, P, mode, fdir):
    """
    Extend profiles to increase vertical coverage.
    """
    newIclp = Iclp.copy()
    if fdir=='down':
        dvec = [1, -1]
    elif fdir=='up':
        dvec = [-1, 1]
    
    for j,idx in enumerate(Iclp):
        nso,neo = 0,0
        if (idx.start == 0) | (idx.stop == len(P)):
            continue
        pre_not_prof = mode[idx.start-1]!=2
        aft_not_prof = mode[idx.stop+1]!=0

        if (idx.start > 0) & pre_not_prof:
            iw = np.arange(idx.start-1, idx.start+1)
            Pw = P[iw]
            while (np.dot(Pw, dvec) < -0.15):# & (np.dot(Pw, dvec) > -10): 
                nso += 1
                Pw = P[iw-nso]

        if ((idx.stop+1)<len(P)) & aft_not_prof:
            iw = np.arange(idx.stop-1, idx.stop+1)
            Pw = P[iw]
            while (np.dot(Pw, dvec) < -0.15):# & (np.dot(Pw, dvec) > -10): 
                neo += 1
                Pw = P[iw+neo]
        newIclp[j] = re_slice(idx, -nso, neo)
    return newIclp


def butter_lpfilt(data, cutoff, fs, order=2, interp=False):
    """
    Do butterworth low-pass filtering.
    data should be unifromly sampled.
    Allow NaNs at the edge, but only small chunks NaN in the interior.
    """
    y = np.full(data.shape, nan)
    Nyq = fs/2
    normal_cutoff = cutoff/Nyq
    # Get the filter coefficients 
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    
    if not interp:
        data_clps, idx_clps = get_clumps(data)
        for iclp,idata in zip(idx_clps, data_clps):
            if idata.size > np.maximum(1/normal_cutoff, 9):
                y[iclp] = signal.filtfilt(b, a, idata)#, method='gust'
            else:
                y[iclp] = idata.copy()
                print('Data segment too short to filter! Original value is returned.')
    else:
        data_itrp = data.copy()
        nan_mask = np.isnan(data)
        dum_coord = np.arange(np.size(data))
        data_itrp[nan_mask] = np.interp(dum_coord[nan_mask], dum_coord[~nan_mask], data[~nan_mask])
        y = signal.filtfilt(b, a, data_itrp)#, method='gust'
    return y


def get_RuvL(k, theta, rg, alpha):
    """
    ADCP Response functions (Ru) for u velocity in Nortek beam-plane (3,1),
    (Rv) for v velocity in Nortek beam-plane (2,4),
    accounting for beam spread, orbital motions of platform, beam-wave mis-alignment.
    k: wavenumber of surface waves
    theta: ADCP beam angle 
    rg: measurement height
    alpha: the angle of beam (3,1), counter-clockwise from wave direction (to)
    """
    r = 1/np.tan(theta)
    delta   = rg/r
    delta_u = delta*np.cos(alpha)
    delta_v = delta*np.sin(alpha)
    
    if np.cos(alpha) < 1e-3:
        Ru = np.ones(k.shape)
    else:
        Ru = np.cos(k*delta_u) - np.exp(-k*rg) + r*np.sin(k*delta_u)/np.cos(alpha)
        
    if np.sin(alpha) < 1e-3:
        Rv = np.ones(k.shape)
    else:
        Rv = np.cos(k*delta_v) - np.exp(-k*rg) + r*np.sin(k*delta_v)/np.sin(alpha)
    return Ru, Rv


def crt_uv_waves(Uwindow, Vwindow, f2sides, alpha, rg):
    """
    """
    # wave frequencies in [0.04, 0.2] Hz
    idx_fwaves = np.where( (f2sides >= 0.04) & (f2sides <= 0.2) )[0]
    Ubase,Vbase = Uwindow.copy(), Vwindow.copy()
    Ubase[ idx_fwaves,:] = 0
    Ubase[-idx_fwaves,:] = 0
    Vbase[ idx_fwaves,:] = 0
    Vbase[-idx_fwaves,:] = 0
    ubase = np.real_if_close( sp_fft.ifft(Ubase, axis=0) )
    vbase = np.real_if_close( sp_fft.ifft(Vbase, axis=0) )
    uwaves = np.zeros( (ubase.shape[0], ubase.shape[1], len(idx_fwaves)) )
    vwaves = np.zeros( (vbase.shape[0], vbase.shape[1], len(idx_fwaves)) )
    
    # loop through wave frequencies and apply ADCP response functions
    for i,ifw in enumerate(idx_fwaves):
        Uwave,Vwave   = 1j*np.zeros(Uwindow.shape), 1j*np.zeros(Vwindow.shape)
        Uwave[ ifw,:] = Uwindow[ ifw,:]
        Uwave[-ifw,:] = Uwindow[-ifw,:]
        Vwave[ ifw,:] = Vwindow[ ifw,:]
        Vwave[-ifw,:] = Vwindow[-ifw,:]
        uwave = np.real_if_close( sp_fft.ifft(Uwave, axis=0) )
        vwave = np.real_if_close( sp_fft.ifft(Vwave, axis=0) )
        fwave = f2sides[ifw]
        kwave = (2*pi*fwave)**2/g
        Ru,Rv = get_RuvL(kwave, 25/180*pi, rg, alpha[ifw-1]) # alpha doesn't have zero-freq part
        uwaves[:,:,i] = uwave/Ru
        vwaves[:,:,i] = vwave/Rv
    uwindow_cor = ubase + uwaves.sum(axis=-1)
    vwindow_cor = vbase + vwaves.sum(axis=-1)
    return uwindow_cor, vwindow_cor


def mt_cross_spec(Sx, Sy, weights):
    """
    Compute one-sided cross-spectrum between two tapered time-series, derived from a
    multi-taper spectral estimation.
    Sx, Sy: ndarray (K, nf), complex DFTs of the tapered sequence x and y
    weights: ndarray (K, nf) or (K,), weights for different tapers
    """
    nf = Sx.shape[-1]
    if isinstance(weights, (list, tuple)):
        autospectrum = False
        weights_x = weights[0]
        weights_y = weights[1]
        denom = (weights_x**2).sum(axis=0) ** 0.5
        denom *= (weights_y**2).sum(axis=0) ** 0.5
    else:
        autospectrum = True
        weights_x = weights
        weights_y = weights
        denom = (weights**2).sum(axis=0)
    Sxy = ((weights_x*Sx) * (weights_y*Sy).conj()).sum(axis=0) / denom
    # use one-sided spectrum
    iNyq = nf//2 + 1
    one_side = slice(0, iNyq)
    Sxy = Sxy[one_side]
    # double power at duplicated freqs
    # the first & last one (zero & Nyquist) is not paired for even-number fft
    dup_freq = slice(1, -1)
    Sxy[dup_freq] *= 2
    if autospectrum:
        return Sxy.real # imginary part is 0
    return Sxy


def mt_adapt_weights(Sx, eigens, sig2, max_iter=150):
    """
    Find the adaptive weights (two-sided) for K direct spectral estimators of DPSS tapered signals.
    Adapted from the Matlab mutitaper function 'pmtm'
    This version uses the equations from Percival & Walden (1993) pp 368-370
    """
    nf = Sx.shape[-1]
    Sxx = abs(Sx)**2
    # tolerance for acceptance of spectral estimate
    tol = 5e-4*sig2/nf
    a = sig2*(1 - eigens[:,None])
    S1 = np.zeros((1, nf))
    Samt = Sxx[:2,:].mean(axis=0) # initial guess
    # avoid sigularity
    a[a <= 0] = 1e-22
    Samt[Samt == 0] = 1e-64
    # do the iteration
    i = 0
    while abs(Samt-S1).sum() > tol:
        i += 1
        if i > max_iter:
            # warning and return non-converged weights
            e_s = 'Breaking due to iterative meltdown in mt_adaptive_weights.'
            warnings.warn(e_s, RuntimeWarning)
            break
        b = Samt/(Samt*eigens[:,None] + a)
        wk = b**2 * eigens[:,None]
        # update S1 with the weights
        S1 = (wk*Sxx).sum(axis=0) / wk.sum(axis=0)
        Samt, S1 = S1, Samt # swap
    weights = np.sqrt(wk)
    return weights


def mt_psd(x, fs=1, NW=8, low_bias=True, agg='adapt', detrend='linear'):
    """
    Estimate one-sided power spectral density (PSD) using multitaper method.
    Ref: https://github.com/preraulab/multitaper_toolbox/blob/master/python/multitaper_spectrogram_python.py
    x: time series of which PSD is estimated
    fs: sampling frequency in Hz
    NW: (time length)*(half bandwidth in spectral concentration maximization)
    low_bias: use tapers with > 90% energy concentration
    agg: method to combine the PSD estimates of different tapers
    detrend: {False, 'constant', 'linear'}, specify how to detrend data
    """
    Kmax = int(NW*2-1) # number of tapers
    if len(x)%2:
        x = x[1:] # even number of data for fft
    N = len(x)
    if detrend is not False:
        x = signal.detrend(x, type=detrend)
    tapers, eigens = signal.windows.dpss(N, NW=NW, Kmax=Kmax, sym=False, return_ratios=True)
    tapers = tapers/(tapers**2).sum(axis=1).reshape(Kmax, 1) # sym=False gives slightly un-normalized windows
    # tapers has shape (Kmax, N)
    if low_bias:
        keepers = eigens > 0.9
        K = keepers.sum()
        if K < Kmax:
            tapers = tapers[keepers,:]
            eigens = eigens[keepers]
    Sx = sp_fft.fft(tapers*x, n=N, axis=-1)
    if agg == 'unity':
        weights = np.ones((K, 1))
    elif agg == 'eigen':
        weights = np.sqrt(eigens).reshape(K, 1)
    elif agg == 'adapt':
        sig2 = np.var(x)
        weights = mt_adapt_weights(Sx, eigens, sig2)
    else:
        raise ValueError('multitaper weighting method not supported.')
    PSDf = mt_cross_spec(Sx, Sx, weights)
    # don't normalize the spectrum by 1/N as usual, since the taper
    # windows are orthonormal, they effectively scale the signal by 1/N
    PSDf /= fs
    f = sp_fft.rfftfreq(N, d=1/fs)
    return f, PSDf


def mt_csd(x, y, fs=1, NW=8, low_bias=True, agg='adapt', detrend='linear'):
    """
    Estimate one-sided cross-power spectral density (CSD) using multitaper method.
    Ref: https://github.com/preraulab/multitaper_toolbox/blob/master/python/multitaper_spectrogram_python.py
    x, y: time series between which CSD is estimated
    fs: sampling frequency in Hz
    NW: (time length)*(half bandwidth in spectral concentration maximization)
    low_bias: use tapers with > 90% energy concentration
    agg: method to combine the PSD estimates of different tapers
    detrend: {False, 'constant', 'linear'}, specify how to detrend data
    """
    Kmax = int(NW*2-1) # number of tapers
    if len(x)%2:
        x, y = x[1:], y[1:] # even number of data for fft
    N = len(x)
    if detrend is not False:
        x = signal.detrend(x, type=detrend)
        y = signal.detrend(y, type=detrend)
    tapers, eigens = signal.windows.dpss(N, NW=NW, Kmax=Kmax, sym=False, return_ratios=True)
    tapers = tapers/(tapers**2).sum(axis=1).reshape(Kmax, 1) # sym=False gives slightly un-normalized windows
    # tapers has shape (Kmax, N)
    if low_bias:
        keepers = eigens > 0.9
        K = keepers.sum()
        if K < Kmax:
            tapers = tapers[keepers,:]
            eigens = eigens[keepers]
    Sx = sp_fft.fft(tapers*x, n=N, axis=-1)
    Sy = sp_fft.fft(tapers*y, n=N, axis=-1)
    weights = [None]*2
    if agg == 'unity':
        weights[0] = np.ones((K, 1))
        weights[1] = np.ones((K, 1))
    elif agg == 'eigen':
        weights[0] = np.sqrt(eigens).reshape(K, 1)
        weights[1] = np.sqrt(eigens).reshape(K, 1)
    elif agg == 'adapt':
        sig2x, sig2y = np.var(x), np.var(y)
        weights[0] = mt_adapt_weights(Sx, eigens, sig2x)
        weights[1] = mt_adapt_weights(Sy, eigens, sig2y)
    else:
        raise ValueError('multitaper weighting method not supported.')
    CSDf = mt_cross_spec(Sx, Sy, weights)
    CSDf *= 2 # the full CSD has contribution from Syx too, Syx.conj() = Sxy
    # don't normalize the spectrum by 1/N as usual, since the taper
    # windows are orthonormal, they effectively scale the signal by 1/N
    CSDf /= fs
    f = sp_fft.rfftfreq(N, d=1/fs)
    return f, CSDf


def spec_average(omega, spec, nprsv=10, navg=50):
    """
    Average spectrum in frequency bands.
    nprsv: number of frequencies to preserve at the low end.
    navg: number of frequency bands resulted from averaging.
    """
    domega = omega[0]
    prsv_bins = omega[:nprsv] - domega/2
    omega_b = omega[nprsv-1] + domega/2
    omega_e = omega[-1] + domega/2
    avg_bins = np.logspace(np.log10(omega_b), np.log10(omega_e), navg+1)
    omega_bins = np.concatenate((prsv_bins,avg_bins))
    
    df = DataFrame(data=dict(omega=omega, spec=spec))
    dfm = df.groupby(cut(df.omega, omega_bins)).mean().dropna()
    return dfm.omega.values, dfm.spec.values


def get_psd(x, fs=1, method='multitaper', detrend='linear', spec_avg=True):
    """
    Estimate one-sided power spectral density (PSD) for time series x.
    Return PSD in unit of [x]**2/(radian/s)
    fs: sampling frequency in Hz
    method: method for PSD calculation
    """
    if method == 'wosa':
        N = len(x)
        nps = sp_fft.next_fast_len( int(N/4) ) # number of points per segment
        overlap = 0.5 # fraction for overlapping between segments
        nol = int(nps*overlap)
        f, PSDf = signal.welch(x, fs=fs, nperseg=nps, noverlap=nol, detrend=detrend)
    elif method == 'multitaper':
        f, PSDf = mt_psd(x, fs=fs, NW=16, agg='adapt', detrend=detrend)
    else:
        raise ValueError('spectrum method not supported.')
    # convert to radial spectrum and remove zero freq
    omega = 2*pi*f[1:]
    PSD = PSDf[1:]/2/pi
    # average in frequecy bins
    if spec_avg:
        omega, PSD = spec_average(omega, PSD)
    return omega, PSD


def get_csd(x, y, fs=1, method='multitaper', detrend='linear', spec_avg=True):
    """
    Estimate one-sided cross-power spectral density (CSD) for time series x and y.
    Return CSD in unit of [x]**2/(radian/s)
    fs: sampling frequency in Hz
    method: method for CSD calculation
    """
    if y.shape != x.shape:
        raise ValueError('shape mismatch between x, y')
    
    if method == 'wosa':
        N = len(x)
        nps = sp_fft.next_fast_len( int(N/4) ) # number of points per segment
        overlap = 0.5 # fraction for overlapping between segments
        nol = int(nps*overlap)
        f, CSDf = signal.csd(x, y, fs=fs, nperseg=nps, noverlap=nol, detrend=detrend)
    elif method == 'multitaper':
        f, CSDf = mt_csd(x, y, fs=fs, NW=50, agg='adapt', detrend=detrend)
    else:
        raise ValueError('spectrum method not supported.')
    # convert to radial spectrum and remove zero freq
    omega = 2*pi*f[1:]
    CSD = CSDf[1:]/2/pi
    # average in frequecy bins
    if spec_avg:
        omega, CSD = spec_average(omega, CSD)
    return omega, CSD


def modelaccspL(epsilon, omega0, omega, Lfloat):
    """
    Compute the empirical model for Lagrangian acceleration spectrum using
    parameters given by the inputs.
    """
    beta = 1.8
    omegaL = (epsilon/Lfloat**2)**(1/3)
    modsp = beta*epsilon * (1 + (2.2*omega0/omega)**4)**(-0.5) * (1 + 0.4*(omega/omegaL)**2)**(-0.8)
    return modsp


def fit_spec(omega, Saa, Pnoise, Lfloat, show_fig=False, site=None, dftn=None):
    """
    Fit the measured Lagrangian acceleration spectrum to the empirical model
    derived in Lien at al. 1998.
    omegaM: maximum frequency used in fit 
    """
    def sqerr(p):
        epsilon, omega0 = p
        modsp = modelaccspL(epsilon, omega0, omega_data, Lfloat)
        res = np.sum((Saa_data - modsp)**2)
        return res
    epsilon = np.max(Saa[omega<=0.1])
    imax = np.argmax(Saa[omega<=0.1])
    Guess = [epsilon, omega[imax]/5] # initial epsilon and omega0
    # noise cutoff
    Snoise_P = np.full_like(Saa, Pnoise)
    Snoise_w = Snoise_P*omega**2
    Snoise_a = Snoise_w*omega**2
    # find maximum frequency
    idx_noisy = np.where(Snoise_a >= Saa)[0]
    if idx_noisy.size == 0:
        iomegaM = len(omega)
    else:
        iomegaM = idx_noisy[0]
    omega_data, Saa_data = omega[:iomegaM], Saa[:iomegaM]
    omegaM = omega_data.max()
    # minimize residual, note python inner function can access variables outside its scope
    pbounds = [(1e-22, 1e-1), (omega_data[0], omega_data[-1])]
    popt = optimize.minimize(sqerr, Guess, method='Nelder-Mead', bounds=pbounds, 
                             options={'maxiter':1e3, 'disp':False, 'xatol':1e-16, 'fatol':1e-16})
    if not popt.success:
        raise RuntimeError('Failed to fit spectrum')
    epsilon, omega0 = popt.x
    
    if show_fig:
        plot_accL_spec(omega, Saa, epsilon, omega0, Lfloat, Snoise_a, omegaM, site, dftn)
    return epsilon, omega0, omegaM


def crt_wrms_bulk(eps, Lfloat, wF2):
    """
    Correct float measured wrms according to the fit function
    to account for the float size effect.
    """
    def fit_func(x):
        # return -0.206*x**3 + 0.309*x**2 + 0.206*x + 0.97 # D'Asaro et al. 2014
        return -0.356*x**3 + 0.706*x**2 + 0.021*x + 1
    wrmsF = np.sqrt(wF2)
    x = (eps*Lfloat)**(1/3) / wrmsF
    # ilarge = x > 1.263
    # ismall = x < 0.125
    cfac = fit_func(x)
    # cfac[ilarge] = fit_func(1.263)
    # cfac[ismall] = fit_func(0.125)
    return cfac


def crt_wrms_prof(cfac, zoh, cshape='linear', m=0.5):
    """
    Correct float measured wrms profile according to a prescribed shape function.
    """
    cfac, zoh = xrpd2np(cfac), xrpd2np(zoh)
    cfacz2 = np.full_like(zoh, nan)
    if cshape == 'linear':
        iupp = zoh >= -m
        ilow = (zoh < m-1) & (zoh >= -1)
        ibeyond = zoh < -1
        cfacz2[iupp] = zoh[iupp]*(cfac[iupp]**2-1)/(m**2) + (cfac[iupp]**2-1)/m + 1
        cfacz2[ilow] = -(zoh[ilow]+1-m)*(cfac[ilow]**2-1)/(m**2) + 1
        cfacz2[ibeyond] = 1 #2*cfac[ibeyond]**2-1
    elif cshape == 'parabolic':
        a = 2.036
        ibeyond = zoh < -1
        cfacz2[~ibeyond] = ((-zoh[~ibeyond])*(1+zoh[~ibeyond]))**(-1/3)/a * (cfac[~ibeyond]**2-1) + 1
        cfacz2[ibeyond] = 1
    else:
        raise ValueError('Correction function shape not supprted!')
    return np.sqrt(cfacz2)


def auto_cross_spec(uvw, az, coruv=0, alpha=None, rg=None,
                    fs=1, wsecs=128, overlap=.5, merge=1):
    """
    fs: sampling frequency
    wsecs: window length in seconds
    overlap: percent of window overlapping
    merge: number of frequency bands to merge
    """
    pts = len(az) # segment length in data points

    ## break into windows
    Nw = round(fs * wsecs) # window length in number of points
    Nw = Nw-1 if Nw%2!=0 else Nw # make Nw an even number
    windows = int( round( 1/(1-overlap) )*(pts/Nw - 1) + 1 ) # number of windows
    dof = 2*windows*merge # degrees of freedom
    # loop to create a matrix of time series, where COLUMN = WINDOW 
    uwindow  = np.zeros( (Nw,windows) )
    vwindow  = np.zeros( (Nw,windows) )
    wwindow  = np.zeros( (Nw,windows) )
    azwindow = np.zeros( (Nw,windows) )
    for q in range(windows):
        lwin_edge = int( q*(1-overlap)*Nw )
        uwindow[:,q]  = uvw[0, lwin_edge : lwin_edge+Nw ]
        vwindow[:,q]  = uvw[1, lwin_edge : lwin_edge+Nw ]
        wwindow[:,q]  = uvw[2, lwin_edge : lwin_edge+Nw ]
        azwindow[:,q] =    az[ lwin_edge : lwin_edge+Nw ]
    
    ## detrend individual windows (full series already detrended)
    uwindow  = signal.detrend(uwindow,  axis=0)
    vwindow  = signal.detrend(vwindow,  axis=0)
    wwindow  = signal.detrend(wwindow,  axis=0)
    azwindow = signal.detrend(azwindow, axis=0)

    ## taper and rescale (to preserve variance)
    # form taper matrix (columns of taper coef) 
    taper = np.tile( signal.windows.hann(Nw, sym=False), (windows,1) ).T
    # taper each window
    uwindowtaper  =  uwindow * taper
    vwindowtaper  =  vwindow * taper
    wwindowtaper  =  wwindow * taper
    azwindowtaper = azwindow * taper
    # now find the correction factor (comparing old/new variance)
    factu  =  uwindow.std(axis=0) /  uwindowtaper.std(axis=0) # no sqrt?
    factv  =  vwindow.std(axis=0) /  vwindowtaper.std(axis=0)
    factw  =  wwindow.std(axis=0) /  wwindowtaper.std(axis=0)
    factaz = azwindow.std(axis=0) / azwindowtaper.std(axis=0)
    # and correct for the change in variance
    # (multiply each window by it's variance ratio factor)
    uwindowready  =  factu * uwindowtaper
    vwindowready  =  factv * vwindowtaper
    wwindowready  =  factw * wwindowtaper
    azwindowready = factaz * azwindowtaper

    ## FFT
    # calculate Fourier coefs
    Uwindow  = sp_fft.fft(uwindowready,  axis=0)
    Vwindow  = sp_fft.fft(vwindowready,  axis=0)
    Wwindow  = sp_fft.fft(wwindowready,  axis=0)
    AZwindow = sp_fft.fft(azwindowready, axis=0)
    
    ## Inverse FFT for wave frequencies in [0.04, 0.2] and apply response function
    if coruv == 1:
        f2sides = sp_fft.fftfreq(Nw, 1/fs)
        uwindow_cor,vwindow_cor = crt_uv_waves(Uwindow, Vwindow, f2sides, alpha, rg)
        Uwindow = sp_fft.fft(uwindow_cor, axis=0)
        Vwindow = sp_fft.fft(vwindow_cor, axis=0)
    
    # pick out the positive nonzero freq part, including Nyquist freq (negative in fftfreq)
    Uwindow  =  Uwindow[1:(Nw//2+1), :]
    Vwindow  =  Vwindow[1:(Nw//2+1), :]
    Wwindow  =  Wwindow[1:(Nw//2+1), :]
    AZwindow = AZwindow[1:(Nw//2+1), :]
    # POWER SPECTRA DENSITY (auto-spectra & cross-spectra) 
    # divide by wsecs*fs^2 (or Nw*fs)
    UUwindow   = ( Uwindow  * np.conj(Uwindow)  ).real / (wsecs * fs**2)
    VVwindow   = ( Vwindow  * np.conj(Vwindow)  ).real / (wsecs * fs**2)
    WWwindow   = ( Wwindow  * np.conj(Wwindow)  ).real / (wsecs * fs**2)
    AZAZwindow = ( AZwindow * np.conj(AZwindow) ).real / (wsecs * fs**2)
    UVwindow   =   Uwindow  * np.conj(Vwindow)  / (wsecs * fs**2)
    UAZwindow  =   Uwindow  * np.conj(AZwindow) / (wsecs * fs**2)
    VAZwindow  =   Vwindow  * np.conj(AZwindow) / (wsecs * fs**2)
    # the factor of 2 is b/c the FFT is symmetric, and we only used the positive side
    UUwindow[:-1, :]   = 2*UUwindow[:-1, :]
    VVwindow[:-1, :]   = 2*VVwindow[:-1, :]
    WWwindow[:-1, :]   = 2*WWwindow[:-1, :]
    AZAZwindow[:-1, :] = 2*AZAZwindow[:-1, :]
    UVwindow[:-1, :]   = 2*UVwindow[:-1, :]
    UAZwindow[:-1, :]  = 2*UAZwindow[:-1, :]
    VAZwindow[:-1, :]  = 2*VAZwindow[:-1, :]

    ## merge neighboring freq bands (number of bands to merge is a fixed parameter)
    UUwindowmerged   =   np.zeros( (int(Nw/2/merge), windows) )
    VVwindowmerged   =   np.zeros( (int(Nw/2/merge), windows) )
    WWwindowmerged   =   np.zeros( (int(Nw/2/merge), windows) )
    AZAZwindowmerged =   np.zeros( (int(Nw/2/merge), windows) )
    UVwindowmerged   = 1j*np.ones( (int(Nw/2/merge), windows) )
    UAZwindowmerged  = 1j*np.ones( (int(Nw/2/merge), windows) )
    VAZwindowmerged  = 1j*np.ones( (int(Nw/2/merge), windows) )

    for mi in range(merge, Nw//2+1, merge):
        UUwindowmerged[mi//merge-1, :]   = UUwindow[(mi-merge):mi, :].mean(axis=0)
        VVwindowmerged[mi//merge-1, :]   = VVwindow[(mi-merge):mi, :].mean(axis=0)
        WWwindowmerged[mi//merge-1, :]   = WWwindow[(mi-merge):mi, :].mean(axis=0)
        AZAZwindowmerged[mi//merge-1,: ] = AZAZwindow[(mi-merge):mi, :].mean(axis=0)
        UVwindowmerged[mi//merge-1, :]   = UVwindow[(mi-merge):mi, :].mean(axis=0)
        UAZwindowmerged[mi//merge-1, :]  = UAZwindow[(mi-merge):mi, :].mean(axis=0)
        VAZwindowmerged[mi//merge-1, :]  = VAZwindow[(mi-merge):mi, :].mean(axis=0)
    
    # freq range and bandwidth
    bd_DFT = 1/wsecs # original DFT bandwitdh [Hz]
    bandwidth = bd_DFT*merge # bandwitdh [Hz] after merging
    # find middle of each freq band, ONLY WORKS WHEN MERGING ODD NUMBER OF BANDS!
    # f = 1/wsecs + bd_DFT*(merge/2) + bandwidth*np.arange(Nw/2//merge) # Jim's way
    f = 1/wsecs + bd_DFT*(merge/2-0.5) + bandwidth*np.arange(Nw/2//merge)

    # ensemble average windows together
    # Thomson et al. 2018 JTech, Eqs(7-8)
    Suu  = UUwindowmerged.mean(axis=1) 
    Svv  = VVwindowmerged.mean(axis=1)
    Sww  = WWwindowmerged.mean(axis=1)
    Saa  = AZAZwindowmerged.mean(axis=1)
    CQuv = UVwindowmerged.mean(axis=1)
    CQua = UAZwindowmerged.mean(axis=1)
    CQva = VAZwindowmerged.mean(axis=1)
    return f, Suu, Svv, Sww, Saa, CQuv, CQua, CQva


def bin_vel_prof(u, z):
    """
    """
    zbins = np.arange(np.nanmin( np.floor(z) ), 
                      np.nanmax( np.ceil(z)  )+0.2, 0.2)
    return bin_stat(z, u, zbins), bin_stat(z, z, zbins)


def is_overlapping(x, y):
    """
    """
    if np.nanmax(x) < np.nanmin(y):
        ol = (False, 1)
    elif np.nanmin(x) > np.nanmax(y):
        ol = (False, -1)
    else:
        ol = (True, 0)
    return ol


def get_overlap_chunks(z):
    """
    """
    jchunks,zref = [], []
    zi = z[:,0]
    schunk = 0
    for j in range(1, z.shape[1]):
        ol = is_overlapping(zi, z[:,j])
        if ol[0]:
            continue
        else:
            echunk = j # stop exclusive
            jchunks.append(slice(schunk, echunk))
            if ol[1] == 1:
                zref.append( (np.nanmax(zi) + np.nanmin(z[:,j-1]))/2 )
            else:
                zref.append( (np.nanmin(zi) + np.nanmax(z[:,j-1]))/2 )
            # prepare new chunk
            schunk = j
            zi = z[:,j]
    # deal with the last one
    jchunks.append(slice(schunk, None))
    if ol[1] == 1:
        zref.append( (np.nanmax(zi) + np.nanmin(z[:,j]))/2 )
    else:
        zref.append( (np.nanmin(zi) + np.nanmax(z[:,j]))/2 )
    return jchunks, zref


def get_dummy_chunks(z):
    """
    """
    jchunks = []
    schunk = 0
    for j in range(1, z.shape[1]):
        if (j%120 == 0): # 2-min chunk
            echunk = j-1
            jchunks.append(slice(schunk, echunk))
            schunk = j
        else:
            continue
    jchunks.append(slice(schunk, None))
    return jchunks


def get_MRcur(u, z, dc=0):
    """
    dc: switch to use dummy chunks
    """
    jnull = ( ~np.isnan(u) ).sum(axis=0) <= 1
    u = np.delete(u, jnull, axis=1)
    z = np.delete(z, jnull, axis=1)
    z[np.isnan(u)] = nan
    
    # split data into chunks where the depths overlap
    if dc==1:
        jchunks = get_dummy_chunks(z)
        zref_chunks = [np.nanmean(z)]*len(jchunks)
    else:
        jchunks,zref_chunks = get_overlap_chunks(z)
    
    # compute & average relative velocities in each chunk (1 zref per chunk)
    ru,zchunks,nchunks = [], [], []
    mru,mz,zref = [], [], []
    nmin = 20
    for j,jc in enumerate(jchunks):
        uc,zc = u[:,jc], z[:,jc]
        zchunks.append(zc)
        nchunks.append(zc.shape[1])
        iref = np.nanargmin(np.abs(zc - zref_chunks[j]), axis=0)
        ru.append(uc - uc[iref, np.arange(nchunks[j])])
        
        # average in depth bins
        if nchunks[j] >= nmin:
            BSu,BSz = bin_vel_prof(ru[j], zchunks[j])
            Isufficient = BSu['ns'] >= nmin*6
            if Isufficient.sum() >= 3:
                mru.append(BSu['mean'][Isufficient])
                mz.append(BSz['mean'][Isufficient])
                zref.append(zref_chunks[j])
    return mru, mz, zref


def xr_trend(xarr):    
    """
    Computes the trend for an xarray.DataArray over the 'time' variable.
    Returns the slopes and p-values for each time series.
    """
    # getting shapes
    m = np.prod(xarr.shape[1:]).squeeze()
    n = xarr.shape[0]
    
    # creating x and y variables for linear regression
    x = xarr.time.to_pandas().index.to_julian_date().values[:, None]
    y = xarr.to_masked_array().reshape(n, -1)
    
    # ############################ #
    # LINEAR REGRESSION DONE BELOW #
    xm = x.mean(0)  # mean
    ym = y.mean(0)  # mean
    ya = y - ym  # anomaly
    xa = x - xm  # anomaly
    
    # variance and covariances
    xss = (xa ** 2).sum(0) / (n - 1)  # variance of x (with df as n-1)
    yss = (ya ** 2).sum(0) / (n - 1)  # variance of y (with df as n-1)
    xys = (xa * ya).sum(0) / (n - 1)  # covariance (with df as n-1)
    # slope and intercept
    slope = xys / xss
    intercept = ym - (slope * xm)
    # statistics about fit
    df = n - 2
    r = xys / (xss * yss)**0.5
    t = r * (df / ((1 - r) * (1 + r)))**0.5
    p = stats.distributions.t.sf(abs(t), df)
    
    # misclaneous additional functions
    # yhat = dot(x, slope[None]) + intercept
    # sse = ((yhat - y)**2).sum(0) / (n - 2)  # n-2 is df
    # se = ((1 - r**2) * yss / xss / df)**0.5
    
    # preparing outputs
    out = xarr[:2].mean('time')
    # first create variable for slope and adjust meta
    xarr_slope = out.copy()
    xarr_slope.name += '_slope'
    xarr_slope.attrs['units'] = 'units / day'
    xarr_slope.values = slope.reshape(xarr.shape[1:])
    # do the same for the p value
    xarr_p = out.copy()
    xarr_p.name += '_Pvalue'
    xarr_p.attrs['info'] = "If p < 0.05 then the results from 'slope' are significant."
    xarr_p.values = p.reshape(xarr.shape[1:])
    # join these variables
    xarr_out = xarr_slope.to_dataset(name='slope')
    xarr_out['pval'] = xarr_p
    return xarr_out


def saturation_vapor_pressure(T):
    r"""
    Calculate the saturation water vapor (partial) pressure.
    Formula: Bolton 1980 for T in degrees Celsius:
    .. math:: 6.112 e^\frac{17.67T}{T + 243.5}
    """
    sat_pressure_0c = 6.112 # [mbar]
    return sat_pressure_0c * np.exp(17.67*T / (T+243.5))


def rh_from_dewpoint(Tdp, Tair):
    """
    Uses temperature and dewpoint in Celsius to calculate relative
    humidity using the ratio of vapor pressure to saturation vapor pressures.
    """
    if np.max(Tdp)>100:
        Tdp = Tdp - 273.15
        
    if np.max(Tair)>100:
        Tair = Tair - 273.15
    e = saturation_vapor_pressure(Tdp)
    e_s = saturation_vapor_pressure(Tair)
    return e/e_s


def get_Hsww(fctr, Espd, bw):
    """
    Compute the significant wave height of wind waves, according to Zippel et al. 2022
    """
    fe = np.sum(fctr*Espd*bw) / np.sum(Espd*bw)
    iww = np.where(fctr>=fe)[0][0]
    return 4*np.sqrt(np.sum(Espd[iww:]*bw[iww:]))


def get_Hsww_d2f(fctr, Espd):
    """
    Compute the significant wave height of wind waves, according to Zippel et al. 2022
    """
    if np.isnan(Espd).sum()==0:
        fe = np.trapz(fctr*Espd, fctr) / np.trapz(Espd, fctr)
        iww = np.where(fctr>=fe)[0][0]
        return 4*np.sqrt(np.trapz(Espd[iww:], fctr[iww:]))
    else:
        return nan

def get_phimL19(zeta):
    if zeta>0:
        phimL19 = 1 + 5*zeta
    else:
        phimL19 = (1 - 14*zeta)**(-1/3)
    return phimL19


def get_xi(Usdw0, Usdw_SLa, Usdw_10ph, ustar, wstar, bld):
    """
    Compute the fraction of Stokes production.
    """
    PS = 0.94*(1 - 0.9*Usdw_10ph/Usdw0 - 0.1*Usdw_SLa/Usdw0)
    PB = 0.09
    Lat2 = ustar/Usdw0
    PU0 = 2.5
    L = -bld/kappa*(ustar/wstar)**3
    zetaL19 = 0.1*bld/(1+PS/PU0/Lat2)/L
    phim_10ph = get_phimL19(zetaL19)
    fLambda = (PS/Lat2 - 0.91 - 3.6/phim_10ph)/2
    PU = np.sqrt(fLambda**2 + 0.91*PS/Lat2) - fLambda
    xi = (PS/Lat2 / (PS/Lat2 + PU + PB*(np.abs(wstar)/ustar)**3)) # xi should be less than 1
    return xi
    

def get_Usdw(z, fctr, bw, a1, b1, theta_tau, tail=True):
    """
    Compute the downwind Stokes drift.
    """
    if z.size==1 and z>0:
        Usdw, Uscw = nan, nan
    else:
        if z.size>1:
            z = z[None,:]
            fctr = fctr[:,None]
            bw = bw[:,None]
            a1 = a1[:,None]
            b1 = b1[:,None]
        multi = 16*pi**3/g
        mu = 8*pi**2/g
        fc = fctr[-1] + bw[-1]/2
        Ec = np.sqrt(a1[-1]**2 + b1[-1]**2)
        us = -multi*np.sum(fctr**3*pi*b1*np.exp(z*mu*fctr**2)*bw, axis=0)
        vs = -multi*np.sum(fctr**3*pi*a1*np.exp(z*mu*fctr**2)*bw, axis=0)
        if tail:
            Us_tail = multi*fc**5*Ec*pi*(np.exp(z*mu*fc**2)/fc - \
                                         np.sqrt(-z*mu*pi)*special.erfc(fc*np.sqrt(-z*mu)))
        else:
            Us_tail = 0
        theta_tail = np.arctan2(-a1, -b1)[-10:].mean()
        Usdw =  us*np.cos(theta_tau) + vs*np.sin(theta_tau) + Us_tail*np.cos(theta_tail-theta_tau)
        Uscw = -us*np.sin(theta_tau) + vs*np.cos(theta_tau) + Us_tail*np.sin(theta_tail-theta_tau)
    return Usdw, Uscw


def get_Usdw_d2f(z, fctr, us_d2f, vs_d2f, theta_tau, tail=True):
    """
    Compute the downwind Stokes drift.
    """
    if z.size==1 and z>0:
        Usdw, Uscw = nan, nan
    else:
        if z.size>1:
            z = z[None,:]
            fctr = fctr[:,None]
            us_d2f = us_d2f[:,None]
            vs_d2f = vs_d2f[:,None]
        multi = 16*pi**3/g
        mu = 8*pi**2/g
        fc = fctr[-1]
        Ec = np.sqrt(us_d2f[-1]**2 + vs_d2f[-1]**2)
        us = multi*np.trapz(fctr**3*us_d2f*np.exp(z*mu*fctr**2), fctr, axis=0)
        vs = multi*np.trapz(fctr**3*vs_d2f*np.exp(z*mu*fctr**2), fctr, axis=0)
        if tail:
            Us_tail = multi*fc**5*Ec*(np.exp(z*mu*fc**2)/fc - \
                                      np.sqrt(-z*mu*pi)*special.erfc(fc*np.sqrt(-z*mu)))
        else:
            Us_tail = 0
        theta_tail = np.arctan2(vs_d2f, us_d2f)[-10:].mean()
        Usdw =  us*np.cos(theta_tau) + vs*np.sin(theta_tau) + Us_tail*np.cos(theta_tail-theta_tau)
        Uscw = -us*np.sin(theta_tau) + vs*np.cos(theta_tau) + Us_tail*np.sin(theta_tail-theta_tau)
    return Usdw, Uscw


def get_dUsdwdz(z, fctr, bw, a1, b1, theta_tau, downwind=True, tail=True):
    """
    Compute the downwind Stokes drift shear.
    """
    if z.size==1 and z>=0:
        dUsdwdz, dUscwdz = nan, nan
    else:
        if z.size>1:
            z = z[None,:]
            fctr = fctr[:,None]
            bw = bw[:,None]
            a1 = a1[:,None]
            b1 = b1[:,None]
        multi1 = 128*pi**5/(g**2)
        multi2 = 16*pi**3/g
        mu = 8*pi**2/g
        fc = fctr[-1] + bw[-1]/2
        Ec = np.sqrt(a1[-1]**2 + b1[-1]**2)
        dusdz_re = -multi1*np.sum(fctr**5*pi*b1*np.exp(mu*fctr**2*z)*bw, axis=0)
        dvsdz_re = -multi1*np.sum(fctr**5*pi*a1*np.exp(mu*fctr**2*z)*bw, axis=0)
        if tail:
            dUsdz_tail = multi2*fc**5*np.sqrt(-2*pi**3/g/z)*special.erfc(fc*np.sqrt(-mu*z))*Ec*pi # saturation range tail
        else:
            dUsdz_tail = 0
        theta_tail = np.arctan2(-a1, -b1)[-10:].mean()
        if downwind:
            dUsdwdz =  dusdz_re*np.cos(theta_tau) + dvsdz_re*np.sin(theta_tau) + dUsdz_tail*np.cos(theta_tail-theta_tau)
            dUscwdz = -dusdz_re*np.sin(theta_tau) + dvsdz_re*np.cos(theta_tau) + dUsdz_tail*np.sin(theta_tail-theta_tau)
        else:
            dUsdwdz = np.sqrt( (dusdz_re + dUsdz_tail*np.cos(theta_tail))**2 + \
                               (dvsdz_re + dUsdz_tail*np.sin(theta_tail))**2 )
            dUscwdz = 0
    return dUsdwdz, dUscwdz


def get_dUsdwdz_d2f(z, fctr, us_d2f, vs_d2f, theta_tau, tail=True):
    """
    Compute the downwind Stokes drift shear.
    """
    if z.size==1 and z>=0:
        dUsdwdz, dUscwdz = nan, nan
    else:
        if z.size>1:
            z = z[None,:]
            fctr = fctr[:,None]
            us_d2f = us_d2f[:,None]
            vs_d2f = vs_d2f[:,None]
        multi1 = 128*pi**5/(g**2)
        multi2 = 16*pi**3/g
        mu = 8*pi**2/g
        fc = fctr[-1]
        Ec = np.sqrt(us_d2f[-1]**2 + vs_d2f[-1]**2)
        dusdz_re = multi1*np.trapz(fctr**5*us_d2f*np.exp(mu*fctr**2*z), fctr, axis=0)
        dvsdz_re = multi1*np.trapz(fctr**5*vs_d2f*np.exp(mu*fctr**2*z), fctr, axis=0)
        if tail:
            dUsdz_tail = multi2*fc**5*np.sqrt(-2*pi**3/g/z)*special.erfc(fc*np.sqrt(-mu*z))*Ec
        else:
            dUsdz_tail = 0
        theta_tail = np.arctan2(vs_d2f, us_d2f)[-10:].mean()
        dUsdwdz =  dusdz_re*np.cos(theta_tau) + dvsdz_re*np.sin(theta_tau) + dUsdz_tail*np.cos(theta_tail-theta_tau)
        dUscwdz = -dusdz_re*np.sin(theta_tau) + dvsdz_re*np.cos(theta_tau) + dUsdz_tail*np.sin(theta_tail-theta_tau)
    return dUsdwdz, dUscwdz


def get_dUsdwdzM(mz, zstd, fctr, bw, a1, b1, theta_tau, downwind=True, tail=True):
    """
    Compute the mean downwind Stokes drift shear over a vertical layer.
    """
    if mz>=0 or np.isnan(mz):
        dUsdwdzM = nan
    elif zstd==0 or np.isnan(zstd):
        dUsdwdzM = get_dUsdwdz(mz, fctr, bw, a1, b1, theta_tau, downwind=downwind, tail=tail)[0]
    else:
        z = np.arange(np.floor((mz-zstd)*10)/10, np.ceil((mz+zstd)*10)/10, 0.1)
        dUsdwdzM = get_dUsdwdz(z, fctr, bw, a1, b1, theta_tau, downwind=downwind, tail=tail)[0].mean()
    return dUsdwdzM


def get_dUsdwdzM_d2f(mz, zstd, fctr, us_d2f, vs_d2f, theta_tau, tail=True):
    """
    Compute the mean downwind Stokes drift shear over a vertical layer.
    """
    if mz>=0 or np.isnan(mz):
        dUsdwdzM = nan
    elif zstd==0 or np.isnan(zstd):
        dUsdwdzM = get_dUsdwdz_d2f(mz, fctr, us_d2f, vs_d2f, theta_tau, tail=tail)[0]
    else:
        z = np.arange(np.floor((mz-zstd)*10)/10, np.ceil((mz+zstd)*10)/10, 0.1)
        dUsdwdzM = get_dUsdwdz_d2f(z, fctr, us_d2f, vs_d2f, theta_tau, tail=tail)[0].mean()
    return dUsdwdzM


def get_oneozM(mz, zstd):
    """
    Compute the mean Eulerian shear over a vertical layer.
    """
    if mz>=0 or np.isnan(mz):
        oneozM = nan
    elif zstd==0 or np.isnan(zstd):
        oneozM = 1/np.abs(mz)
    else:
        z = np.arange(np.floor((mz-zstd)*10)/10, np.ceil((mz+zstd)*10)/10, 0.1)
        oneozM = (1/np.abs(z)).mean()
    return oneozM


def get_fzSM(mz, zstd, Ls):
    """
    Compute the mean surface proximity function over a vertical layer.
    """
    if mz>=0 or np.isnan(mz):
        fzSM = nan
    elif zstd==0 or np.isnan(zstd):
        fzSM = 1 + np.tanh(mz/Ls/24)
    else:
        z = np.arange(np.floor((mz-zstd)*10)/10, np.ceil((mz+zstd)*10)/10, 0.1)
        fzSM = 1 + np.tanh(z/Ls/24).mean()
    return fzSM


def get_stability_length(fctr, bw, a1, b1, theta_tau, ustar, wbf, chim=1, tail=True):
    """
    Compute the generalized Lobukhov, Stokes stability length and Hoenikker stability length.
    """
    def MP_minus_BF(z, fctr, bw, a1, b1, theta_tau, ustar, wbf, chim, tail):
        dUdz = ustar/kappa/(np.abs(z)+0.1)*chim
        dUsdz = np.maximum(get_dUsdwdz(z, fctr, bw, a1, b1, theta_tau, tail=tail)[0], 0)
        return dUsdz + dUdz - np.abs(wbf)/(ustar**2)
    
    def StP_minus_SP(z, fctr, bw, a1, b1, theta_tau, ustar, chim, tail):
        dUdz = ustar/kappa/(np.abs(z)+0.1)*chim
        dUsdz = np.maximum(get_dUsdwdz(z, fctr, bw, a1, b1, theta_tau, tail=tail)[0], 0)
        return dUsdz - dUdz
    
    def StP_minus_BF(z, fctr, bw, a1, b1, theta_tau, ustar, wbf, tail):
        dUsdz = np.maximum(get_dUsdwdz(z, fctr, bw, a1, b1, theta_tau, tail=tail)[0], 0)
        return dUsdz - np.abs(wbf)/(ustar**2)
    
    # z_test = -np.logspace(-3,1.5,100)
    # StSh_test = get_dUsdwdz(z_test, fctr, bw, a1, b1, theta_tau, tail=tail)[0]
    if np.all(~np.isnan([theta_tau, ustar, wbf])):
        root,_,ier,_ = optimize.fsolve(MP_minus_BF, -1e-5, 
                                       args=(fctr, bw, a1, b1, theta_tau, ustar, wbf, chim, tail), full_output=True)
        LOg = np.copysign(root[0], -wbf)
        if ier!=1:
            # print('LOg solution not found!') # when downwind Stokes shear is negative
            LOg = np.copysign(1e-6, -wbf)
        
        roots = np.array([])
        current_root = -1e-3
        for zguess in [-1e-3, -1, -10]:
            if zguess > current_root:
                continue
            x,_,ier,_ = optimize.fsolve(StP_minus_SP, zguess, 
                                        args=(fctr, bw, a1, b1, theta_tau, ustar, chim, tail), full_output=True)
            root = None
            if ier == 1:
                root = x[0]
            if root is None: # no root found
                continue
            current_root = root
            if all(abs(roots - root) > 1e-4):
                roots = np.append(roots, x)
        if roots.size==0:
            Lss = 1e-4
            # print('Stokes shear always smaller than Eulerian shear below 0.1 mm depth.')
        else:
            Lss = np.abs(np.min(roots))
                
        root,_,ier,_ = optimize.fsolve(StP_minus_BF, -1e-5, 
                                       args=(fctr, bw, a1, b1, theta_tau, ustar, wbf, tail), full_output=True)
        LHo = np.copysign(root[0], -wbf)
        if ier!=1:
            # print('Stokes production always smaller than buoyancy flux')
            LHo = np.copysign(1e-6, -wbf)
    else:
        LOg = nan
        Lss = nan
        LHo = nan
    return np.array([LOg, Lss, LHo])


def get_stability_length_d2f(fctr, us_d2f, vs_d2f, theta_tau, ustar, wbf, chim=1, tail=True):
    """
    Compute the generalized Lobukhov, Stokes stability length and Hoenikker stability length.
    """
    def MP_minus_BF_d2f(z, fctr, us_d2f, vs_d2f, theta_tau, ustar, wbf, chim, tail):
        dUdz = ustar/kappa/(np.abs(z)+0.1)*chim
        dUsdz = np.maximum(get_dUsdwdz_d2f(z, fctr, us_d2f, vs_d2f, theta_tau, tail=tail)[0], 0)
        return dUsdz + dUdz - np.abs(wbf)/(ustar**2)
    
    def StP_minus_SP_d2f(z, fctr, us_d2f, vs_d2f, theta_tau, ustar, chim, tail):
        dUdz = ustar/kappa/(np.abs(z)+0.1)*chim
        dUsdz = np.maximum(get_dUsdwdz_d2f(z, fctr, us_d2f, vs_d2f, theta_tau, tail=tail)[0], 0)
        return dUsdz - dUdz
    
    def StP_minus_BF_d2f(z, fctr, us_d2f, vs_d2f, theta_tau, ustar, wbf, tail):
        dUsdz = np.maximum(get_dUsdwdz_d2f(z, fctr, us_d2f, vs_d2f, theta_tau, tail=tail)[0], 0)
        return dUsdz - np.abs(wbf)/(ustar**2)
    
    if np.all(~np.isnan([theta_tau, ustar, wbf])) and np.isnan([us_d2f, vs_d2f]).sum()==0:
        root,_,ier,_ = optimize.fsolve(MP_minus_BF_d2f, -1e-5, 
                                       args=(fctr, us_d2f, vs_d2f, theta_tau, ustar, wbf, chim, tail), full_output=True)
        LOg = np.copysign(root[0], -wbf)
        if ier!=1:
            # print(f'i: {i}, wbf: {wbf}, LOg solution not found!') # when downwind Stokes shear is negative
            LOg = np.copysign(1e-6, -wbf)
        
        roots = np.array([])
        current_root = -1e-3
        for zguess in [-1e-3, -1, -10]:
            if zguess > current_root:
                continue
            x,_,ier,_ = optimize.fsolve(StP_minus_SP_d2f, zguess, 
                                        args=(fctr, us_d2f, vs_d2f, theta_tau, ustar, chim, tail), full_output=True)
            root = None
            if ier == 1:
                root = x[0]
            if root is None: # no root found
                continue
            current_root = root
            if all(abs(roots - root) > 1e-6):
                roots = np.append(roots, x)
        if roots.size==0:
            Lss = 1e-4
            # print('Stokes shear always smaller than Eulerian shear below 0.1 mm depth.')
        else:
            Lss = np.abs(np.min(roots))
        
        root,_,ier,_ = optimize.fsolve(StP_minus_BF_d2f, -1e-5, 
                                       args=(fctr, us_d2f, vs_d2f, theta_tau, ustar, wbf, tail), full_output=True)
        LHo = np.copysign(root[0], -wbf)
        if ier!=1:
            # print('Stokes production always smaller than buoyancy flux')
            LHo = np.copysign(1e-6, -wbf)
    else:
        LOg = nan
        Lss = nan
        LHo = nan
    return np.array([LOg, Lss, LHo])


def get_Ls(fctr, bw, a1, b1, ustar, bld, theta_tau, lopt='parabolic', dz=0.1, downwind=True, tail=True):
    """
    Compute the CL vortex forcing length scale in in-homogenenous pressure-strain rate closure (Harcourt 2015)
    """
    h = np.around(bld/dz)*dz
    nz = int(h/dz+1)
    z = np.linspace(-h-dz, 0, nz) - 1e-3 # avoid singularity for Stokes drift shear at tail
    if lopt=='parabolic':
        l = kappa*np.abs(z)*np.maximum(1+z/h, 0)
    elif lopt=='kpp':
        l = kappa*np.abs(z)*np.maximum(1+z/h, 0)**2
    else:
        raise ValueError(f'length scale option <{lopt}> not supprted!')
    stress = ustar**2*np.maximum(1+z/h, 0)
    Usdwdz = np.squeeze(get_dUsdwdz(z, fctr, bw, a1, b1, theta_tau, downwind=downwind, tail=tail)[0])
    PCL = stress*Usdwdz
    idx_nPCL = np.where(PCL<=0)[0]
    if idx_nPCL[-1]==(nz-1):
        # print('all CL production negative near surface')
        Ls = nan
    else:
        ipPCL = idx_nPCL[-1] + 1
        Ls = np.sum(PCL[ipPCL:]*l[ipPCL:]) / np.sum(PCL[ipPCL:])
    return Ls


def get_Ls_d2f(fctr, us_d2f, vs_d2f, ustar, bld, theta_tau, lopt='parabolic', dz=0.1, tail=True):
    """
    Compute the CL vortex forcing length scale in in-homogenenous pressure-strain rate closure (Harcourt 2015)
    """
    if np.isnan([us_d2f, vs_d2f]).sum()==0:
        h = np.around(bld/dz)*dz
        nz = int(h/dz+1)
        z = np.linspace(-h-dz, 0, nz)
        if lopt=='parabolic':
            l = kappa*np.abs(z)*np.maximum(1+z/h, 0)
        elif lopt=='kpp':
            l = kappa*np.abs(z)*np.maximum(1+z/h, 0)**2
        else:
            raise ValueError(f'length scale option <{lopt}> not supprted!')
        stress = ustar**2*np.maximum(1+z/h, 0)
        Usdw = np.squeeze(get_Usdw_d2f(z, fctr, us_d2f, vs_d2f, theta_tau, tail=tail)[0])
        PCL = stress*Usdw
        idx_nPCL = np.where(PCL<=0)[0]
        if idx_nPCL[-1]==(nz-1):
            # print('all CL production negative near surface')
            Ls = nan
        else:
            ipPCL = idx_nPCL[-1] + 1
            Ls = np.sum(PCL[ipPCL:]*l[ipPCL:]) / np.sum(PCL[ipPCL:])
    else:
        Ls = nan
    return Ls


def get_stds(fctr, bw, a1, b1, theta_wave, tail=True):
    """
    Compute the Stokes e-folding decay scale.
    """
    def St_minus_St0e(z, fctr, bw, a1, b1, theta_wave, tail):
        Us0 = get_Usdw(np.array([0]), fctr, bw, a1, b1, theta_wave, tail=tail)[0]
        return get_Usdw(z, fctr, bw, a1, b1, theta_wave, tail=tail)[0] - Us0/np.exp(1)
    
    z_test = -np.logspace(-3,1,100)
    StSh_test = get_dUsdwdz(z_test, fctr, bw, a1, b1, theta_wave, tail=tail)[0]
    if ~np.isnan(theta_wave) and np.sum(StSh_test<0) == 0:
        root,_,ier,_ = optimize.fsolve(St_minus_St0e, -1e-1, 
                                       args=(fctr, bw, a1, b1, theta_wave, tail), full_output=True)
        stds = -root[0]
        if ier!=1:
            print('stds solution not found!')
            stds = nan
    else:
        stds = nan
    return stds


def get_stds_d2f(fctr, us_d2f, vs_d2f, theta_wave, tail=True):
    """
    Compute the Stokes e-folding decay scale.
    """
    def St_minus_St0e_d2f(z, fctr, us_d2f, vs_d2f, theta_wave, tail):
        Us0 = get_Usdw_d2f(np.array([0]), fctr, us_d2f, vs_d2f, theta_wave, tail=tail)[0]
        return get_Usdw_d2f(z, fctr, us_d2f, vs_d2f, theta_wave, tail=tail)[0] - Us0/np.exp(1)
    
    z_test = -np.logspace(-3,1,100)
    StSh_test = get_dUsdwdz_d2f(z_test, fctr, us_d2f, vs_d2f, theta_wave, tail=tail)[0]
    if ~np.isnan(theta_wave) and np.sum(StSh_test<0) == 0 and np.isnan([us_d2f, vs_d2f]).sum()==0:
        root,_,ier,_ = optimize.fsolve(St_minus_St0e_d2f, -1e-1, 
                                       args=(fctr, us_d2f, vs_d2f, theta_wave, tail), full_output=True)
        stds = -root[0]
        if ier!=1:
            print('stds solution not found!')
            print(f'root: {stds}, theta_tau: {theta_tau}')
            stds = nan
    else:
        stds = nan
    return stds


def get_ust_param(fctr, bw, a1, b1, bld, theta_tau, tail=True):
    """
    Compute surface layer averaged Stokes drift and other relevant parameters.
    """
    dz = 0.25
    sld = np.around(bld/5*2)/2
    nz = int(sld/dz+1)
    ir_slad = int((nz-1)/2+1)
    ir_ssld = int((nz-1)/4+1)
    z = np.linspace(-sld, 0, nz)
    Usdw, Uscw = get_Usdw(z, fctr, bw, a1, b1, theta_tau, tail=tail)
    Usdw_SL  = Usdw.mean()
    Uscw_SL  = Uscw.mean()
    Usdw_SLa = Usdw[-ir_slad:].mean()
    Uscw_SLa = Uscw[-ir_slad:].mean()
    Usdw_SSL = Usdw[-ir_ssld:].mean()
    Uscw_SSL = Uscw[-ir_ssld:].mean()
    
    rref = 0.765
    Usdw_ref, Uscw_ref = get_Usdw(np.array(-bld*rref), fctr, bw, a1, b1, theta_tau, tail=tail)
    Usdw_20ph, Uscw_20ph = get_Usdw(np.array(-bld/5), fctr, bw, a1, b1, theta_tau, tail=tail)
    Usdw_10ph, Uscw_10ph = get_Usdw(np.array(-bld/10), fctr, bw, a1, b1, theta_tau, tail=tail)
    Usdw0, Uscw0 = get_Usdw(np.array(0), fctr, bw, a1, b1, theta_tau, tail=tail)
    return np.array([Usdw_SL, Uscw_SL, Usdw_ref, Uscw_ref, Usdw0, Uscw0, Usdw_SLa, Uscw_SLa, 
                     Usdw_10ph, Uscw_10ph, Usdw_20ph, Uscw_20ph, Usdw_SSL, Uscw_SSL])


def get_ust_param_d2f(fctr, us_d2f, vs_d2f, bld, theta_tau, tail=True):
    """
    Compute surface layer averaged Stokes drift and other relevant parameters from 2D wave spectrum
    """
    if np.isnan([us_d2f, vs_d2f]).sum()==0:
        dz = 0.25
        sld = np.around(bld/5*2)/2
        nz = int(sld/dz+1)
        ir_slad = int((nz-1)/2+1)
        z = np.linspace(-sld, 0, nz)
        Usdw, Uscw = get_Usdw_d2f(z, fctr, us_d2f, vs_d2f, theta_tau, tail=tail)
        Usdw_SL  = Usdw.mean()
        Uscw_SL  = Uscw.mean()
        Usdw_SLa = Usdw[-ir_slad:].mean()
        Uscw_SLa = Uscw[-ir_slad:].mean()
        
        rref = 0.765
        Usdw_ref, Uscw_ref = get_Usdw_d2f(np.array(-bld*rref), fctr, us_d2f, vs_d2f, theta_tau, tail=tail)
        Usdw_20ph, Uscw_20ph = get_Usdw_d2f(np.array(-bld/5), fctr, us_d2f, vs_d2f, theta_tau, tail=tail)
        Usdw_10ph, Uscw_10ph = get_Usdw_d2f(np.array(-bld/10), fctr, us_d2f, vs_d2f, theta_tau, tail=tail)
        Usdw0, Uscw0 = get_Usdw_d2f(np.array(0), fctr, us_d2f, vs_d2f, theta_tau, tail=tail)
        return np.array([Usdw_SL, Uscw_SL, Usdw_ref, Uscw_ref, Usdw0, Uscw0, Usdw_SLa, Uscw_SLa, 
                         Usdw_10ph, Uscw_10ph, Usdw_20ph, Uscw_20ph])
    else:
        return np.ones(12)*nan


def get_wbr_Jwt(Bnsw, z, h, Jwtype='I'):
    """
    Compute the turbulent buoyancy flux profile induced by shortwave radiation.
    """
    rs, mu = get_Jwt_param(Jwtype)
    wbr0 = -Bnsw*rs*( 1 - np.exp(z/mu[0]) + z/h*(1-np.exp(-h/mu[0])) )
    wbr1 = -Bnsw*(1-rs)*( 1 - np.exp(z/mu[1]) + z/h*(1-np.exp(-h/mu[1])) )
    return wbr0 + wbr1


def get_wbr_PAR(Bnsw, z, h, kdPAR):
    """
    Compute the turbulent buoyancy flux profile induced by shortwave radiation.
    """
    rs = 1 - np.maximum(0.695 - 5.7*kdPAR, 0.27)
    mu0 = 0.5
    mu1 = 1/kdPAR
    wbr0 = -Bnsw*rs*( 1 - np.exp(z/mu0) + z/h*(1-np.exp(-h/mu0)) )
    wbr1 = -Bnsw*(1-rs)*( 1 - np.exp(z/mu1) + z/h*(1-np.exp(-h/mu1)) )
    return wbr0 + wbr1


def get_BFs_Jwt(wb0, Bnsw, h, Jwtype='I'):
    """
    Compute the effective buoyancy forcing including shortwave radiation absorbed within the boundary layer,
    and the total turbulent buoyancy flux scale.
    Use two-band parameterization following Paulson & Simpson (1977)
    wb0: surface turbulent buoyancy flux
    Bnsw: buoyanyc flux equivalent to net shortwave radiation (g*alpha*nsw/rho0/cp)
    Jwtype: Jerlov water type
    """
    rs, mu = get_Jwt_param(Jwtype)
    Bnswd = Bnsw*(rs*np.exp(-h/mu[0]) + (1-rs)*np.exp(-h/mu[1]))
    Bswr = Bnsw - Bnswd # absorbed portion
    Bf = -wb0 + Bswr # effective buoyancy forcing, positive into the ocean
    
    wbr0SL = -Bnsw*rs*( -5*mu[0]/h*(1-np.exp(-h/5/mu[0])) + 1 - (1-np.exp(-h/mu[0]))/10 )
    wbr1SL = -Bnsw*(1-rs)*( -5*mu[1]/h*(1-np.exp(-h/5/mu[1])) + 1 - (1-np.exp(-h/mu[1]))/10 )
    c = 0.9
    wbf = wb0 + (wbr0SL + wbr1SL)/c
    return Bf, wbf


def get_BFs_PAR(wb0, Bnsw, h, kdPAR):
    """
    Compute effective buoyancy flux including shortwave radiation absorbed within the boundary layer,
    and the total turbulent buoyancy flux scale.
    Use k_dPAR parameterization in HYCOM following Kara et al. 2005
    wb0: surface turbulent buoyancy flux
    Bnsw: buoyanyc flux equivalent to net shortwave radiation (g*alpha*nsw/rho0/cp)
    kdPAR: downwelling diffuse attenuation coefficient for the Photosynthetically Available Radiation
    """
    rs = 1 - np.maximum(0.695 - 5.7*kdPAR, 0.27)
    mu0 = 0.5
    mu1 = 1/kdPAR
    Bnswd = Bnsw*(rs*np.exp(-h/mu0) + (1-rs)*np.exp(-h/mu1))
    Bswr = Bnsw - Bnswd # absorbed portion
    Bf = -wb0 + Bswr # effective buoyancy forcing, positive into the ocean
    
    wbr0SL = -Bnsw*rs*( -5*mu0/h*(1-np.exp(-h/5/mu0)) + 1 - (1-np.exp(-h/mu0))/10 )
    wbr1SL = -Bnsw*(1-rs)*( -5*mu1/h*(1-np.exp(-h/5/mu1)) + 1 - (1-np.exp(-h/mu1))/10 )
    c = 0.9
    wbf = wb0 + (wbr0SL + wbr1SL)/c
    return Bf, wbf


def get_Jwt_param(Jwtype):
    """
    Compute the two-band solar radiation decay parameters.
    """
    if Jwtype=='I':
        rs = 0.58
        mu = [0.35, 23]
    elif Jwtype=='IA':
        rs = 0.62
        mu = [0.6, 20]
    elif Jwtype=='IB':
        rs = 0.67
        mu = [1, 17]
    elif Jwtype=='II':
        rs = 0.77
        mu = [1.5, 14]
    elif Jwtype=='III':
        rs = 0.78
        mu = [1.4, 7.9]
    else:
        print('water type not supported.')   
    return rs, mu


def get_kdPAR(Chl):
    """
    Compute kdPAR according to near-surface Chlorophyll concentration [mg/m3, or ug/L]
    Use relationships in Morel et al. 2007, Eq. 8 and 9'
    """
    kd490 = 0.0166 + 0.0773*Chl**(0.6715)
    return 0.0665 + 0.874*kd490 - 0.00121/kd490


def loglog_hist2d(xdata, ydata, xpr, ypr, bins=100, weights=None, **kwargs):
    xdata = xrpd2np(xdata).ravel()
    ydata = xrpd2np(ydata).ravel()
    if weights is not None:
        weights = xrpd2np(weights).ravel()
    hist, xi, yi = np.histogram2d(xdata, ydata, bins=bins, range=[xpr, ypr], weights=weights, **kwargs)
    # get the centers from the edges
    xc = (xi[1:]+xi[:-1])/2
    yc = (yi[1:]+yi[:-1])/2
    # convert back to actual values
    xc = 10**xc
    yc = 10**yc
    dx = np.diff(10**xi)
    dy = np.diff(10**yi)
    return hist, xc, yc, dx, dy


def hist2d(xdata, ydata, xr=None, yr=None, bins=100, weights=None, **kwargs):
    xdata = xrpd2np(xdata).ravel()
    ydata = xrpd2np(ydata).ravel()
    if weights is not None:
        weights = xrpd2np(weights).ravel()
    hist, xi, yi = np.histogram2d(xdata, ydata, bins=bins, range=[xr, yr], weights=weights, **kwargs)
    # get the centers from the edges
    xc = (xi[1:]+xi[:-1])/2
    yc = (yi[1:]+yi[:-1])/2
    return hist, xc, yc


def round_away_from_0(arr):
    return np.copysign(np.ceil(np.abs(arr)), arr)