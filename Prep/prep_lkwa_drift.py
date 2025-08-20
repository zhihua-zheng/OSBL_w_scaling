import xarray as xr
import numpy as np
import pandas as pd
import sca_osbl_tool as sot
import matplotlib as mpl
import gsw
from tqdm.auto import tqdm
from scipy import stats
from sys import platform
from constants import nan, kappa, pi


def select_vars(ds, vars=['zsbl','drift']):
    return ds[vars].squeeze(['lon','lat'], drop=True)


def first_turb(P):
    df = pd.DataFrame(data=P, columns=['P'])
    Prms = df.P.rolling(window=360, min_periods=180, center=True, win_type='hamming').std() # dt = 30s
    iturb = np.where(Prms>=1)[0]
    if iturb.size==0:
        first_iturb = len(P)
    else:
        first_iturb = iturb[0]
    return first_iturb


def count_midx(x):
    """
    Count number of times crossing the middle value
    """
    x = sot.butter_lpfilt(x, 1/3600/1.5, 1/30)
    xlow = np.percentile(x, 5)
    xhigh = np.percentile(x, 95)
    xmxmd = x - (xlow + xhigh)/2
    return ((xmxmd[:-1] * xmxmd[1:]) < 0).sum()


def trim_traj(z):
    """
    Trim the float trajectory so it starts and ends at about the same depth,
    by minimizing the fraction of data removed.
    """
    zint = np.around(z)
    nz = len(zint)
    zarray = np.unique(zint)
    rm_count = np.full(zarray.shape, nan) # number of points to remove
    for i,m in enumerate(zarray):
        idx1, idx2 = np.where(zint==m)[0][[0, -1]]
        if idx1 != idx2: # more than 1 occurance
            # reverse idx2 to count from the end
            rm_count[i] = idx1 + abs(idx2+1-nz)
    zopt = zarray[np.argmin(rm_count)]
    rmp = np.min(rm_count)/nz
    idx1, idx2 = np.where(zint==zopt)[0][[0, -1]]
    return slice(idx1, idx2+1), rmp


def w_from_drift(z, yd, floatID, Lfloat, dt=30, dftn=None):
    """
    Compute vertical velocity from pressure change during a drift
    """
    dts = yd*24*60*(60/dt) # in unit of dt
    # even spacing time array
    yd_even = np.arange(np.ceil(dts[0]), np.floor(dts[-1])+1)/(60/dt)/60/24
    df = pd.DataFrame(np.column_stack((yd, z)), columns=['yd', 'z']).set_index('yd')
    df = df.loc[~df.index.duplicated(), :] # remove duplicated time points
    dfi = df.reindex(df.index.union(yd_even)) \
            .interpolate(method='pchip', limit_direction='both', limit_area='inside').reindex(yd_even)

    # trimmed = trim_traj(dfi.z)
    # dfi = dfi.iloc[trimmed]
    dfi['w'] = np.convolve(dfi.z, [1, 0, -1], 'same')/(2*dt)
    dfi = dfi.iloc[1:-1,:] # invalid results at both ends
    if dt==1:
        # filter w
        wmedf = dfi.w.rolling(5, center=True).median() # edges are NaN
        dfi['wf'] = sot.butter_lpfilt(wmedf, 0.1/2/pi, 1)
        dfi = dfi[dfi.wf.notnull()] # drop NaN at egdes
        
        dfi['wf2'] = dfi.wf**2
        dfi['wf3'] = dfi.wf**3
        wrms2 = dfi.wf2.mean()
        dfi['wf2bulk'] = wrms2
        
        # layer-averaged dissipation rate from Lagrangian spectrum
        omega,Pww = sot.get_psd(dfi.wf, fs=1/dt)
        Paa = omega**2*Pww
        Pnoise = 0.001 # better moise model?
        site = 'LKWA'+str(floatID)
        epsb, omega0, omegaM = sot.fit_spec(omega, Paa, Pnoise, Lfloat, show_fig=True, site=site, dftn=dftn)
        omegaL = (epsb/(Lfloat**2))**(1/3)
        nu = 1.14e-6 # kinematic viscosity [m^2/s], from [https://www.omnicalculator.com/physics/water-viscosity]
        omega_Komogrov = 2*pi/np.sqrt(nu/epsb)
        omega_i = np.logspace(-4, np.around(np.log10(omega_Komogrov),1), 100)
        PaaM_i = sot.modelaccspL(epsb, omega0, omega_i, Lfloat=Lfloat)
        PaaM0_i = sot.modelaccspL(epsb, omega0, omega_i, Lfloat=1e-10)
        delw2 = np.trapz(PaaM0_i*omega_i**(-2),omega_i) - np.trapz(PaaM_i*omega_i**(-2),omega_i)
        omega_c = np.logspace(-4, 0, 80)
        PaaM_c = sot.modelaccspL(epsb, omega0, omega_c, Lfloat=Lfloat)
        Paa_c = np.interp(omega_c, omega, Paa, left=nan, right=nan)
        Paa_c[omega_c >= omegaM] = nan
        dfi['epsbulk'] = epsb
        dfi['delw2'] = delw2
        speci = pd.DataFrame(data=dict(raw=Paa_c, fit=PaaM_c, eps=epsb, omega0=omega0, omegaL=omegaL, myd=df.index.values.mean(),
                                       wrms2=wrms2, delw2=delw2, omega=omega_c)).set_index('omega')
    # max dt in fast pressure drift data is ~ 3s, it's ok to interpolate evenly spaced time vector 
    # skip the re-interpolation back to original sampling time
    return dfi, speci


def bin_w(df, year):
    """
    Bin average vertical velocities in unifom time and z grid
    """
    # set bin edges
    time_start, time_end = df.time.iloc[[0, -1]].round('60min')
    hhtime = pd.Timedelta(minutes=30)
    time_bins = pd.date_range(time_start-hhtime, time_end+hhtime, freq='H')
    zbt = np.floor(df.z.min())
    z_bins = np.arange(zbt, 1)
    
    # bin data
    grpd = df.groupby([pd.cut(df.time, time_bins), pd.cut(df.z, z_bins)])
    grpm = grpd.mean()
    grpc = grpd.w.count().replace(0, nan).rename('n')
    grp_zstd = grpd.z.std(ddof=0).rename('zstd')
    bnd = pd.concat([grpm, grpc, grp_zstd], axis=1).rename(columns={'z': 'mz', 'yd': 'myd'}).reset_index()
    bnd.time = bnd.time.apply(lambda x: x.mid)
    bnd.z = bnd.z.apply(lambda x: x.mid)
    
    # to xarray dataset
    bnd_ds = bnd.set_index(['time', 'z']).to_xarray()
    bnd_ds['myd'] = bnd_ds.myd.mean(dim='z')
    bnd_ds['drift'] = bnd_ds.drift.mean(dim='z')
    bnd_ds['epsbulk'] = bnd_ds.epsbulk.mean(dim='z')
    bnd_ds['wf2bulk'] = bnd_ds.wf2bulk.mean(dim='z')
    # bnd_ds = bnd_ds.dropna('time', how='all') # remove empty time intervals between drifts
    return bnd_ds


def _prep_lkwa_drift(GDrive_root, floatID):
    site_root = GDrive_root + 'UW/Research/Data/LakeWA/'
    nbf_dir = site_root + 'NBF/'
    flux_dir = site_root + 'Mooring/'
    simfiles = nbf_dir + 'Sim/'+str(floatID)+'/*.nc'
    year = 2011
    Lfloat = 0.92/2

    # original float data
    with xr.open_dataset(nbf_dir+'LKWA_env_'+str(floatID)+'.nc') as E:
        drift0 = E.drift.values
        time_env = E.time.values
        yd = E.yd.values
        ydP = E.ydP.values
        Pc = E.Pc.values
        P = E.P.values
        B = E.B.values
        lat = E.lat.reset_coords('yd', drop=True)
        lon = E.lon.reset_coords('yd', drop=True)
    mlat = np.mean(lat.values)
    Zc = gsw.z_from_p(Pc, mlat)

    # float buoyancy
    # with xr.open_dataset(nbf_dir+'OCSP_fb_'+str(year)+'.nc') as Fb:
    #     b = Fb.b.values

    # KPP BLD
    with xr.open_mfdataset(simfiles, combine='by_coords', preprocess=select_vars,
                           data_vars='minimal', coords='minimal', compat='override') as ds:
        time_gotm = pd.date_range(start=ds.time[0].values, end=ds.time[-1].values, freq='5min')
        zsbl = ds.zsbl.interp(time=time_gotm).load()
    zsbl = zsbl.assign_coords(yd=('time', sot.pytime2yd(zsbl.time)))
    zsbl = xr.apply_ufunc(sot.butter_lpfilt, zsbl, input_core_dims=[['time']],
                          output_core_dims=[['time']], kwargs={'cutoff': 1/60/12, 'fs': 1/5})
    bld = -zsbl.resample(time='60min', base=30, loffset=pd.Timedelta(minutes=30)).mean().rename('bld')
    bld[bld<1e-2] = 1e-2
    nc_name = 'LKWA_bld_'+str(floatID)+'.nc'
    bld.to_netcdf(nbf_dir + nc_name, engine='netcdf4')
    print(f'Processed boundary layer depth data: {nc_name} saved at {nbf_dir}.')
    
    # surface forcing
    with xr.open_dataset(flux_dir+'LKWA_fluxes.nc') as F: 
        ustar = F.ustar.squeeze(drop=True).reset_coords('yd', drop=True).sel(time=bld.time)
        wbnet = F.wbnet.squeeze(drop=True).reset_coords('yd', drop=True).sel(time=bld.time)
        wb0 = F.wb0.squeeze(drop=True).reset_coords('yd', drop=True).sel(time=bld.time)
        Chl = F.Chl.squeeze(drop=True).reset_coords('yd', drop=True).sel(time=bld.time)
        Bnsw = -(wbnet-wb0)
        kdPAR = sot.get_kdPAR(Chl)
        Bf, wbf = sot.get_BFs_PAR(wb0, Bnsw, bld, kdPAR)
        theta_tau = np.arctan2(F.tauy, F.taux).squeeze(drop=True).reset_coords('yd', drop=True)
    
    wstar = np.copysign((np.abs(wbf)*bld)**(1/3), wbf)
    wfs2 = 1.1*ustar**2 + np.copysign(0.3*wstar**2, wbf)
    wfs2[wfs2<1e-10] = 1e-10
    wfs = np.sqrt(wfs2)
    LObukhov = -ustar**3/kappa/wbf
    
    # drift selection
    _,Iclump = sot.get_clumps(drift0, min_len=360) # at least 3-hours long, 3*3600/30=360
    # identify the beginning of drift: when the rate of change of float buoyancy control B is less than 0.1 cc/s
    i_drift_on = [np.where(np.abs(np.gradient(B[I], yd[I]*24*3600)) > 0.1)[0][-1] + 1 for I in Iclump]
    Idrift = [sot.re_slice(I,i) for I,i in zip(Iclump, i_drift_on)]
    # require drift spend more than 70% time in the boundary layer, and trim it to start and end in the boundary layer
    bld_env = bld.interp(time=time_env)
    Idrift = [I for I in Idrift if np.sum(P[I]>bld_env[I]) < (I.stop-I.start)*0.3]
    Idrift = [sot.re_slice(I, np.where(P[I]<bld_env[I])[0][0], np.where(P[I]<bld_env[I])[0][-1] - len(P[I])) for I in Idrift]
    # Idrift = [I for I in Idrift if count_midx(P[I])>=3]
    
    i_turb_on = [first_turb(P[I]) for I in Idrift]
    Idrift = [sot.re_slice(I,i) for I,i in zip(Idrift, i_turb_on)]
    
    skewP = np.around(np.array([stats.skew(P[I]) for I in Idrift]),1)
    Idrift = [I for I,SK in zip(Idrift,skewP) if abs(SK) < 1]
    Idrift = [I for I in Idrift if (I.stop-I.start)>=360]
    
    # compute w
    drift, dfs, specs = ([] for i in range(3))
    mpl.use('Agg') # change backend to avoid memory leak when doing batches of plotting
    mpl.pyplot.ioff()
    for i in tqdm(Idrift, desc='Compute w for each drift'):
        dftn = int(np.unique(drift0[i])[0])
        drift.append(dftn)
        idf = (ydP >= yd[i.start]) & (ydP <= yd[i.stop-1])
        df, spec = w_from_drift(Zc[idf], ydP[idf], floatID, Lfloat, dt=1, dftn=dftn)
        if np.unique(spec.omegaL/spec.omega0) <= 1: # don't use drift data when the spectral fit is bad
            continue
        if df.wf.isnull().sum() != df.w.isnull().sum():
            print('Filtered w has NaN')
            break
        dfs.append(df)
        specs.append(spec)
    mpl.pyplot.ion()
    mpl.use('module://ipympl.backend_nbagg')
    
    # combine data from each drift
    specs = pd.concat(specs, keys=drift, names=['drift', 'omega']).to_xarray().transpose('omega', 'drift')
    specs.attrs['Lfloat'] = Lfloat
    varlist = ['eps', 'omega0', 'omegaL', 'myd', 'wrms2', 'delw2']
    for var in varlist:
        specs.update({var: specs[var].mean('omega')})
    dfs = pd.concat(dfs, keys=drift, names=['drift', 'yd']).reset_index()
    dfs.insert(0, 'time', sot.yd2pytime(dfs.yd, year))
    dft = dfs.set_index('time').to_xarray()
    dft['ustar'] = np.sqrt((ustar**2).interp(time=dft.time))
    dft['wbf'] = wbf.interp(time=dft.time)
    dft['Bf'] = Bf.interp(time=dft.time)
    dft['bld'] = bld.interp(time=dft.time)
    dft['lat'] = lat.interp(time=dft.time)
    dft['lon'] = lon.interp(time=dft.time)
    
    # average data in 1hr-1m bin, used for scaling
    dftb = bin_w(dfs, year)
    dftb['theta_tau'] = theta_tau.sel(time=dftb.time)
    dftb['ustar'] = ustar.sel(time=dftb.time)
    dftb['wstar'] = wstar.sel(time=dftb.time)
    dftb['Bf'] = Bf.sel(time=dftb.time)
    dftb['wbf'] = wbf.sel(time=dftb.time)
    dftb['wfs'] = wfs.sel(time=dftb.time)
    dftb['LObukhov'] = LObukhov.sel(time=dftb.time)
    dftb['zeta'] = np.abs(dftb.mz)/dftb.LObukhov
    dftb['bld'] = bld.sel(time=dftb.time)
    dftb['zoh'] = dftb.mz/dftb.bld
    dftb['wf2our2'] = dftb.wf2/dftb.ustar**2
    dftb['wf2owr2'] = dftb.wf2/dftb.wstar**2
    
    oneozM = xr.apply_ufunc(sot.get_oneozM, dftb.mz, dftb.zstd, 
                            input_core_dims=[[], []], vectorize=True, output_dtypes=[float])
    dftb['ESP'] = dftb.ustar**3*(1+dftb.zoh)/kappa*oneozM
    dftb['BF'] = wb0.sel(time=dftb.time)*(1+dftb.zoh) + \
                 sot.get_wbr_PAR(Bnsw.sel(time=dftb.time), dftb.mz, dftb.bld, kdPAR.sel(time=dftb.time))
    dftb['BFsl'] = dftb.wbf*(1+dftb.zoh)
    
    cfac = sot.crt_wrms_bulk(dftb.epsbulk, Lfloat, dftb.wf2bulk)
    dftb['cfac2'] = cfac**2
    cfac,_ = xr.broadcast(cfac, dftb.z)
    dftb['cfacz2'] = (('time', 'z'), sot.crt_wrms_prof(cfac, dftb.zoh, cshape='linear')**2)

    # timescale for BLD variation
    dbld_dt = bld.differentiate('time', datetime_unit='h')
    Tbld = (bld/np.abs(dbld_dt)).interp(time=dftb.time, method='nearest') # [hr]
    # timescale for overturning
    Tot = (2*bld/wfs/3600).interp(time=dftb.time, method='nearest') # [hr]
    DOF = 10
    # categorize boundary layer stationarity 
    dftb['Iequil'] = Tbld > DOF*Tot
    dftb['Ishoal'] = (Tbld <= DOF*Tot) & (dbld_dt.interp(time=dftb.time, method='nearest') < 0)
    dftb['Ideepen'] = (Tbld <= DOF*Tot) & (dbld_dt.interp(time=dftb.time, method='nearest') > 0)

    # # mixing layer depth from pressure
    # hmix = sot.get_hmix(yd, -z, ustar, wbf)
    # attrs={'description': 'Processed drift data from the Lagrangian float deployed in '+str(year)+' near ocean climate station Papa'})
    
    nc_name = 'LKWA_drifts_'+str(floatID)+'.nc'
    dft.to_netcdf(nbf_dir + nc_name, engine='netcdf4')
    print(f'Processed float drift data: {nc_name} saved at {nbf_dir}.')
    
    nc_name = 'LKWA_drifts_binned_'+str(floatID)+'.nc'
    dftb.to_netcdf(nbf_dir + nc_name, engine='netcdf4')
    print(f'Binned float drift data: {nc_name} saved at {nbf_dir}.')
    
    
    nc_name = 'LKWA_drifts_spec_'+str(floatID)+'.nc'
    specs.to_netcdf(nbf_dir + nc_name, engine='netcdf4')
    print(f'Float drift spectra: {nc_name} saved at {nbf_dir}.\n')


if platform == 'linux':
    Groot = '/media/zhihua/internal1/GDrive/'
elif platform == 'darwin':
    Groot = '/Volumes/GoogleDrive/My Drive/'
_prep_lkwa_drift(Groot, 71)
_prep_lkwa_drift(Groot, 72)