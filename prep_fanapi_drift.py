import xarray as xr
import numpy as np
import pandas as pd
import sca_osbl_tool as sot
import matplotlib as mpl
import gsw
from tqdm.auto import tqdm
from scipy import stats
from sys import platform
from constants import nan, kappa, pi, rho0


def w_from_drift(z, yd, floatID, Lfloat, dt=30, dftn=None):
    """
    Compute vertical velocity from pressure change during a drift
    """
    dts = yd*24*60*(60/dt) # in unit of dt
    # even spacing time array
    yd_even = np.arange(np.ceil(dts[0]), np.floor(dts[-1])+1)/(60/dt)/60/24
    df = pd.DataFrame(np.column_stack((yd, z)), columns=['yd', 'z']).set_index('yd')
    df = df.loc[~df.index.duplicated(), :] # remove duplicated time points
    # igap = np.where(np.diff(df.index)*24*3600 > 10)[0][0] + 1 # first gap longer than 10s
    # df = df.iloc[:igap,:]
    dfi = df.reindex(df.index.union(yd_even)) \
            .interpolate(method='pchip', limit_direction='both', limit_area='inside').reindex(yd_even)
    
    dfi['w'] = np.convolve(dfi.z, [1, 0, -1], 'same')/(2*dt)
    dfi = dfi.iloc[1:-1,:] # invalid results at both ends, right exclusive
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
        _,Pww_raw = sot.get_psd(dfi.w,  fs=1/dt)
        omega,Pww = sot.get_psd(dfi.wf, fs=1/dt)
        Paa     = omega**2*Pww
        Paa_raw = omega**2*Pww_raw
        Pnoise = 0.001 # better moise model?
        site = 'Fanapi'+str(floatID)
        epsb, omega0, omegaM = sot.fit_spec(omega, Paa, Pnoise, Lfloat, show_fig=True, site=site, dftn=dftn)
        omegaL = (epsb/(Lfloat**2))**(1/3)
        nu = 1.19e-6 # kinematic viscosity [m^2/s], from The Engineering Toolbox [T=15, S=35]
        omega_Komogrov = 2*pi/np.sqrt(nu/epsb)
        omega_i = np.logspace(-4, np.around(np.log10(omega_Komogrov),1), 100)
        PaaM_i = sot.modelaccspL(epsb, omega0, omega_i, Lfloat=Lfloat)
        PaaM0_i = sot.modelaccspL(epsb, omega0, omega_i, Lfloat=1e-10)
        delw2 = np.trapz(PaaM0_i*omega_i**(-2),omega_i) - np.trapz(PaaM_i*omega_i**(-2),omega_i)
        omega_c = np.logspace(-4, 0.5, 100)
        PaaM_c = sot.modelaccspL(epsb, omega0, omega_c, Lfloat=Lfloat)
        Paa_c = np.interp(omega_c, omega, Paa, left=nan, right=nan)
        Paa_raw_c = np.interp(omega_c, omega, Paa_raw, left=nan, right=nan)
        # Paa_c[omega_c >= omegaM] = nan
        dfi['epsbulk'] = epsb
        dfi['delw2'] = delw2
        speci = pd.DataFrame(data=dict(raw=Paa_raw_c, fil=Paa_c, fit=PaaM_c, eps=epsb, omega0=omega0, omegaL=omegaL, omegaM=omegaM,
                                       myd=df.index.values.mean(), wrms2=wrms2, delw2=delw2, omega=omega_c)).set_index('omega')
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
    bnd_ds['bld'] = bnd_ds.bld.mean(dim='z')
    bnd_ds['drift'] = bnd_ds.drift.mean(dim='z')
    bnd_ds['epsbulk'] = bnd_ds.epsbulk.mean(dim='z')
    bnd_ds['wf2bulk'] = bnd_ds.wf2bulk.mean(dim='z')
    # bnd_ds = bnd_ds.dropna('time', how='all') # remove empty time intervals between drifts
    return bnd_ds


def _prep_fanapi_drift(GDrive_root, floatID, tau_option='URI'):
    site_root = GDrive_root + 'Data/Hurricanes/ITOP/'
    met_dir = site_root + 'Met/'
    nbf_dir = site_root + 'NBF/'
    env_name = 'Fanapi_env_'+str(floatID)+'.nc'
    year = 2010
    Lfloat = 0.92/2
    
    # forcing
    F = xr.open_dataset(met_dir+f'Fanapi_fluxes_{floatID:02d}_{tau_option}.nc').drop_vars(['lon','lat'])
    
    # original float data
    with xr.open_dataset(nbf_dir+env_name) as E:
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
    
    # drift selection
    _,Iclump = sot.get_clumps(drift0, min_len=360) # at least 3-hours long, 3*3600/30=360
    # index for the beginning of drift: when the rate of change of float buoyancy is less than 0.02 cc/s
    i_drift_on = [np.where( np.abs(np.gradient(B[I], yd[I]*24*3600)) < 0.02 )[0][0] for I in Iclump]
    Idrift = [sot.re_slice(I,i) for I,i in zip(Iclump, i_drift_on)]
    
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
        idf = ((ydP >= yd[i.start]) & (ydP <= yd[i.stop-1])) & (ydP <= F.yd.values[-1])
        if floatID == 61:
            yd0 = sot.pytime2yd(np.datetime64('2010-09-17 19:10:00'))
            idf = idf & (ydP >= yd0) # float 61 before yd0 drifted below mixed layer
        df, spec = w_from_drift(Zc[idf], ydP[idf], floatID, Lfloat, dt=1, dftn=dftn)
        # mixing layere depth
        df.insert(0, 'time', sot.yd2pytime(df.index, year))
        hmix = sot.get_hmix(df.index, -df.z, np.sqrt(F.taum_GFDL.interp(time=df.time)/rho0),
                            F.wbnet.interp(time=df.time), 1, DOF=20)
        df['bld'] = sot.butter_lpfilt(hmix, 1/3600/6, 1) # 6-hr lowpass
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
    varlist = ['eps', 'omega0', 'omegaL', 'omegaM', 'myd', 'wrms2', 'delw2']
    for var in varlist:
        specs.update({var: specs[var].mean('omega')})
    dfs = pd.concat(dfs, keys=drift, names=['drift', 'yd']).reset_index()
    dft = dfs.set_index('time').to_xarray()
    dft['ustar'] = F.ustar.reset_coords('yd', drop=True).interp(time=dft.time)
    wb0 = F.wb0.reset_coords('yd', drop=True).interp(time=dft.time)
    wbnet = F.wbnet.reset_coords('yd', drop=True).interp(time=dft.time)
    Bnsw = -(wbnet-wb0)
    dft['Bf'], dft['wbf'] = sot.get_BFs_Jwt(wb0, Bnsw, dft.bld, Jwtype='I')
    dft['lat'] = lat.interp(time=dft.time)
    dft['lon'] = lon.interp(time=dft.time)
    
    # average data in 1hr-1m bin
    Fhf = F.where(F.ihfront, drop=True) # time range with high wind and before storm arrives
    dfs = dfs.loc[(dfs.time >= Fhf.time.values[0]) & (dfs.time <= Fhf.time.values[-1])]
    dftb = bin_w(dfs, year)
    dftb['mtime'] = ('time', sot.yd2pytime(dftb.myd, year))
    
    hhr = pd.Timedelta(minutes=30)
    taux = F.taux.resample(time='60min', base=30, loffset=hhr).mean().sel(time=dftb.time)
    tauy = F.tauy.resample(time='60min', base=30, loffset=hhr).mean().sel(time=dftb.time)
    ustar = np.sqrt( np.sqrt(taux**2 + tauy**2)/rho0 )
    theta_tau = np.arctan2(tauy, taux)
    
    wbf = dft.wbf.resample(time='60min', base=30, loffset=hhr).mean().sel(time=dftb.time)
    wstar = np.copysign((np.abs(wbf)*dftb.bld)**(1/3), wbf)
    wfs2 = 1.1*(ustar**2) + np.copysign(0.3*wstar**2, wbf)
    wfs2[wfs2<1e-10] = 1e-10
    dftb['lat'] = lat.interp(time=dftb.time)
    dftb['lon'] = lon.interp(time=dftb.time)
    dftb['theta_tau'] = theta_tau
    dftb['ustar'] = ustar
    dftb['wstar'] = wstar
    dftb['Bf'] = dft.Bf.resample(time='60min', base=30, loffset=hhr).mean().sel(time=dftb.time)
    dftb['wbf'] = wbf
    dftb['wfs'] = np.sqrt(wfs2)
    dftb['LObukhov'] = -ustar**3/kappa/wbf
    dftb['zeta'] = np.abs(dftb.mz)/dftb.LObukhov
    dftb['zoh'] = dftb.mz/dftb.bld
    dftb['wf2our2'] = dftb.wf2/dftb.ustar**2
    dftb['wf2owr2'] = dftb.wf2/dftb.wstar**2
    
    oneozM = xr.apply_ufunc(sot.get_oneozM, dftb.mz, dftb.zstd, 
                            input_core_dims=[[], []], vectorize=True, output_dtypes=[float])
    dftb['ESP'] = ustar**3*(1+dftb.zoh)/kappa*oneozM
    dftb['BF'] = wb0.resample(time='60min', base=30, loffset=hhr).mean().sel(time=dftb.time)*(1+dftb.zoh) + \
                 sot.get_wbr_Jwt(Bnsw.resample(time='60min', base=30, loffset=hhr).mean().sel(time=dftb.time), 
                                 dftb.mz, dftb.bld, Jwtype='I')
    dftb['BFsl'] = dftb.wbf*(1+dftb.zoh)
    
    cfac = sot.crt_wrms_bulk(dftb.epsbulk, Lfloat, dftb.wf2bulk)
    dftb['cfac2'] = cfac**2
    cfac,_ = xr.broadcast(cfac, dftb.z)
    dftb['cfacz2'] = (('time', 'z'), sot.crt_wrms_prof(cfac, dftb.zoh, cshape='linear')**2)
    
    # identify quadrant
    fdx = (dftb.lon-F.trk_lon.interp(time=dftb.time))*F.lon2km
    fdy = (dftb.lat-F.trk_lat.interp(time=dftb.time))*F.lat2km
    fdist = np.sqrt(fdx**2 + fdy**2) # [km]
    fdir = np.arctan2(fdy, fdx)*180/pi
    frdir = fdir - F.uhdir.interp(time=dftb.time) # float direction relative to Fanapi translation
    frdir[frdir < -180] = frdir[frdir < -180] + 360
    frdir[frdir >  180] = frdir[frdir >  180] - 360
    # identify if float was under the storm
    in_storm = fdist < F.mrmw # max wind radius
    dftb['ifl'] = frdir > 10 # theta_s = 10 as the boundary between sectors in Fig. 2c of Hsu et al. (2019)
    dftb['ifr'] = (frdir <= 10 | in_storm)
    
    # timescale for BLD variation
    dbld_dt = dftb.bld.differentiate('time', datetime_unit='h')
    Tbld = dftb.bld/np.abs(dbld_dt) # [hr]
    # timescale for overturning
    Tot = 2*dftb.bld/dftb.wfs/3600 # [hr]
    DOF = 10
    # categorize boundary layer stationarity 
    dftb['Iequil'] = Tbld > DOF*Tot
    dftb['Ishoal'] = (Tbld <= DOF*Tot) & (dbld_dt < 0)
    dftb['Ideepen'] = (Tbld <= DOF*Tot) & (dbld_dt > 0)
    
    nc_name = f'Fanapi_drifts_{floatID}_{tau_option}.nc'
    dft.to_netcdf(nbf_dir + nc_name, engine='netcdf4')
    print(f'Processed float drift data: {nc_name} saved at {nbf_dir}.')
    
    nc_name = f'Fanapi_drifts_binned_{floatID}_{tau_option}.nc'
    dftb.to_netcdf(nbf_dir + nc_name, engine='netcdf4')
    print(f'Binned float drift data: {nc_name} saved at {nbf_dir}.')
    
    if tau_option == 'APL':
        nc_name = f'Fanapi_drifts_spec_{floatID}.nc'
        specs.to_netcdf(nbf_dir + nc_name, engine='netcdf4')
        print(f'Float drift spectra: {nc_name} saved at {nbf_dir}.\n')


if platform == 'linux':
    Groot = '/media/zhihua/internal1/GDrive/'
elif platform == 'darwin':
    Groot = '/Users/zhihua/Documents/Work/Research/Projects/PhD-projects/'

opt = 'APL'
_prep_fanapi_drift(Groot, 60, tau_option=opt)
_prep_fanapi_drift(Groot, 61, tau_option=opt)
_prep_fanapi_drift(Groot, 62, tau_option=opt)
_prep_fanapi_drift(Groot, 64, tau_option=opt)

# opt = 'URI'
# _prep_fanapi_drift(Groot, 60, tau_option=opt)
# _prep_fanapi_drift(Groot, 61, tau_option=opt)
# _prep_fanapi_drift(Groot, 62, tau_option=opt)
# _prep_fanapi_drift(Groot, 64, tau_option=opt)
