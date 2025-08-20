import numpy as np
import pandas as pd
import xarray as xr
import gsw
import sca_osbl_tool as sot
from sys import platform
from scipy import signal, stats
from constants import kappa, nan, rho0, pi


def accspecL(g):
    omega_c = np.logspace(-5, -1, 80)
    if g.w.size < (1024/2):
        eps, omega0, omegaL, delw2, w2dfs = (nan for i in range(5))
        Paa_c, PaaM_c = (np.full_like(omega_c, nan) for i in range(2))
    else:
        omega,Pww = sot.get_psd(g.w, fs=1/_prep_labsea_drift.dt)
        Paa = omega**2*Pww
        Pnoise = 1 # better moise model?
        eps, omega0, _ = sot.fit_spec(omega, Paa, Pnoise, _prep_labsea_drift.Lfloat)
        omegaL = (eps/(_prep_labsea_drift.Lfloat**2))**(1/3)
        nu = 1.67e-6 # kinematic viscosity [m^2/s], from The Engineering Toolbox [T=3, S=35]
        omega_Komogrov = 2*pi/np.sqrt(nu/eps) # ~ 1 rad/s
        omega_i = np.logspace(-5, np.around(np.log10(omega_Komogrov),1), 100)
        PaaM_c = sot.modelaccspL(eps, omega0, omega_c, Lfloat=_prep_labsea_drift.Lfloat)
        PaaM_i = sot.modelaccspL(eps, omega0, omega_i, Lfloat=_prep_labsea_drift.Lfloat)
        PaaM0_i = sot.modelaccspL(eps, omega0, omega_i, Lfloat=1e-10)
        delw2 = np.trapz(PaaM0_i*omega_i**(-2),omega_i) - np.trapz(PaaM_i*omega_i**(-2),omega_i)
        # missed variance from the lower Nyquist frequency (~ 1e-3 Hz), up to 0.5 Hz
        idx_dfs = (omega_i >= 2*pi*1e-3) & (omega_i <= 2*pi*0.5)
        w2dfs = np.trapz(PaaM_i[idx_dfs]*omega_i[idx_dfs]**(-2), omega_i[idx_dfs])
        Paa_c = np.interp(omega_c, omega, Paa, left=nan, right=nan)
    wrms2 = np.mean(g.w**2)
    return pd.DataFrame(data=dict(raw=Paa_c, fit=PaaM_c, eps=eps, omega0=omega0, omegaL=omegaL, myd=g.yd.values.mean(),
                                  wrms2=wrms2, delw2=delw2, w2dfs=w2dfs, omega=omega_c)).set_index('omega')


def adjust_fbld(dftb, idf, tstr1=None, tstr2=None, method='trend', pivot='end'):
    biased_bld = dftb.bld.sel(ID=idf,time=slice(tstr1,tstr2))
    fmean_bld = xr.apply_ufunc(sot.butter_lpfilt, dftb.bld.mean('ID').sel(time=slice(tstr1,tstr2)),
                               input_core_dims=[['time']], output_core_dims=[['time']], kwargs={'cutoff': 1/72, 'fs': 1})
    if method == 'trend':
        res_biased = stats.linregress(sot.pytime2yd(biased_bld.time), biased_bld)
        res_fmean = stats.linregress(sot.pytime2yd(fmean_bld.time), fmean_bld)
        r = res_fmean.slope/res_biased.slope
        if pivot == 'end':
            biased_bld_diff = biased_bld - biased_bld[-1]
            ref_bld = biased_bld[-1]
        elif pivot == 'start':
            biased_bld_diff = biased_bld - biased_bld[0]
            ref_bld = biased_bld[0]
    elif method == 'shift':
        r = 1
        if pivot == 'end':
            biased_bld_diff = biased_bld - biased_bld[-1]
            ref_bld = fmean_bld[-1]
        elif pivot == 'start':
            biased_bld_diff = biased_bld - biased_bld[0]
            ref_bld = fmean_bld[0]
    elif method == 'max':
        biased_bld_linear = np.linspace(biased_bld[0], biased_bld[-1], biased_bld.size)
        biased_bld_diff = biased_bld - biased_bld_linear
        fmean_bld_linear = np.linspace(fmean_bld[0], fmean_bld[-1], fmean_bld.size)
        fmean_bld_diff = fmean_bld - fmean_bld_linear
        r = max(abs(fmean_bld_diff))/max(abs(biased_bld_diff))
        ref_bld = biased_bld_linear
    crted_bld = biased_bld_diff*r + ref_bld
    dftb.bld.loc[idf,slice(tstr1,tstr2)] = crted_bld
    return dftb


def _prep_labsea_drift(GDrive_root, year, use_mean_bld=True, crt_flux=True):
    site_root = GDrive_root + 'UW/Research/Data/' + 'LabSea/'
    nbf_dir = site_root + 'NBF/'
    flux_dir = site_root + 'Met/'
    _prep_labsea_drift.Lfloat = 0.66/2 # [m]
    _prep_labsea_drift.dt = 300 # [s]
    p_calconst = 1e5 # [Pa]
    depth_std_min = 45 # [m]
    dth_ml = 0.1 # [C]
    dth_dt_max = 0.5 # [C/day]
    
    Env = xr.open_dataset(nbf_dir+'LabSea_env_'+str(year)+'.nc').load()
    Env.close()
    mlat = Env.lat.mean()
    latest_time = pd.Timestamp(Env.ftime.where(Env.drift, drop=True).max(skipna=True).values)
    earliest_time = pd.Timestamp(Env.ftime.where(Env.drift, drop=True).min(skipna=True).values)
    
    if crt_flux:
        nametag = ''
        save_spec_bld = True
    else:
        nametag = 'E'
        save_spec_bld = False
    flux = xr.open_dataset(flux_dir+'LabSea_fluxes_'+str(year)+nametag+'.nc').load()
    flux.close()
    
    # set bin edges
    time_start = earliest_time.floor('60min')
    time_end = latest_time.ceil('60min')
    hhtime = pd.Timedelta(minutes=30)
    time_bins = pd.date_range(time_start-hhtime, time_end+hhtime, freq='H')
    dz = 10 # [m]
    zbt = np.floor(-Env.p.where(Env.drift, drop=True).max()/dz)*dz
    z_bins = np.arange(zbt, dz, dz)
    
    ds, dsspec, dsbnd = ([] for i in range(3))
    for i,idf in enumerate(Env.ID.values):
        iEnv = Env[['p', 'th', 'drift']].sel(ID=idf).dropna('time')
        iEnv = iEnv.where(iEnv.drift, drop=True).rename({'ftime': 'time'})
        
        # remove float pressure signal due to atmopsheric pressure variations
        aPair = (flux.sp.mean('ID').interp(time=iEnv.time) - p_calconst)/1e4 # [dbar]
        iEnv.update({'p': iEnv.p-aPair})
        # all floats have minimum pressure > 1 dbar
        
        time = iEnv.time.values
        time_even = pd.date_range(start=time[0], end=time[-1], freq='5min')
        z = gsw.z_from_p(iEnv.p, mlat)
        df = pd.DataFrame(data=dict(time=time, z=z, th=iEnv.th.values)).set_index('time')
        df = df.loc[~df.index.duplicated(), :] # remove duplicated time points
        dfi = df.reindex(time_even)
        dfi.insert(0, 'yd', sot.pytime2yd(time_even.to_pydatetime()))
        
        # despike (identified by eye)
        if idf==12: # i=3 in 1997
            zdiff = dfi.z.diff().abs()
            ijump = np.where(zdiff>40)[0]
            dfi.z.iloc[ijump[0]:ijump[1]] = dfi.z.iloc[ijump[0]:ijump[1]] + 50
        
        # fill gaps
        dfi.insert(2, 'zitp', dfi.z.interpolate(method='pchip', limit_area='inside'))
        dfi.insert(4, 'thitp', dfi.th.interpolate(method='pchip', limit_area='inside'))
        
        # identify times in mixed layer
        if year==1997:
            thlp = sot.butter_lpfilt(dfi.thitp, 1/60/12, 1/5)
            dth_dt = np.gradient(thlp, dfi.yd)
            irestrat = np.where(np.abs(dth_dt) > dth_dt_max)[0]
            if irestrat.size!=0: # not empty
                dfi = dfi.iloc[:irestrat[0],:]
        elif year==1998:
            zinML = dfi.zitp.where(dfi.zitp > -200)
            _,isurf = sot.get_clumps(zinML)
            isurf0 = isurf[0]
            th_mixbot = dfi.thitp.iloc[isurf0].mean() + dth_ml
            iexcited = np.where(dfi.thitp.iloc[:isurf0.start] > th_mixbot)[0][-1] + 1
            dfi = dfi.iloc[iexcited:]
            # ignore precipitation affected period
            idx_rain = (dfi.index >= pd.Timestamp('1998-03-10 05:30:00')) & \
                       (dfi.index <= pd.Timestamp('1998-03-12 19:30:00'))
            dfi.iloc[idx_rain,:] = nan
        
        # compute mixing depth
        if dfi.z.notnull().sum()/12/24 < 3: # less than 3 days
            print('drift too short')
            dfi['bld'] = np.full_like(dfi.index, nan, dtype=float)
        else:
            ustar = np.sqrt(flux.tau.mean('ID').shift(time=0).interp(time=dfi.index)/rho0)
            wbnet = flux.wbnet.mean('ID').shift(time=0).interp(time=dfi.index)
            # remove bottom trapped period by requiring large 3-day rolling std of depth
            depth_std = (-dfi.zitp).rolling(window=12*72, min_periods=12*12, center=True, win_type='hamming').std()
            idx_trap = depth_std < depth_std_min
            dfi.iloc[idx_trap,:] = nan
            dfi['bld'] = sot.get_hmix(dfi.yd, -dfi.zitp, ustar, wbnet, _prep_labsea_drift.dt, DOF=10)
            dfi['bld'] = sot.butter_lpfilt(dfi.bld, 1/60/72, 1/5) # 3-day lowpass
        
        # drop leading and trailing NaNs
        first_tstamp = dfi.zitp.first_valid_index()
        last_tstamp = dfi.zitp.last_valid_index()
        dfi = dfi.loc[first_tstamp:last_tstamp,:]
        
        dfi['w'] = np.convolve(dfi.zitp, [1, 0, -1], 'same')/(2*_prep_labsea_drift.dt)
        dfi['wf'] = np.convolve(dfi.z, [1, 0, -1], 'same')/(2*_prep_labsea_drift.dt)
        # if idf==12:
        #     ijump_w = np.concatenate((ijump-1,ijump)) # center difference affects more data
        #     dfi.wf.iloc[ijump_w] = nan
        dfi = dfi.iloc[1:-1,:] # invalid w at both ends
        dfi['wf2'] = dfi.wf**2
        dfi['wf3'] = dfi.wf**3
        
        # compute spectrum
        if dfi.w.isnull().sum() > 0:
            _,Iclps = sot.get_clumps(dfi.w)
            dfi['drift'] = np.full_like(dfi.w, nan)
            drift_pre = 0
            for iclp in Iclps:
                Nseg = int(np.around((iclp.stop-iclp.start)/1024))
                dfi.drift.iloc[iclp] = pd.cut(dfi.index[iclp], max(1,Nseg), labels=False) + drift_pre
                drift_pre = dfi.drift.iloc[iclp].max() + 1
        else:
            Nseg = int(np.around(dfi.index.size/1024)) # ~ 3.56 days per segment
            dfi['drift'] = pd.cut(dfi.index, max(1,Nseg), labels=False)
        dsspeci = dfi[dfi.drift.notnull()].groupby('drift').apply(accspecL) \
                                          .to_xarray().expand_dims('ID').assign_coords(ID=[idf])
        varlist = ['wrms2', 'delw2', 'w2dfs', 'eps', 'omega0', 'omegaL', 'myd']
        for var in varlist:
            dsspeci.update({var: dsspeci[var].mean('omega')})
        dsspec.append(dsspeci)
        
        # save each drift
        dfi = dfi.reset_index().rename({'index': 'time'}, axis='columns')
        wf2bulk = dsspeci.wrms2.to_dataframe().reset_index(level='ID', drop=True).rename(columns={'wrms2':'wf2bulk'})
        w2dfs   = dsspeci.w2dfs.to_dataframe().reset_index(level='ID', drop=True)
        epsbulk = dsspeci.eps.to_dataframe().reset_index(level='ID', drop=True).rename(columns={'eps':'epsbulk'})
        dfi = dfi.set_index('drift').join(wf2bulk, on='drift')
        dfi = dfi.join(w2dfs, on='drift')
        dfi = dfi.join(epsbulk, on='drift')
        dfi = dfi.reset_index().rename({'index': 'drift'}, axis='columns')
        dsi = dfi.rename({'time': 'ftime'}, axis='columns').to_xarray().expand_dims('ID').assign_coords(ID=[idf])
        ds.append(dsi)
        
        # bin w
        grpd = dfi.groupby([pd.cut(dfi.time, time_bins), pd.cut(dfi.zitp, z_bins)])
        grpm = grpd.mean()
        grpc = grpd.wf.count().replace(0, nan).rename('n')
        grp_zstd = grpd.zitp.std(ddof=0).rename('zstd')
        bnd = pd.concat([grpm, grpc, grp_zstd], axis=1)
        bnd.index.set_names(['time','z'], inplace=True)
        bnd = bnd.rename(columns={'z': 'mz', 'yd': 'myd'}).drop(columns=['zitp','thitp']).reset_index()
        bnd.time = bnd.time.apply(lambda x: x.mid)
        bnd.z = bnd.z.apply(lambda x: x.mid)
        dsbndi = bnd.set_index(['time', 'z']).to_xarray().expand_dims('ID').assign_coords(ID=[idf])
        dsbnd.append(dsbndi)
    
    # concat each float
    dft = xr.concat(ds, dim='ID').rename_dims({'index': 'time'}).drop_vars('index').set_coords('ftime')
    spec = xr.concat(dsspec, dim='ID')
    spec.attrs['Lfloat'] = _prep_labsea_drift.Lfloat
    dftb = xr.concat(dsbnd, dim='ID')
    dftb.update({'myd': dftb.myd.mean('z')})
    dftb.update({'drift': dftb.drift.mean('z').round()})
    dftb.update({'bld': dftb.bld.mean('z')})
    dftb.update({'epsbulk': dftb.epsbulk.mean(dim='z')})
    dftb.update({'wf2bulk': dftb.wf2bulk.mean(dim='z')})
    dftb.update({'w2dfs': dftb.w2dfs.mean(dim='z')})
    dftb = dftb.assign_coords(mtime=(('ID','time'), sot.yd2pytime(dftb.myd, year)))
    dftb = dftb.dropna('time', how='all')
    # w data has more NaNs at the edges of P gaps
    dftb.update({'mz': dftb.mz.where(dftb.wf.notnull())})
    dftb.update({'zstd': dftb.zstd.where(dftb.wf.notnull())})
    
    # modify biased bld
    # if year==1997:
    #     dftb = adjust_fbld(dftb, 12, tstr2='1997-03-04 17:00:00')
    #     dftb = adjust_fbld(dftb, 19, tstr2='1997-02-28 09:00:00')
    # elif year==1998:
    #     dftb = adjust_fbld(dftb, 27, tstr1='1998-03-14 04:00:00', method='shift')
    #     dftb = adjust_fbld(dftb, 27, tstr1='1998-02-24 17:00:00', tstr2='1998-03-06 07:00:00', pivot='start')
    #     dftb = adjust_fbld(dftb, 33, tstr1='1998-02-27 00:00:00', tstr2='1998-03-05 01:00:00', method='max')
    #     dftb = adjust_fbld(dftb, 23, tstr1='1998-03-16 04:00:00', method='shift')
    
    if use_mean_bld:
        fID = dftb.ID
    else: # only use mean bld for short drifts which have no bld info
        fID = dftb.ID[dftb.bld.isnull().all(dim='time')]
    fmean_bld = xr.apply_ufunc(sot.butter_lpfilt, dftb.bld.mean('ID'), input_core_dims=[['time']],
                               output_core_dims=[['time']], kwargs={'cutoff': 1/72, 'fs': 1})
    for idf in fID:
        # time_no_nan = dftb.time[dftb.mz.sel(ID=idf).notnull().any('z')]
        # dftb.bld.loc[idf,time_no_nan] = fmean_bld.sel(time=time_no_nan)
        dftb.bld.loc[idf,:] = fmean_bld
    
    # add forcing
    dftb_flux = flux[['ustar','taux','tauy','wb0','wbnet']].shift(time=0).sel(time=dftb.time)
    Bnsw = -(dftb_flux.wbnet - dftb_flux.wb0)
    Bf, wbf = sot.get_BFs_Jwt(dftb_flux.wb0, Bnsw, dftb.bld, Jwtype='III')
    
    wstar = np.cbrt(wbf*dftb.bld)
    dftb['theta_tau'] =  np.arctan2(dftb_flux.tauy, dftb_flux.taux)
    dftb['ustar'] = dftb_flux.ustar
    dftb['wstar'] = wstar
    wfs2 = 1.1*dftb.ustar**2 + np.copysign(0.3*wstar**2, wbf)
    wfs2 = wfs2.where((wfs2>=1e-10) | wfs2.isnull(), other=1e-10)
    dftb['Bf'] = Bf
    dftb['wbf'] = wbf
    dftb['wfs'] = np.sqrt(wfs2)
    dftb['LObukhov'] = -dftb.ustar**3/kappa/wbf
    dftb['zeta'] = np.abs(dftb.mz)/dftb.LObukhov
    dftb['zoh'] = dftb.mz/dftb.bld
    
    oneozM = xr.apply_ufunc(sot.get_oneozM, dftb.mz, dftb.zstd, 
                            input_core_dims=[[], []], vectorize=True, output_dtypes=[float])
    dftb['ESP'] = dftb.ustar**3*(1+dftb.zoh)/kappa*oneozM
    dftb['BF'] = dftb_flux.wb0*(1+dftb.zoh) + sot.get_wbr_Jwt(Bnsw, dftb.mz, dftb.bld, Jwtype='III')
    dftb['BFsl'] = dftb.wbf*(1+dftb.zoh)
    
    # fill gaps in epsbulk, w2dfs using the mean of other floats (cfac2 correcion less than 5% in LabSea)
    idx_float = dftb.ID[dftb.wf2bulk.notnull().sum('time') != dftb.epsbulk.notnull().sum('time')]
    for idf in idx_float.data:
        idx_no_cfac = dftb.wf2bulk.sel(ID=idf).notnull() & dftb.epsbulk.sel(ID=idf).isnull()
        dftb.epsbulk.sel(ID=idf)[idx_no_cfac] = (dftb.epsbulk[:,idx_no_cfac.data]).mean('ID')
        dftb.w2dfs.sel(ID=idf)[idx_no_cfac] = (dftb.w2dfs[:,idx_no_cfac.data]).mean('ID')
    
    dftb['wf2bulk'] = dftb.wf2bulk + dftb.w2dfs
    cfac = sot.crt_wrms_bulk(dftb.epsbulk, _prep_labsea_drift.Lfloat, dftb.wf2bulk)
    dftb['cfac2'] = cfac**2
    cfac,_ = xr.broadcast(cfac, dftb.z)
    dftb['cfacz2'] = (('ID', 'time', 'z'), sot.crt_wrms_prof(cfac, dftb.zoh, cshape='linear')**2)
    
    # unable to correct for wf3
    dftb['wf2'] = dftb.wf2 + dftb.w2dfs
    dftb['wf2our2'] = dftb.wf2/dftb.ustar**2
    dftb['wf2owr2'] = dftb.wf2/dftb.wstar**2
    
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
    
    # save
    nc_name = 'LabSea_drifts_'+str(year)+nametag+'.nc'
    dft.to_netcdf(nbf_dir + nc_name, engine='netcdf4')
    print(f'Processed float drift data: {nc_name} saved at {nbf_dir}.')
    
    nc_name = 'LabSea_drifts_binned_'+str(year)+nametag+'.nc'
    dftb.to_netcdf(nbf_dir + nc_name, engine='netcdf4')
    print(f'Binned float drift data: {nc_name} saved at {nbf_dir}.')
    
    if save_spec_bld:
        nc_name = 'LabSea_drifts_spec_'+str(year)+'.nc'
        spec.to_netcdf(nbf_dir + nc_name, engine='netcdf4')
        print(f'Float drift spectra: {nc_name} saved at {nbf_dir}.')

        nc_name = 'LabSea_env_'+str(year)+'.nc'
        Env = Env.assign({'bld': ('timePOS', dftb.bld.mean('ID').interp(time=Env.timePOS).data, 
                                  {'longname': 'Mean boundary layer depth'})})
        Env.to_netcdf(nbf_dir + nc_name, engine='netcdf4')
        print(f'Adding boundary layer depth to env file: {nc_name} saved at {nbf_dir}.\n')


if platform == 'linux':
    Groot = '/media/zhihua/internal1/GDrive/'
elif platform == 'darwin':
    Groot = '/Volumes/GoogleDrive/My Drive/'
_prep_labsea_drift(Groot, 1997)
_prep_labsea_drift(Groot, 1997, crt_flux=False)
# _prep_labsea_drift(Groot, 1998)