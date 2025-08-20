import xarray as xr
import numpy as np
import pandas as pd
import sca_osbl_tool as sot
from sys import platform
from constants import nan, pi, g, rho0, kappa


def _prep_ocsp_wave(GDrive_root, year, downwind=True):
    site_root = GDrive_root + 'UW/Research/Data/OCSP/'
    nbf_dir = site_root + 'NBF/'
    wave_dir = site_root + 'Wave/'
    flux_dir = site_root + 'Mooring/'+str(year)+'_high_res/fluxes/'
    dftb_name = 'OCSP_drifts_binned_'+str(year)+'.nc'
    bld_name = 'OCSP_bld_'+str(year)+'.nc'
    flux_name = 'OCSP_fluxes_'+str(year)+'.nc'
    
    dftb = xr.open_dataset(nbf_dir+dftb_name).load()
    dftb.close()
    ustar = dftb.ustar.copy()
    wstar = dftb.wstar.copy()
    bld = dftb.bld.copy()
    
    with xr.open_dataset(wave_dir+'166p1_historic.nc') \
           .drop_vars(['metaStationLatitude', 'metaStationLongitude']) \
           .rename({'waveTime': 'time', 'waveFrequency': 'freq'}) \
           .sel(time=slice(dftb.time[0] - pd.Timedelta('1hr'),
                           dftb.time[-1] + pd.Timedelta('1hr'))) as cdip:
        wave_a0 = cdip.waveEnergyDensity/pi
        wave_a1 = cdip.waveA1Value*wave_a0
        wave_b1 = cdip.waveB1Value*wave_a0
        
        with xr.open_dataarray(nbf_dir+bld_name) as ds:
            cdip_bld = ds.interp(time=cdip.time, kwargs={'fill_value': (ds[0], ds[-1])})
        
        # with xr.open_dataset(flux_dir+flux_name) as F:
        #     taux = F.taux.squeeze(['lon','lat','depth'], drop=True).reset_coords('yd', drop=True).interp(time=cdip.time)
        #     tauy = F.tauy.squeeze(['lon','lat','depth'], drop=True).reset_coords('yd', drop=True).interp(time=cdip.time)
        #     ustar = np.sqrt(np.sqrt(taux**2 + tauy**2)/rho0)
        #     theta_tau = np.arctan2(tauy, taux)
    
    # significant wave height of wind waves 
    Hsww = xr.apply_ufunc(sot.get_Hsww, cdip.freq, cdip.waveEnergyDensity, cdip.waveBandwidth, 
                          input_core_dims=[['freq'], ['freq'], ['freq']],
                          vectorize=True, output_dtypes=[float]) \
             .resample(time='60min', base=30, loffset=pd.Timedelta(minutes=30)).mean().interp(time=dftb.time)
    
    cdip_theta_tau = dftb.theta_tau.interp(time=cdip.time, method='nearest', kwargs={'fill_value': (dftb.theta_tau[0], dftb.theta_tau[-1])})
    ust_param = xr.apply_ufunc(sot.get_ust_param, cdip.freq, cdip.waveBandwidth, wave_a1, wave_b1, cdip_bld, cdip_theta_tau,
                               input_core_dims=[['freq'], ['freq'], ['freq'], ['freq'], [], []],
                               vectorize=True, output_core_dims=[['param']], output_dtypes=[float])
    
    # average in 1 hour and assign to integer hours, actually no interpolation here
    hhour = pd.Timedelta(minutes=30)
    Usdw_SL   = ust_param.isel(param=0).resample(time='60min', base=30, loffset=hhour).mean().interp(time=dftb.time)
    Uscw_SL   = ust_param.isel(param=1).resample(time='60min', base=30, loffset=hhour).mean().interp(time=dftb.time)
    Usdw_ref  = ust_param.isel(param=2).resample(time='60min', base=30, loffset=hhour).mean().interp(time=dftb.time)
    Uscw_ref  = ust_param.isel(param=3).resample(time='60min', base=30, loffset=hhour).mean().interp(time=dftb.time)
    Usdw0     = ust_param.isel(param=4).resample(time='60min', base=30, loffset=hhour).mean().interp(time=dftb.time)
    Uscw0     = ust_param.isel(param=5).resample(time='60min', base=30, loffset=hhour).mean().interp(time=dftb.time)
    Usdw_SLa  = ust_param.isel(param=6).resample(time='60min', base=30, loffset=hhour).mean().interp(time=dftb.time)
    Uscw_SLa  = ust_param.isel(param=7).resample(time='60min', base=30, loffset=hhour).mean().interp(time=dftb.time)
    Usdw_10ph = ust_param.isel(param=8).resample(time='60min', base=30, loffset=hhour).mean().interp(time=dftb.time)
    Uscw_10ph = ust_param.isel(param=9).resample(time='60min', base=30, loffset=hhour).mean().interp(time=dftb.time)
    Usdw_20ph = ust_param.isel(param=10).resample(time='60min', base=30, loffset=hhour).mean().interp(time=dftb.time)
    Uscw_20ph = ust_param.isel(param=11).resample(time='60min', base=30, loffset=hhour).mean().interp(time=dftb.time)
    Usdw_SSL  = ust_param.isel(param=12).resample(time='60min', base=30, loffset=hhour).mean().interp(time=dftb.time)
    Uscw_SSL  = ust_param.isel(param=13).resample(time='60min', base=30, loffset=hhour).mean().interp(time=dftb.time)
    # Hs        = cdip.waveHs.resample(time='60min', base=30, loffset=hhour).mean().interp(time=dftb.time)
    
    Us_SL  = np.sqrt(Usdw_SL**2 + Uscw_SL**2)
    Us_SLa = np.sqrt(Usdw_SLa**2 + Uscw_SLa**2)
    Us_ref = np.sqrt(Usdw_ref**2 + Uscw_ref**2)
    Us0    = np.sqrt(Usdw0**2 + Uscw0**2)
    Us_10ph = np.sqrt(Usdw_10ph**2 + Uscw_10ph**2)
    Us_20ph = np.sqrt(Usdw_20ph**2 + Uscw_20ph**2)
    
    # theta_wave = np.arctan2(vs_SL, us_SL) # np.arctan2(vs_SL-vs_ref, us_SL-us_ref)
    # theta_ww = theta_wave - dftb.theta_tau
    theta_ww = np.arctan2(Uscw_SL, Usdw_SL)
    z1 = np.maximum(Hsww, 0.02)*4 # Van Roekle et al. 2012, Eq. (29) for alpha_LOW
    theta_wL = np.arctan2(np.sin(theta_ww), ustar/kappa/Us0*np.log(bld/z1) + np.cos(theta_ww))
    idx_noLOW = bld <= z1 # assume Langmuir cells align with wave direction when bld <= z1
    theta_wL[idx_noLOW] = theta_ww[idx_noLOW]
    
    # Stokes decay scale
    # theta_ww0 = np.arctan2(Uscw0, Usdw0)
    theta_wave = theta_ww + dftb.theta_tau
    cdip_theta_wave = theta_wave.interp(time=cdip.time, method='nearest', kwargs={'fill_value': (theta_wave[0], theta_wave[-1])})
    stds = xr.apply_ufunc(sot.get_stds, cdip.freq, cdip.waveBandwidth, wave_a1, wave_b1, cdip_theta_wave,
                          input_core_dims=[['freq'], ['freq'], ['freq'], ['freq'], []], 
                          vectorize=True, output_dtypes=[float]) \
             .resample(time='60min', base=30, loffset=hhour).mean().interp(time=dftb.time)
    
    # assume Us_ref is in the same direction of Us_SL
    # theta_wL and (theta_ww - theta_wL) is limited to 90
    # small = 1e-8
    # La2_SLP = ustar*np.maximum(np.cos(theta_wL), small) / np.maximum((Us_SL - Us_ref)*np.cos(theta_ww - theta_wL), small)
    # La2_SL = ustar/np.maximum((Usdw_SL - Usdw_ref), small) # Usdw_ref can be larger than Usdw_SL sometimes
    # La2_t = ustar/np.maximum(Usdw0, small)
    
    dftb['zb'] = -0.6*Hsww
    dftb['zt'] = -10*Hsww
    dftb['stds'] = stds
    dftb['Us0'] = Us0
    dftb['UsSL'] = Us_SL - Us_ref
    dftb['theta_ww'] = theta_ww
    dftb['theta_wL'] = theta_wL
    dftb['LaSLP2'] = ustar*np.cos(theta_wL) / ((Us_SL - Us_ref)*np.cos(theta_ww - theta_wL)) # not filtered
    if downwind:
        dftb['LaSL2'] = ustar/(Usdw_SL - Usdw_ref)
        dftb['LaSSL2'] = ustar/(Usdw0 - Usdw_20ph)
        dftb['Lat2'] = ustar/Usdw0
        tag = 'dw'
    else:
        dftb['LaSL2'] = ustar/dftb.UsSL
        dftb['LaSSL2'] = ustar/(Us0 - Us_20ph)
        dftb['Lat2'] = ustar/dftb.Us0
        tag = ''
    
    # Stokes similarity function
    if downwind:
        dftb['xi'] = xr.apply_ufunc(sot.get_xi, Usdw0, Usdw_SLa, Usdw_10ph, ustar, wstar, bld, 
                                    input_core_dims=[[], [], [], [], [], []], vectorize=True, output_dtypes=[float])
    else:
        dftb['xi'] = xr.apply_ufunc(sot.get_xi, Us0, Us_SLa, Us_10ph, ustar, wstar, bld, 
                                    input_core_dims=[[], [], [], [], [], []], vectorize=True, output_dtypes=[float])
    dftb['chim'] = xr.apply_ufunc(sot.get_emp_chi, dftb.xi, input_core_dims=[[]], 
                                  kwargs={'var': 'mom'}, vectorize=True, output_dtypes=[float])
    
    # stability lengths
    cdip_ustar = ustar.interp(time=cdip.time, method='nearest', kwargs={'fill_value': (ustar[0], ustar[-1])})
    # cdip_wbf = dftb.wbf.interp(time=cdip.time, method='nearest', kwargs={'fill_value': (dftb.wbf[0], dftb.wbf[-1])})
    # cdip_chim = dftb.chim.interp(time=cdip.time, method='nearest', kwargs={'fill_value': (dftb.chim[0], dftb.chim[-1])})
    # Lstbl = xr.apply_ufunc(sot.get_stability_length, cdip.freq, cdip.waveBandwidth, wave_a1, wave_b1, 
    #                        cdip_theta_tau, cdip_ustar, cdip_wbf, #cdip_chim, 
    #                        input_core_dims=[['freq'], ['freq'], ['freq'], ['freq'], [], [], []],#, []],
    #                        vectorize=True, output_core_dims=[['param']], output_dtypes=[float])
    # dftb['LOg'] = Lstbl.isel(param=0).resample(time='60min', base=30, loffset=pd.Timedelta(minutes=30)).mean().interp(time=dftb.time)
    # dftb['Lss'] = Lstbl.isel(param=1).resample(time='60min', base=30, loffset=pd.Timedelta(minutes=30)).mean().interp(time=dftb.time)
    # dftb['LHo'] = Lstbl.isel(param=2).resample(time='60min', base=30, loffset=pd.Timedelta(minutes=30)).mean().interp(time=dftb.time)
    # dftb['zoLOg'] = np.abs(dftb.mz)/dftb.LOg
    # dftb['zoLss'] = np.abs(dftb.mz)/dftb.Lss
    # dftb['zoLHo'] = np.abs(dftb.mz)/dftb.LHo
#     SSP_SL = 5*dftb.ustar2/bld*(Usdw0 - 0.8*Usdw_20ph - 0.2*Usdw_SL)
#     c = 0.9
#     LOb = -np.sign(dftb.wbf)*dftb.chim*dftb.ustar2**(3/2)/kappa/(np.abs(dftb.wbf)-SSP_SL/c)
#     Lss = dftb.chim*dftsb.ustar2**(3/2)/kappa/(SSP_SL/c)
#     dftb['LOb_sl'] = LOb
#     dftb['Lss_sl'] = Lss
    
    # downwind Stokes shear
    cdip_mz = dftb.mz.interp(time=cdip.time, method='nearest', kwargs={'fill_value': (dftb.mz[0,:], dftb.mz[-1,:])})
    cdip_zstd = dftb.zstd.interp(time=cdip.time, method='nearest', kwargs={'fill_value': (dftb.zstd[0,:], dftb.zstd[-1,:])})
    # cdip_zstd = xr.full_like(cdip_mz, 0)
    UsdwshM = xr.apply_ufunc(sot.get_dUsdwdzM, cdip_mz, cdip_zstd, 
                             cdip.freq, cdip.waveBandwidth, wave_a1, wave_b1, cdip_theta_tau, 
                             input_core_dims=[[], [], ['freq'], ['freq'], ['freq'], ['freq'], []],
                             vectorize=True, output_dtypes=[float], kwargs={'downwind': downwind}) \
                .resample(time='60min', base=30, loffset=hhour).mean('time')
    # need to use sel here, interp would mess up the data since the non-nan mz varies with time
    if UsdwshM.time[-1] < dftb.time[-1]: # if wave data is shorter, then pad with nan
        inulls = np.where(dftb.time>UsdwshM.time[-1])[0][0]
        UsdwshM_null = xr.DataArray(data=np.full((dftb.time[inulls:].size, dftb.dims['z']), nan), 
                                    dims=['time','z'], 
                                    coords=dict(time=dftb.time[inulls:], z=dftb.z))
        dftb['UsdwshM'] = xr.concat([UsdwshM, UsdwshM_null], dim='time').sel(time=dftb.time)
    else:
        dftb['UsdwshM'] = UsdwshM.sel(time=dftb.time)
    dftb['SSP'] = ustar**2*(1+dftb.zoh)*dftb.UsdwshM
    
    # surface proximity function (1-fzS) decay depth
    dftb['Ls'] = xr.apply_ufunc(sot.get_Ls, cdip.freq, cdip.waveBandwidth, wave_a1, wave_b1, 
                                cdip_ustar, cdip_bld, cdip_theta_tau, 
                                input_core_dims=[['freq'], ['freq'], ['freq'], ['freq'], [], [], []],
                                vectorize=True, output_dtypes=[float], kwargs={'downwind': downwind}) \
                   .resample(time='60min', base=30, loffset=hhour).mean('time').interp(time=dftb.time)
    
    dftb['spf'] = 1 - xr.apply_ufunc(sot.get_fzSM, dftb.mz, dftb.zstd, dftb.Ls, 
                                     input_core_dims=[[], [], []], vectorize=True, output_dtypes=[float])
    
    nc_name = f'OCSP_dftb_wave_{year}{tag}.nc'
    dftb.to_netcdf(nbf_dir + nc_name, engine='netcdf4', # encoding makes sure the boolean dtype
                   encoding={'Iequil': {'dtype': 'bool'}, 'Ishoal': {'dtype': 'bool'}, 'Ideepen': {'dtype': 'bool'}})
    print(f'Binned float drift and wave data: {nc_name} saved at {nbf_dir}.')


if platform == 'linux':
    Groot = '/media/zhihua/internal1/GDrive/'
elif platform == 'darwin':
    Groot = '/Volumes/GoogleDrive/My Drive/'
# _prep_ocsp_wave(Groot, 2011, downwind=False)
_prep_ocsp_wave(Groot, 2011, downwind=True)
# _prep_ocsp_wave(Groot, 2012, downwind=False)
_prep_ocsp_wave(Groot, 2012, downwind=True)