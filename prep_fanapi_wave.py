import xarray as xr
import numpy as np
import pandas as pd
import sca_osbl_tool as sot
import scipy.io as sio
from sys import platform
from constants import nan, pi, rho0, kappa


def get_SL_mean(q, z, delz, bld):
    """
    Compute mean quantity in surface layer
    """
    sld = bld/5
    isld = np.where(z >= -sld)[0][0]
    delz_res = z[isld] + sld
    return (np.sum(delz[isld:]*q[isld:]) + delz_res*q[isld-1]) / sld


# def get_La(Lagcur, ustar, theta_tau, h='hmix'):
#     """
#     Compute Langmuir numbers
#     """
#     if h == 'hmix':
#         bld = Lagcur.hmix
#     elif h == 'hobl':
#         bld = Lagcur.hobl_kpp
#     us_SL = xr.apply_ufunc(get_SL_mean, Lagcur.us, Lagcur.z, Lagcur.delz, bld,
#                            input_core_dims=[['z'], ['z'], ['z'], []], vectorize=True, output_dtypes=[float])
#     vs_SL = xr.apply_ufunc(get_SL_mean, Lagcur.vs, Lagcur.z, Lagcur.delz, bld,
#                            input_core_dims=[['z'], ['z'], ['z'], []], vectorize=True, output_dtypes=[float])
#     us_ref = xr.apply_ufunc(np.interp, -bld, Lagcur.z, Lagcur.us,
#                             input_core_dims=[[], ['z'], ['z']], vectorize=True, output_dtypes=[float])
#     vs_ref = xr.apply_ufunc(np.interp, -bld, Lagcur.z, Lagcur.vs,
#                             input_core_dims=[[], ['z'], ['z']], vectorize=True, output_dtypes=[float])
#     us1 = Lagcur.us.sel(z=-1)
#     vs1 = Lagcur.vs.sel(z=-1)
#     duLdz_SL = xr.apply_ufunc(get_SL_mean, Lagcur.duLdz, Lagcur.z, Lagcur.delz, bld,
#                               input_core_dims=[['z'], ['z'], ['z'], []], vectorize=True, output_dtypes=[float])
#     dvLdz_SL = xr.apply_ufunc(get_SL_mean, Lagcur.dvLdz, Lagcur.z, Lagcur.delz, bld,
#                               input_core_dims=[['z'], ['z'], ['z'], []], vectorize=True, output_dtypes=[float])
#     Us_SL = np.sqrt(us_SL**2 + vs_SL**2)
#     Us_ref = np.sqrt(us_ref**2 + vs_ref**2)
#     Us1 = np.sqrt(us1**2 + vs1**2) # Stokes drift averaged over upper 2m
#     Usdw_SL  = us_SL*np.cos(theta_tau) + vs_SL*np.sin(theta_tau)
#     Uscw_SL  = -us_SL*np.sin(theta_tau) + vs_SL*np.cos(theta_tau)
#     Usdw_ref = us_ref*np.cos(theta_tau) + vs_ref*np.sin(theta_tau)
#     Uscw_ref = -us_ref*np.sin(theta_tau) + vs_ref*np.cos(theta_tau)
#     Usdw1    = us1*np.cos(theta_tau) + vs1*np.sin(theta_tau)
#     Uscw1    = -us1*np.sin(theta_tau) + vs1*np.cos(theta_tau)
#     theta_wave = np.arctan2(vs_SL, us_SL)
#     theta_Lshr = np.arctan2(dvLdz_SL, duLdz_SL)
#     theta_ww = theta_wave - theta_tau
#     z0 = 1 # first model level, Van Roekle et al. 2012
#     theta_wL = np.arctan2(Uscw_SL-Uscw_ref, ustar/kappa/(bld-z0)*np.log(bld/z0)*bld*0.9 + (Usdw_SL-Usdw_ref))
#     # idx_noLOW = bld <= z0 # assume Langmuir cells align with wave direction when bld <= z0
#     # theta_wL[idx_noLOW] = theta_ww[idx_noLOW]
#     # theta_wL = theta_Lshr - theta_tau
#     # assume Us_ref is in the same direction of Us_SL
#     # theta_wL and (theta_ww - theta_wL) is limited to 90
#     small = 1e-8
#     La_SLP = np.sqrt(ustar*np.maximum(np.cos(theta_wL), small) / 
#                       np.maximum((Us_SL - Us_ref)*np.cos(theta_ww - theta_wL), small))
#     La_SL = np.sqrt(ustar/np.maximum((Usdw_SL - Usdw_ref), small)) # Usdw_ref can be larger than Usdw_SL sometimes
#     La_t = np.sqrt(ustar/np.maximum(Usdw1, small))
#     return La_SLP, La_SL, La_t


def _prep_fanapi_wave(GDrive_root, floatID, tau_opt, tail):
    site_root = GDrive_root + 'UW/Research/Data/Hurricanes/ITOP/'
    nbf_dir = site_root + 'NBF/'
    met_dir = site_root + 'Met/'
    wave_dir = site_root + 'Wave/'
    drift_name = f'Fanapi_drifts_{floatID}_{tau_opt}.nc'
    dftb_name = f'Fanapi_drifts_binned_{floatID}_{tau_opt}.nc'
    year = 2010
    
    dftb = xr.open_dataset(nbf_dir+dftb_name).load()
    dftb.close()
    dftb['lon'] = np.around(dftb.lon, 3)
    dftb['lat'] = np.around(dftb.lat, 3)
    ustar = dftb.ustar.copy()
    wstar = dftb.wstar.copy()
    theta_tau = dftb.theta_tau.copy()
    bld = dftb.bld.copy()
    
#    # MOM6-WW3 simulation from Zhou et al. 2022
#     sim = sio.loadmat(wave_dir+'UV_Stokes_'+str(floatID)+'.mat', squeeze_me=True, struct_as_record=False)
#     pytime = sot.mtime2pytime(sim['timef'])
#     # Stokes drift is from WW3 resovled spectrum
#     Lagcur = xr.Dataset(data_vars=dict(u=(('z','time'), np.flip(sim['u_obs_MOM6'],0)), 
#                                        v=(('z','time'), np.flip(sim['v_obs_MOM6'],0)),
#                                        us=(('z','time'), np.flip(sim['us_obs_MOM6'],0)),
#                                        vs=(('z','time'), np.flip(sim['vs_obs_MOM6'],0)),
#                                        delz=(('z','time'), np.flip(sim['thkcello_obs_MOM6'],0)),
#                                        hobl_kpp=(('time'), sim['KPP_OBL_float']),
#                                        La_SL_kpp=(('time'), sim['KPP_La_SL_float'])),
#                         coords=dict(z=(('z'), np.flip(-sim['zl'])),
#                                     time=(('time'), pytime)),
#                         attrs=dict(description='Simulated current and Stokes drift profiles under Typhoon Fanapi ' + \
#                                                'for Lagrangian float ' + str(floatID),
#                                    source='Zhou et al. 2022'))
#     with xr.open_dataset(nbf_dir+f'Fanapi_drifts_{floatID:02d}.nc') as dft:
#         Lagcur['hmix'] = dft.bld.interp(time=Lagcur.time)
#     with xr.open_dataset(nbf_dir+f'Fanapi_drifts_binned_{floatID:02d}.nc') as dftb:
#         in_dftb = (Lagcur.time >= (dftb.time[0]-pd.Timedelta(hours=0.5))) & \
#                   (Lagcur.time <= (dftb.time[-1]+pd.Timedelta(hours=0.5)))
#         Lagcur = Lagcur.where(in_dftb, drop=True)
#     tau_option = 'APL'
#     with xr.open_dataset(met_dir+f'Fanapi_fluxes_{floatID:02d}_{tau_option}.nc') as F:
#         taux = F.taux.interp(time=Lagcur.time)
#         tauy = F.tauy.interp(time=Lagcur.time)
#         ustar = np.sqrt( np.sqrt(taux**2 + tauy**2) / rho0)
#         theta_tau = np.arctan2(tauy, taux)
#     Lagcur['duLdz'] = (Lagcur.u + Lagcur.us).differentiate('z')
#     Lagcur['dvLdz'] = (Lagcur.v + Lagcur.vs).differentiate('z')
    
    dws = xr.open_dataset(wave_dir+'ww3_2010_src.nc').load() \
            .rename({'time': 'dws_time', 'longitude': 'lon', 'latitude': 'lat', 'frequency': 'freq', 'direction': 'dirc'})
    dws = dws.where(dws.lon.isin(dftb.lon), drop=True)
    dws = dws.sel(dws_time=dftb.time.data, method='nearest')
    dws = dws.sel(dws_time=xr.DataArray(dws.dws_time.data, dims='time'), 
                  station=xr.DataArray(dws.station.data, dims='time')) \
             .drop_vars(['string16', 'station']).rename_vars({'dws_time': 'time'})
    
    drad = dws.dirc/180*pi # to, clockwise relative to the North
    bw_drad = np.abs(drad[1] - drad[0])
    us_d2f = (np.sin(drad)*dws.efth*bw_drad).sum('dirc', skipna=False).set_index(time='time')
    vs_d2f = (np.cos(drad)*dws.efth*bw_drad).sum('dirc', skipna=False).set_index(time='time')
    d2f = np.sqrt(us_d2f**2 + vs_d2f**2)
    
    # significant wave height of wind waves 
    Hsww = xr.apply_ufunc(sot.get_Hsww_d2f, dws.freq, d2f, 
                          input_core_dims=[['freq'], ['freq']], vectorize=True, output_dtypes=[float])
    
    ust_param = xr.apply_ufunc(sot.get_ust_param_d2f, dws.freq, us_d2f, vs_d2f, bld, theta_tau,
                               input_core_dims=[['freq'], ['freq'], ['freq'], [], []], kwargs={'tail': tail}, 
                               vectorize=True, output_core_dims=[['param']], output_dtypes=[float])
    Usdw_SL   = ust_param.isel(param=0)
    Uscw_SL   = ust_param.isel(param=1)
    Usdw_ref  = ust_param.isel(param=2)
    Uscw_ref  = ust_param.isel(param=3)
    Usdw0     = ust_param.isel(param=4)
    Uscw0     = ust_param.isel(param=5)
    Usdw_SLa  = ust_param.isel(param=6)
    Uscw_SLa  = ust_param.isel(param=7)
    Usdw_10ph = ust_param.isel(param=8)
    Uscw_10ph = ust_param.isel(param=9)
    Usdw_20ph = ust_param.isel(param=10)
    Uscw_20ph = ust_param.isel(param=11)
    
    Us_SL  = np.sqrt(Usdw_SL**2 + Uscw_SL**2)
    Us_ref = np.sqrt(Usdw_ref**2 + Uscw_ref**2)
    Us0    = np.sqrt(Usdw0**2 + Uscw0**2)
    
    theta_ww = np.arctan2(Uscw_SL, Usdw_SL)
    z1 = np.maximum(Hsww, 0.02)*4 # Van Roekle et al. 2012, Eq. (29) for alpha_LOW
    theta_wL = np.arctan2(np.sin(theta_ww), ustar/kappa/Us0*np.log(bld/z1) + np.cos(theta_ww))
    idx_noLOW = bld <= z1 # assume Langmuir cells align with wave direction when bld <= z1
    theta_wL = xr.where(idx_noLOW, theta_ww, theta_wL)
    
    # Stokes decay scale
    # theta_wave0 = np.arctan2(Uscw0, Usdw0) + theta_tau
    theta_wave = theta_ww + theta_tau
    stds = xr.apply_ufunc(sot.get_stds_d2f, dws.freq, us_d2f, vs_d2f, theta_wave,
                          input_core_dims=[['freq'], ['freq'], ['freq'], []], kwargs={'tail': tail}, 
                          vectorize=True, output_dtypes=[float])
    
    dftb['zb'] = -0.6*Hsww
    dftb['zt'] = -10*Hsww
    dftb['stds'] = stds
    dftb['Us0'] = Us0
    dftb['UsSL'] = Us_SL - Us_ref
    dftb['theta_ww'] = theta_ww
    dftb['theta_wL'] = theta_wL
    dftb['LaSLP2'] = ustar*np.cos(theta_wL) / ((Us_SL - Us_ref)*np.cos(theta_ww - theta_wL))
    dftb['LaSL2'] = ustar/(Usdw_SL - Usdw_ref)
    dftb['LaSSL2'] = ustar/(Usdw_SLa - Usdw_ref)
    dftb['Lat2'] = ustar/Usdw0
    
    # Stokes similarity function    
    dftb['xi'] = xr.apply_ufunc(sot.get_xi, Usdw0, Usdw_SLa, Usdw_10ph, ustar, wstar, bld, 
                                input_core_dims=[[], [], [], [], [], []], vectorize=True, output_dtypes=[float])
    dftb['chim'] = xr.apply_ufunc(sot.get_emp_chi, dftb.xi, input_core_dims=[[]], 
                                  kwargs={'var': 'mom'}, vectorize=True, output_dtypes=[float])
    
    # # stability lengths
    # Lstbl = xr.apply_ufunc(sot.get_stability_length_d2f, dws.freq, us_d2f, vs_d2f,  
    #                        theta_tau, ustar, dftb.wbf, #dftb.chim, 
    #                        input_core_dims=[['freq'], ['freq'], ['freq'], [], [], []],#, []],
    #                        kwargs={'tail': tail}, vectorize=True, output_core_dims=[['param']], output_dtypes=[float])
    # dftb['LOg'] = Lstbl.isel(param=0)
    # dftb['Lss'] = Lstbl.isel(param=1)
    # dftb['LHo'] = Lstbl.isel(param=2)
    # dftb['zoLOg'] = np.abs(dftb.mz)/dftb.LOg
    # dftb['zoLss'] = np.abs(dftb.mz)/dftb.Lss
    # dftb['zoLHo'] = np.abs(dftb.mz)/dftb.LHo
    
    # downwind Stokes shear
    dftb['UsdwshM'] = xr.apply_ufunc(sot.get_dUsdwdzM_d2f, dftb.mz, dftb.zstd, dws.freq, us_d2f, vs_d2f, theta_tau, 
                                     input_core_dims=[[], [], ['freq'], ['freq'], ['freq'], []], kwargs={'tail': tail}, 
                                     vectorize=True, output_dtypes=[float])
    dftb['SSP'] = ustar**2*(1+dftb.zoh)*dftb.UsdwshM
    
    # surface proximity function (1-fzS) decay depth
    dftb['Ls'] = xr.apply_ufunc(sot.get_Ls_d2f, dws.freq, us_d2f, vs_d2f, 
                                ustar, bld, theta_tau, 
                                input_core_dims=[['freq'], ['freq'], ['freq'], [], [], []],
                                vectorize=True, output_dtypes=[float])
    dftb['spf'] = 1 - xr.apply_ufunc(sot.get_fzSM, dftb.mz, dftb.zstd, dftb.Ls, 
                                     input_core_dims=[[], [], []], vectorize=True, output_dtypes=[float])
    
    nc_name = f'Fanapi_dftb_wave_{floatID}_{tau_opt}.nc'
    dftb.to_netcdf(nbf_dir + nc_name, engine='netcdf4', # encoding makes sure the boolean dtype
                   encoding={'Iequil': {'dtype': 'bool'}, 'Ishoal': {'dtype': 'bool'}, 'Ideepen': {'dtype': 'bool'},
                             'ifr': {'dtype': 'bool'}, 'ifl': {'dtype': 'bool'},})
    print(f'Binned float drift and wave data: {nc_name} saved at {nbf_dir}.')


if platform == 'linux':
    Groot = '/media/zhihua/internal1/GDrive/'
elif platform == 'darwin':
    Groot = '/Volumes/GoogleDrive/My Drive/'

tau_opt = 'APL'
_prep_fanapi_wave(Groot, 60, tau_opt=tau_opt, tail=True)
_prep_fanapi_wave(Groot, 61, tau_opt=tau_opt, tail=True)
_prep_fanapi_wave(Groot, 62, tau_opt=tau_opt, tail=True)
_prep_fanapi_wave(Groot, 64, tau_opt=tau_opt, tail=True)

# tau_opt = 'URI'
# _prep_fanapi_wave(Groot, 60, tau_opt=tau_opt, tail=False)
# _prep_fanapi_wave(Groot, 61, tau_opt=tau_opt, tail=False)
# _prep_fanapi_wave(Groot, 62, tau_opt=tau_opt, tail=False)
# _prep_fanapi_wave(Groot, 64, tau_opt=tau_opt, tail=False)