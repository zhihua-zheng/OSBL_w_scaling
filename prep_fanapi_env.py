import xarray as xr
import numpy as np
import pandas as pd
import sca_osbl_tool as sot
import scipy.io as sio
import gsw
from constants import g, nan
from sys import platform


def _crt_pressure(GDrive_root, floatID):
    site_root = GDrive_root + 'UW/Research/Data/Hurricanes/ITOP/'
    era5_dir = site_root + 'Reanalysis/ERA5/'
    nbf_dir = site_root + 'NBF/'
    env_name = 'Env_Fanapi_'+str(floatID)+'.mat'
    FPR_name = 'FPR/FPR'+str(floatID)+'.mat'
    year = 2010
    p_calconst = 1e5
    
    ep = sio.loadmat(nbf_dir+env_name, squeeze_me=True, struct_as_record=False)
    pytime = sot.yd2pytime(ep['yd'], year)
    
    pf = sio.loadmat(nbf_dir+FPR_name, squeeze_me=True, struct_as_record=False)
    # trim fast pressure to just cover Env time
    if pf['ydP'][-1] > ep['yd'][-1]:
        last_pf = np.where( pf['ydP'] >= ep['yd'][-1] )[0][0] + 1
        for key in pf.keys():
            if not key.startswith('__'):
                pf[key] = pf[key][:last_pf]
    pytimeP = sot.yd2pytime(pf['ydP'], year)
    
    # remove atmopsheric pressure signal
    flon = xr.DataArray(data=ep['lonnav'], dims=['time'], coords={'time': pytime})\
             .interpolate_na('time', fill_value='extrapolate')
    flat = xr.DataArray(data=ep['latnav'], dims=['time'], coords={'time': pytime})\
             .interpolate_na('time', fill_value='extrapolate')
    with xr.open_dataset(era5_dir+'fanapi_surface_hourly_Sep2010.nc') as ds:
        bp = ds.sp.assign_coords({'time': ('time', ds.time.values - pd.Timedelta(minutes=30))})
        bp = bp.interp(time=xr.DataArray(data=pytimeP, dims=['time']), 
                       latitude=flat.interp(time=pytimeP), longitude=flon.interp(time=pytimeP))
        aPair = (bp - p_calconst)/1e4 # [dbar]
    pf['P0'] = pf['P0'] - aPair.values
    # P1 too noisy
    
    # zeroing pressure according to float geomerty
    P0_surf_mean = np.around(np.mean(pf['P0'][pf['P0']<0.5]),3)
    print(f'The top pressure sensor is {P0_surf_mean} m below the surface on average.')
    P0_offset = np.around(0.461-P0_surf_mean, 3)
    print(f'Applying offset of {P0_offset} m to the top pressure record.')
    P0 = pf['P0'] + P0_offset
    
    # drift = np.full_like(ep['P'], nan)
    # idx_dft = (ep['yd']>=260.2675) & (ep['yd']<=261.26)
    # drift[idx_dft] = 1
    # idx_dft = (ep['yd']>=261.32) & (ep['yd']<=262.32344)
    # drift[idx_dft] = 2
    
    Sig0P = 1e3 + np.interp(pf['ydP'], ep['yd'], ep['Sig0']) # potential density (Unesco 1983)
    Pc = pf['P0'] + Sig0P*g*0.445/1e4 # hydrostatically shift to the float's geomertic center, Pa to dbar
    Pts = pf['P0'] + Sig0P*g*1.15/1e4 # hydrostatically shift to the float's bottom CT sensor, Pa to dbar
    P = np.interp(ep['yd'], pf['ydP'], Pts) # try time averaging?
    
    # only use the first few days' data, the rest didn't sample boundary layer turbulence
    idx_blt = np.where(ep['up']==1)[0][-1] + 1
    idxP_blt = np.where(pf['ydP']<ep['yd'][idx_blt])[0][-1] + 1
    
    env = xr.Dataset(
        data_vars={'P0': (('timeP'), pf['P0'][:idxP_blt], {'units': 'dbar', 'standard_name': 'fast_pressure_top'}),
                   'Pc': (('timeP'), Pc[:idxP_blt], {'units': 'dbar', 'standard_name': 'fast_fake_pressure_center'}),
                   'P': (('time'), P[:idx_blt], {'units': 'dbar', 'standard_name': 'fake_pressure_for_TS'}),
                   'T': (('time'), ep['T'][:idx_blt], {'units': 'C', 'standard_name': 'temperature'}),
                   'S': (('time'), ep['S'][:idx_blt], {'units': 'psu', 'standard_name': 'salinity'}),
                   'B': (('time'), ep['B'][:idx_blt], {'units': 'cubic centimeter (cc)', 'standard_name': 'float_piston_volume'}),
                   'mode': (('time'), ep['mode'][:idx_blt], {'standard_name': 'float_operating_mode'}),
                   'drift': (('time'), ep['drift'][:idx_blt], {'standard_name': 'Lagrangian_drift_number'}),
                   'settle': (('time'), ep['settle'][:idx_blt], {'standard_name': 'settle_number'}),
                   'up': (('time'), ep['up'][:idx_blt], {'standard_name': 'upward_profile_number'}),
                   'down': (('time'), ep['down'][:idx_blt], {'standard_name': 'downward_profile_number'}),
                   'lon': (('time'), flon.data[:idx_blt], {'units': 'degree', 'standard_name': 'longitude_interpolated'}),
                   'lat': (('time'), flat.data[:idx_blt], {'units': 'degree', 'standard_name': 'latitude_interpolated'})},
        coords={'ydP': (('timeP'), pf['ydP'][:idxP_blt], {'standard_name': 'fast_yearday_of_'+str(year)}),
                'timeP': (('timeP'), pytimeP[:idxP_blt]),
                'yd': (('time'), ep['yd'][:idx_blt], {'standard_name': 'yearday_of_'+str(year)}),
                'time': (('time'), pytime[:idx_blt])},
        attrs={'description': 'Raw data from the Lagrangian float ' + str(floatID) + ' deployed in ' + str(year) + ' under Typhoon Fanapi',
               'note': 'All pressures have been corrected for atmospheric pressure and calibration offsets'})
    
    env = env.dropna('timeP', how='all')
    nc_name = 'Fanapi_env_' + str(floatID) + '.nc'
    env.to_netcdf(nbf_dir + nc_name, engine='netcdf4')
    print(f'Float data: {nc_name} saved at {nbf_dir}.\n')


if platform == 'linux':
    Groot = '/media/zhihua/internal1/GDrive/'
elif platform == 'darwin':
    Groot = '/Volumes/GoogleDrive/My Drive/'
_crt_pressure(Groot, 60)
_crt_pressure(Groot, 61)
_crt_pressure(Groot, 62)
_crt_pressure(Groot, 64)