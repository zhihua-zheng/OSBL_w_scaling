import xarray as xr
import numpy as np
import pandas as pd
import sca_osbl_tool as sot
import gsw
from sys import platform
from scipy import integrate


def _prep_fanapi_TSgotm(GDrive_root, floatID):
    site_root = GDrive_root + 'UW/Research/Data/LakeWA/'
    moor_dir = site_root + 'Mooring/'
    nbf_dir = site_root + 'NBF/'
    year = 2011
    depth_max = 40

    with xr.open_dataset(nbf_dir+'LKWA_gridded_'+str(floatID)+'.nc') as gd:
        SPprof = gd.SPprof.values
        PTprof = gd.PTprof.values
        zprof = -gd.depth.values
        timeprof = gd.time.values

    with xr.open_dataset(nbf_dir+'LKWA_env_'+str(floatID)+'.nc') as ds:
        drift = ds.drift.values
        timedrift = ds.time.values

    TSmoor = xr.open_dataset(moor_dir+'LKWA_TSmoor.nc').load()
    TSmoor.close()
    
    # Fill the missing lower profile using mooring data
    time_even = pd.date_range(start='10/25/2011', end='11/29/2011', freq='H')
    for d in np.arange(25,53,2):
        Tmd = TSmoor.PTprof.sel(depth=d).dropna(dim='time')
        Tfd = gd.PTprof.sel(depth=d).dropna(dim='time')
        idx_nan = np.isnan(gd.PTprof.sel(depth=d).values)
        PTprof[zprof==-d,idx_nan] = xr.merge((Tfd,Tmd), join='outer').interp(time=time_even) \
                                      .rolling(time=24, min_periods=6, center=True).median() \
                                      .interp(time=timeprof[idx_nan]).PTprof

        Sfd = gd.SPprof.sel(depth=d).dropna(dim='time')
        idx_nan = np.isnan(gd.SPprof.sel(depth=d).values)
        SPprof[zprof==-d,idx_nan] = Sfd.interp(time=time_even, kwargs={'fill_value': (Sfd.values[0], Sfd.values[-1])}) \
                                       .rolling(time=24, min_periods=6, center=True).median() \
                                       .interp(time=timeprof[idx_nan])

    # Profile index before each drift
    _,Idrift = sot.get_clumps(drift, min_len=360)
    Iprof = []
    for i in Idrift:
        iprof = np.where(timeprof < timedrift[i][0])[0][-1]
        Iprof.append(iprof)

    # Interpolate gaps
    for i,ipf in enumerate(Iprof):
        df = pd.DataFrame(np.column_stack([PTprof[:,ipf], SPprof[:,ipf], -zprof]),
                          columns=['PT', 'SP', 'Z']).set_index('Z') \
               .interpolate(method='pchip', limit_area='inside', limit=8, limit_direction='both') \
               .interpolate(method='nearest', fill_value='extrapolate', limit=8, limit_area='outside',
                            limit_direction='backward')
    PTprof[:,ipf] = df.PT.values
    SPprof[:,ipf] = df.SP.values

    # Repeat the last profile to the end of time series (only initial profiles are actually used)
    timeprof = np.append(timeprof, timedrift[-1] + np.timedelta64(300, 's'))
    PTprof = np.append(PTprof, PTprof[:,Iprof[-1]][:,None], axis=1)
    SPprof = np.append(SPprof, SPprof[:,Iprof[-1]][:,None], axis=1)

    Iprof = Iprof + [len(timeprof)-1]
    TSgotm = xr.Dataset(
        data_vars={'PTprof': (('depth', 'time'), PTprof[:,Iprof], {'units': 'C', 'standard_name': 'potential_temperature_profiles'}),
                   'SPprof': (('depth', 'time'), SPprof[:,Iprof], {'units': 'psu', 'standard_name': 'practical_salinity_profiles'})},
        coords={'time': (('time'), timeprof[Iprof]),
                'depth': (('depth'), -zprof, {'standard_name': 'profile_depth'})},
        attrs={'description': 'Profiles before drift, from the Lagrangian float '+str(floatID)+' deployed in Lake Washington (2011)',
               'note': 'Short profiles are extrpolated downward/upward using mean gradients'})

    nc_name = 'LKWA_TSgotm_'+str(floatID)+'.nc'
    TSgotm.to_netcdf(nbf_dir + nc_name, engine='netcdf4')
    print(f'GOTM profile data: {nc_name} saved at {nbf_dir}.\n')


if platform == 'linux':
    Groot = '/media/zhihua/internal1/GDrive/'
elif platform == 'darwin':
    Groot = '/Volumes/GoogleDrive/My Drive/'
_prep_lkwa_TSgotm(Groot, 71)
_prep_lkwa_TSgotm(Groot, 72)