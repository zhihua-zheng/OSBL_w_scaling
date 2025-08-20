import xarray as xr
import numpy as np
import pandas as pd
import gsw
from sys import platform
from scipy import integrate


def _prep_ocsp_TSgotm(GDrive_root, year):
    site_root = GDrive_root + 'UW/Research/Data/OCSP/'
    nbf_dir = site_root + 'NBF/'

    if year==2011:
        depth_max = 120
    elif year==2012:
        depth_max = 60
    else:
        raise ValueError('Year number not supprted!')

    with xr.open_dataset(nbf_dir+'OCSP_gridded_'+str(year)+'.nc') as ds:
        SPprof = ds.SPprof.values
        PTprof = ds.PTprof.values
        zprof = -ds.depth.values
        timeprof = ds.time.values

    with xr.open_dataset(nbf_dir+'OCSP_env_'+str(year)+'.nc') as ds:
        drift = ds.drift.values
        timedrift = ds.time.values

    # Profile index before each drift
    Ndrift = np.nanmax(drift) + 1 # starts from 0
    Iprof = []
    for i in np.arange(Ndrift):
        drift_index = drift == i
        iprof = np.where(timeprof < timedrift[drift_index][0])[0][-1]
        Iprof.append(iprof)

    # Extrapolate short profiles using previous gradients
    for i,ipf in enumerate(Iprof):
        df = pd.DataFrame(np.column_stack([PTprof[:,ipf], SPprof[:,ipf], zprof]),
                          columns=['PT', 'SP', 'Z']).set_index('Z')
        # index for last/first non-NAN value or None, if no NAN value is found
        last_valid_z = df.apply(pd.Series.last_valid_index, axis=0)
        first_valid_z = df.apply(pd.Series.first_valid_index, axis=0)
        if i == 0:
            ipre = 0
            ipost = Iprof[i+1]+1
        elif i == Ndrift-1:
            ipre = Iprof[i-1]
            ipost = len(timeprof)
        else:
            ipre = Iprof[i-1]
            ipost = Iprof[i+1]+1

        if last_valid_z.PT > -depth_max:
            PTdf = pd.DataFrame(PTprof[:,ipre:ipost], index=zprof, columns=timeprof[ipre:ipost]-timeprof[ipf])
            # index for last non-NAN value per column
            zlowest = PTdf.apply(pd.Series.last_valid_index, axis=0)
            try:
                ibackward = np.where((zlowest < -depth_max) & (zlowest.index < pd.Timedelta(days=0)))[0][-1]
            except:
                ibackward = 0
            try:
                iforward = np.where((zlowest < -depth_max) & (zlowest.index > pd.Timedelta(days=0)))[0][0] + 1
            except:
                iforward = PTdf.shape[1]
            PTdf = PTdf.iloc[:,ibackward:iforward]
            dPTdzdf = pd.DataFrame(np.gradient(PTdf, PTdf.index, axis=0), index=zprof, columns=PTdf.columns)
            # interp in time
            # dPTdzdf.interpolate(method='time', axis=1).loc[last_valid_z.PT:, pd.Timedelta(days=0)]
            dPTdz_mean = dPTdzdf.mean(axis=1).loc[last_valid_z.PT:]
            PTint = integrate.cumulative_trapezoid(dPTdz_mean.values, dPTdz_mean.index.values,
                                                   initial=0) + df.PT.loc[last_valid_z.PT]
            df.PT.loc[last_valid_z.PT:] = PTint
            PTprof[:,ipf] = df.PT.values

        if first_valid_z.PT < -2:
            PTdf = pd.DataFrame(PTprof[:,ipre:ipost], index=zprof, columns=timeprof[ipre:ipost]-timeprof[ipf])
            # index for first non-NAN value per column
            zhighest = PTdf.apply(pd.Series.first_valid_index, axis=0)
            try:
                ibackward = np.where((zhighest > -2) & (zhighest.index < pd.Timedelta(days=0)))[0][-1]
            except:
                ibackward = 0
            try:
                iforward = np.where((zhighest > -2) & (zhighest.index > pd.Timedelta(days=0)))[0][0] + 1
            except:
                iforward = PTdf.shape[1]
            PTdfr = PTdf.iloc[::-1,ibackward:iforward]
            dPTdzdfr = pd.DataFrame(np.gradient(PTdfr, PTdfr.index, axis=0), index=PTdfr.index, columns=PTdfr.columns)
            dPTdz_mean = dPTdzdfr.mean(axis=1).loc[first_valid_z.PT:]
            PTint = integrate.cumulative_trapezoid(dPTdz_mean.values, dPTdz_mean.index.values,
                                                   initial=0) + df.PT.loc[first_valid_z.PT]
            df.PT.loc[:first_valid_z.PT] = np.flip(PTint) # iloc includes the right edge
            PTprof[:,ipf] = df.PT.values

        if last_valid_z.SP > -depth_max:
            SPdf = pd.DataFrame(SPprof[:,ipre:ipost], index=zprof, columns=timeprof[ipre:ipost]-timeprof[ipf])
            zlowest = SPdf.apply(pd.Series.last_valid_index, axis=0)
            try:
                ibackward = np.where((zlowest < -depth_max) & (zlowest.index < pd.Timedelta(days=0)))[0][-1]
            except:
                ibackward = 0
            try:
                iforward = np.where((zlowest < -depth_max) & (zlowest.index > pd.Timedelta(days=0)))[0][0] + 1
            except:
                iforward = SPdf.shape[1]
            SPdf = SPdf.iloc[:,ibackward:iforward]
            dSPdzdf = pd.DataFrame(np.gradient(SPdf, SPdf.index, axis=0), index=zprof, columns=SPdf.columns)
            dSPdz_mean = dSPdzdf.mean(axis=1).loc[last_valid_z.SP:]
            SPint = integrate.cumulative_trapezoid(dSPdz_mean.values, dSPdz_mean.index.values,
                                                   initial=0) + df.SP.loc[last_valid_z.SP]
            df.SP.loc[last_valid_z.SP:] = SPint
            SPprof[:,ipf] = df.SP.values

        if first_valid_z.SP < -2:
            SPdf = pd.DataFrame(SPprof[:,ipre:ipost], index=zprof, columns=timeprof[ipre:ipost]-timeprof[ipf])
            zhighest = SPdf.apply(pd.Series.first_valid_index, axis=0)
            try:
                ibackward = np.where((zhighest > -2) & (zhighest.index < pd.Timedelta(days=0)))[0][-1]
            except:
                ibackward = 0
            try:
                iforward = np.where((zhighest > -2) & (zhighest.index > pd.Timedelta(days=0)))[0][0] + 1
            except:
                iforward = SPdf.shape[1]
            SPdfr = SPdf.iloc[::-1,ibackward:iforward]
            dSPdzdfr = pd.DataFrame(np.gradient(SPdfr, SPdfr.index, axis=0), index=SPdfr.index, columns=SPdfr.columns)
            dSPdz_mean = dSPdzdfr.mean(axis=1).loc[first_valid_z.SP:]
            SPint = integrate.cumulative_trapezoid(dSPdz_mean.values, dSPdz_mean.index.values,
                                                   initial=0) + df.SP.loc[first_valid_z.SP]
            df.SP.loc[:first_valid_z.SP] = np.flip(SPint)
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
        attrs={'description': 'Profiles before drift, from the Lagrangian float deployed in '+str(year)+' near ocean climate station Papa',
               'note': 'Short profiles are extrpolated downward/upward using mean gradients'})

    nc_name = 'OCSP_TSgotm_'+str(year)+'.nc'
    TSgotm.to_netcdf(nbf_dir + nc_name, engine='netcdf4')
    print(f'GOTM profile data: {nc_name} saved at {nbf_dir}.\n')


if platform == 'linux':
    Groot = '/media/zhihua/internal1/GDrive/'
elif platform == 'darwin':
    Groot = '/Volumes/GoogleDrive/My Drive/'
_prep_ocsp_TSgotm(Groot, 2011)
_prep_ocsp_TSgotm(Groot, 2012)