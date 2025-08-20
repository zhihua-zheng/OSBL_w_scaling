import numpy as np
import pandas as pd
import xarray as xr
import sca_osbl_tool as sot
from sys import platform
from constants import nan


def nan_percent(ds):
    nan_perc = []
    for idf in ds.ID.values:
        ids = ds.sel(ID=idf).dropna('time', how='any')
        coverage = (ids.p.size-1)/((ids.ftime[-1] - ids.ftime[0])/pd.Timedelta('5min'))
        nan_perc.append(1 - coverage.clip(min=0, max=1))
    return xr.concat(nan_perc, dim='ID').rename('nan_percent')


if platform == 'linux':
    GDrive_root = '/media/zhihua/internal1/GDrive/'
elif platform == 'darwin':
    GDrive_root = '/Volumes/GoogleDrive/My Drive/'

data_root = GDrive_root + 'UW/Research/Data/LabSea/'
nbf_dir = data_root + 'NBF/'

ds = xr.open_dataset(nbf_dir+'dataset20.cdf').load().astype('float64')
ds.close()

ds = ds.rename_dims({'idx1': 'time', 'idx2': 'timePOS'})
tcomps_str = ['year','month','day','hour','minute','second']

# assemble datetimes
ftime = sot.tcomps2pytime(ds['t-year'],ds['t-month'],ds['t-day'],ds['t-hour'],ds['t-minute'],ds['t-second'])
ds = ds.assign_coords({'ftime': (('ID', 'time'), ftime, {'longname': 'Float time'})})
fyear = [np.unique(np.ma.masked_invalid(a)).compressed()[0] for a in list(ds['t-year'])]
ds = ds.assign({'fyear': ('ID', fyear, {'longname': 'Float year'})})
ds = ds.assign_coords(fyd = (('ID', 'time'), sot.pytime2yd(ds.ftime), {'longname': 'Float yearday'}))
ds = ds.drop_vars(['t-'+i for i in tcomps_str])

time_POS = sot.tcomps2pytime(ds['pos-year'],ds['pos-month'],ds['pos-day'],ds['pos-hour'],ds['pos-minute'],ds['pos-second'])
ds = ds.assign_coords({'time_POS': (('ID', 'timePOS'), time_POS)})
ds = ds.drop_vars(['pos-'+i for i in tcomps_str])

# remove float with scarce data
p_nan_perc = nan_percent(ds)
failed_float = (p_nan_perc > 0.5) & (ds.lon.notnull().sum('timePOS') == 0)
ds = ds.where(~failed_float, drop=True)

# interp hourly positions
# make sure the position data covers the end time of pressure measurements in 1997 (floats end with profiling in 1998);
# gaps at the beginning is fine, as the drift didn't start until a week later.
hhtime = pd.Timedelta(minutes=30)
ds97 = ds.sel(ID=ds.ID[ds.fyear==1997]).dropna('time', how='all')
timePOS_start = ds97.time_POS[:,0].min(skipna=True).dt.round('1H').values
timePOS_end = ds97.time_POS.dropna('timePOS', how='all')[:,-1].max(skipna=True).dt.ceil('1H')
ftime_end = ds97.ftime[:,-1].max(skipna=True).dt.ceil('1H')
timePOS_end = xr.concat([ftime_end, timePOS_end], dim='time').max().values
timePOS_hr97 = pd.date_range(timePOS_start, timePOS_end, freq='1H')
lonhr = np.full((ds97.dims['ID'], timePOS_hr97.size), nan)
lathr = np.full((ds97.dims['ID'], timePOS_hr97.size), nan)
for i,idf in enumerate(ds97.ID.values):
    iPos97 = ds97[['lon', 'lat']].sel(ID=idf).set_index({'timePOS': 'time_POS'}).dropna('timePOS', how='any')
    if iPos97.dims['timePOS'] > 1:
        lonhr[i,:] = iPos97.lon.resample(timePOS='1H').interpolate('linear').interp(timePOS=timePOS_hr97)
        lathr[i,:] = iPos97.lat.resample(timePOS='1H').interpolate('linear').interp(timePOS=timePOS_hr97)
ds97 = ds97.drop_vars(['lon', 'lat', 'time_POS'])
ds97 = ds97.assign({'lon': (('ID', 'timePOS'), lonhr), 'lat': (('ID', 'timePOS'), lathr)})
ds97 = ds97.assign_coords({'timePOS': ('timePOS', timePOS_hr97, {'longname': 'Float track time'})})
# for floats with no GPS fixes, use the mean track of other floats before they diverge.
# the float divergence occurs when any float separation (either lon or lat difference) is larger than 0.125 degree,
# which is half the ERA5 forcing grid size
# float 16 has the longest GPS track
idx_diverge_lon = np.where(((ds97.lon.T-ds97.lon.sel(ID=16).T) > 0.125).any('ID'))[0][0]
idx_diverge_lat = np.where(((ds97.lat.T-ds97.lat.sel(ID=16).T) > 0.125).any('ID'))[0][0]
idx_diverge = np.minimum(idx_diverge_lon, idx_diverge_lat)
time_diverge = ds97.timePOS[idx_diverge].data
mlon = ds97.lon.mean('ID').where(ds97.timePOS<time_diverge)
mlat = ds97.lat.mean('ID').where(ds97.timePOS<time_diverge)
ds97['lon'] = ds97.lon.fillna(value=mlon)
ds97['lat'] = ds97.lat.fillna(value=mlat)


ds98 = ds.sel(ID=ds.ID[ds.fyear==1998]).dropna('time', how='all')
timePOS_start = ds98.time_POS[:,0].min(skipna=True).dt.round('1H').values
timePOS_end = ds98.time_POS.dropna('timePOS', how='all')[:,-1].max(skipna=True).dt.ceil('1H')
ftime_end = ds98.ftime[:,-1].max(skipna=True).dt.ceil('1H')
timePOS_end = xr.concat([ftime_end, timePOS_end], dim='time').max().values
timePOS_hr98 = pd.date_range(timePOS_start, timePOS_end, freq='1H')
lonhr = np.full((ds98.dims['ID'], timePOS_hr98.size), nan)
lathr = np.full((ds98.dims['ID'], timePOS_hr98.size), nan)
for i,idf in enumerate(ds98.ID.values):
    iPos98 = ds98[['lon', 'lat']].sel(ID=idf).set_index({'timePOS': 'time_POS'}).dropna('timePOS', how='any')
    if iPos98.dims['timePOS'] > 1:
        lonhr[i,:] = iPos98.lon.resample(timePOS='1H').interpolate('linear').interp(timePOS=timePOS_hr98)
        lathr[i,:] = iPos98.lat.resample(timePOS='1H').interpolate('linear').interp(timePOS=timePOS_hr98)
ds98 = ds98.drop_vars(['lon', 'lat', 'time_POS'])
ds98 = ds98.assign({'lon': (('ID', 'timePOS'), lonhr), 'lat': (('ID', 'timePOS'), lathr)})
ds98 = ds98.assign_coords({'timePOS': ('timePOS', timePOS_hr98, {'longname': 'Float track time'})})
# float 30 has the longest GPS track
idx_diverge_lon = np.where(((ds98.lon.T-ds98.lon.sel(ID=30).T) > 0.125).any('ID'))[0][0]
idx_diverge_lat = np.where(((ds98.lat.T-ds98.lat.sel(ID=30).T) > 0.125).any('ID'))[0][0]
idx_diverge = np.minimum(idx_diverge_lon, idx_diverge_lat)
time_diverge = ds98.timePOS[idx_diverge].data
mlon = ds98.lon.mean('ID').where(ds98.timePOS<time_diverge)
mlat = ds98.lat.mean('ID').where(ds98.timePOS<time_diverge)
ds98['lon'] = ds98.lon.fillna(value=mlon)
ds98['lat'] = ds98.lat.fillna(value=mlat)


# identify drift times
# the first week is autoballast; last few days in 1998 (picked by eye) are profiles
driftmask97 = ds97.ftime > (ds97.ftime[:,0] + pd.Timedelta(days=7))
ds97 = ds97.assign(drift = driftmask97)

driftmask98 = ds98.ftime > (ds98.ftime[:,0] + pd.Timedelta(days=7))
endprof98 = ds98.ftime > pd.Timestamp('1998-03-21 00:00:00') # see Steffan & D'Asaro 2002 [page 484, Feb day 48 (02-01 as day 0)]
ds98 = ds98.assign(drift = (driftmask98 & ~endprof98))

nc_name = 'LabSea_env.nc'
ds.to_netcdf(nbf_dir + nc_name, engine='netcdf4')
print(f'Combined float data: {nc_name} saved at {nbf_dir}.\n')

nc_name97 = 'LabSea_env_1997.nc'
nc_name98 = 'LabSea_env_1998.nc'
ds97.to_netcdf(nbf_dir + nc_name97, engine='netcdf4')
print(f'Float data: {nc_name97} saved at {nbf_dir}.\n')
ds98.to_netcdf(nbf_dir + nc_name98, engine='netcdf4')
print(f'Float data: {nc_name98} saved at {nbf_dir}.\n')