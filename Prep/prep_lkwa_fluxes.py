import xarray as xr
import numpy as np
import pandas as pd
import sca_osbl_tool as sot
import scipy.io as sio
import gsw
from constants import g, cp, rho_fw, pi
from sys import platform


if platform == 'linux':
    GDrive_root = '/media/zhihua/internal1/GDrive/'
elif platform == 'darwin':
    GDrive_root = '/Volumes/GoogleDrive/My Drive/'
lkwa_root = GDrive_root + 'UW/Research/Data/LakeWA/'
moor_dir = lkwa_root + 'Mooring/'
wave_dir = lkwa_root + 'Wave/'

frc = sio.loadmat(moor_dir+'LkWash_forcing.mat', squeeze_me=True, struct_as_record=False)
pytime = sot.mtime2pytime(frc['time'])
yd = sot.pytime2yd(pytime)
lat = [47.6118]
lon = [-122.2613]

# # wave data
# cdip = xr.open_dataset(wave_dir+'177p1_historic.nc').load() \
#          .drop_vars(['metaStationLatitude', 'metaStationLongitude']) \
#          .rename({'waveTime': 'time', 'waveFrequency': 'freq'})
# cdip.close()
# unitx_wave_eq = -cdip.waveB1Value[:,-10:].mean('freq')
# unity_wave_eq = -cdip.waveA1Value[:,-10:].mean('freq')
# # equilibrium wave mean direction [degree, anti-clockwise referenced to the East]
# theta_wave_eq = np.arctan2(unity_wave_eq, unitx_wave_eq)/pi*180%360

# mixed layer averaged Chlorophyll concentration
waq = sio.loadmat(moor_dir+'LKWA_Chl_Tbd_2011.mat', squeeze_me=True, struct_as_record=False)
Chltime = sot.mtime2pytime(waq['timeprof'])
Chl = np.interp(yd, sot.pytime2yd(Chltime), waq['mChl'])

Tprof = frc['Tprof']
depth = frc['z_prof'].astype('float64')
SPprof = np.ones_like(Tprof)*0.05
SAprof = gsw.SA_from_SP(SPprof, depth[:,None], lon, lat)
PTprof = gsw.pt0_from_t(SAprof, Tprof, depth[:,None])

iCOARE3_Smith = frc['ustar_source'] == 'COARE 3.0+Smith'
ustar = np.squeeze(frc['ustar'][iCOARE3_Smith,:])
taum = ustar**2*rho_fw
# wind direction [degree, anti-clockwise referenced to the East]
wdir = np.angle(frc['W'])
idx_wdir_gap = np.isnan(wdir) & (pytime>pd.Timestamp('2011-11-21'))
wdir[idx_wdir_gap] = np.interp(yd[idx_wdir_gap], yd[~idx_wdir_gap], wdir[~idx_wdir_gap])
# # fill wind direction gap with equilibrium wave mean direction
# wdir_gap_time = pytime[idx_wdir_gap]
# wdir[idx_wdir_gap] = theta_wave_eq.interp(time=wdir_gap_time)
taux = taum*np.cos(wdir)
tauy = taum*np.sin(wdir)

isonic = frc['WSPD_source'] == 'Sonic'
wsp = np.squeeze(frc['WSPD'][isonic,:])

# all positive heat fluxes are into the ocean
nsw = frc['HF_shortwave']
nlw = frc['HF_longwave']
hlb = np.squeeze(frc['HF_latent'][iCOARE3_Smith,:])
hsb = np.squeeze(frc['HF_sensible'][iCOARE3_Smith,:])
sst = frc['SST']
sss = np.full_like(sst, 0)
Le = gsw.latentheat_evap_t(sss, sst) # heat of evaporation [J/kg]
evap = -hlb/Le/rho_fw*1e3*3600 # [mm/hr], positive mostly
rain = np.full_like(evap, 0)
emp = evap - rain

# thermodynamics
ssSA  = np.full_like(sst, 0)
ssCT  = gsw.CT_from_t(ssSA, sst, 1)
alpha = gsw.density.alpha(ssSA, ssCT, 1)
beta  = gsw.density.beta(ssSA, ssCT, 1)

# Salinity flux
ws0 = -(emp*1e-3/3600*sss)

# Combine heat fluxes
hsrf = nlw + hlb + hsb
hnet = nsw + hsrf
wt0 = -hsrf/rho_fw/cp

# Buoyancy fluxes [m^2/s^3]
wb0 = g*(alpha*wt0 - beta*ws0)
wbnet_t = -g*alpha*(hnet/rho_fw/cp)
wbnet_s = -g*beta*ws0
wbnet = wbnet_t + wbnet_s

fluxes = xr.Dataset(
    data_vars={'taux': (('time'), taux, {'units': 'N/m2', 'long_name': 'ZONAL WIND STRESS', 'method': 'COARE 3.0 + Smith (default)'}),
               'tauy': (('time'), tauy, {'units': 'N/m2', 'long_name': 'MERID WIND STRESS', 'method': 'COARE 3.0 + Smith (default)'}),
               'ustar': (('time'), ustar, {'units': 'm/s', 'long_name': 'FRICTION VELOCITY', 'method': 'COARE 3.0 + Smith (default)'}),
               'wsp': (('time'), wsp, {'units': 'm/s', 'long_name': 'WIND SPEED (M/S)', 'depth': '-3.4 m', 'source': 'sonic anemometer'}),
               'nsw': (('time'), nsw, {'units': 'W/m2', 'long_name': 'NET SHORTWAVE RADIATION'}),
               'nlw': (('time'), nlw, {'units': 'W/m2', 'long_name': 'NET LONGWAVE RADIATION'}),
               'hlb': (('time'), hlb, {'units': 'W/m2', 'long_name': 'LATENT HEAT FLUX', 'method': 'COARE 3.0 + Smith (default)'}),
               'hsb': (('time'), hsb, {'units': 'W/m2', 'long_name': 'SENSIBLE HEAT FLUX', 'method': 'COARE 3.0 + Smith (default)'}),
               'hsrf': (('time'), hsrf, {'units': 'W/m2', 'long_name': 'SURFACE HEAT FLUX (NO SW)'}),
               'hnet': (('time'), hnet, {'units': 'W/m2', 'long_name': 'SURFACE HEAT FLUX'}),
               'sst': (('time'), sst, {'units': 'C', 'long_name': 'SEA SURFACE TEMPERATURE', 'depth': '1 m'}),
               'sss': (('time'), sss, {'units': 'PSU', 'long_name': 'SEA SURFACE SALINITY'}),
               'evap': (('time'), evap, {'units': 'MM/HR', 'long_name': 'EVAPORATION'}),
               'rain': (('time'), rain, {'units': 'MM/HR', 'long_name': 'PRECIPITATION'}),
               'emp': (('time'), emp, {'units': 'MM/HR', 'long_name': 'EVAPORATION - PRECIPITATION'}),
               'ws0': (('time'), ws0, {'units': 'PSU m/s', 'long_name': 'KINEMATIC SALT FLUX'}),
               'wt0': (('time'), wt0, {'units': 'C m/s', 'long_name': 'KINEMATIC TEMP FLUX (NO SW)'}),
               'wb0': (('time'), wb0, {'units': 'm2 s-3', 'long_name': 'KINEMATIC BUOY FLUX (NO SW)'}),
               'wbnet_t': (('time'), wbnet_t, {'units': 'm2 s-3', 'long_name': 'KINEMATIC BUOY FLUX (T PART)'}),
               'wbnet_s': (('time'), wbnet_s, {'units': 'm2 s-3', 'long_name': 'KINEMATIC BUOY FLUX (S PART)'}),
               'wbnet': (('time'), wbnet, {'units': 'm2 s-3', 'long_name': 'KINEMATIC BUOY FLUX'}),
               'Chl': (('time'), Chl, {'units': 'mg m-3', 'long_name': 'MIXED LAYER MEAN CHLOROPHYLL'})},
    coords={'yd': (('time'), yd, {'long_name': 'yearday_of_2011'}),
            'time': (('time'), pytime),
            'depth': (('depth'), [0]),
            'lon': (('lon'), lon, {'units': 'degree', 'long_name': 'longitude'}),
            'lat': (('lat'), lat, {'units': 'degree', 'long_name': 'latitude'})},
    attrs={'description': 'Surface forcing for Lake Washington (2011) merged from various sources',
           'note': 'Copied from the data file created by Andrey Shcherbina'})

# time array in fluxes has small deviations (<< 1s) from integer hours
int_hr = pd.date_range(start=fluxes.time[0].dt.round('1H').values,
                       end=fluxes.time[-1].dt.round('1H').values, freq='1H')
fluxes = fluxes.interp(time=int_hr).sel(time=slice('2011-10-01', '2011-12-01'))
nc_name = 'LKWA_fluxes.nc'
fluxes.to_netcdf(moor_dir + nc_name, engine='netcdf4')
print(f'Air-sea fluxes: {nc_name} saved at {moor_dir}.\n')


TSmoor = xr.Dataset(
    data_vars={'PTprof': (('depth', 'time'), PTprof, {'units': 'C', 'long_name': 'POTENTIAL TEMPERATURE'}),
               'SPprof': (('depth', 'time'), SPprof, {'units': 'psu', 'long_name': 'PRACTICAL SALINITY'})},
    coords={'yd': (('time'), yd, {'long_name': 'yearday_of_2011'}),
            'time': (('time'), pytime),
            'depth': (('depth'), depth),
            'lon': (('lon'), lon, {'units': 'degree', 'long_name': 'longitude'}),
            'lat': (('lat'), lat, {'units': 'degree', 'long_name': 'latitude'})},
    attrs={'description': 'Profiles of T-S from the south Lake Washington mooring (2011)',
           'note': 'Copied from the data file created by Andrey Shcherbina. Salinity is assumed zero.'})

TSmoor = TSmoor.sel(time=slice('2011-10-01', '2011-12-01'))
nc_name = 'LKWA_TSmoor.nc'
TSmoor.to_netcdf(moor_dir + nc_name, engine='netcdf4')
print(f'Mooring profiles: {nc_name} saved at {moor_dir}.\n')