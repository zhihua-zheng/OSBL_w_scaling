import xarray as xr
import numpy as np
import pandas as pd
import sca_osbl_tool as sot
import scipy.io as sio
import gsw
from constants import g, cp, rho0
from sys import platform


def _prep_ocsp_fluxes(GDrive_root, year):
    ocsp_root = GDrive_root + 'UW/Research/Data/OCSP/'
    moor_dir = ocsp_root + 'Mooring/'
    flux_dir = moor_dir + str(year) + '_high_res/fluxes/'
    
    frc = sio.loadmat(moor_dir+'Papa'+str(year)+'_forcing_w3_5.mat', squeeze_me=True, struct_as_record=False)
    pytime = sot.mtime2pytime(frc['time'])
    yd = sot.pytime2yd(pytime)
    lat = [50]
    lon = [215]
    
    iCOARE3_Smith = frc['ustar_source'] == 'COARE 3.0+Smith'
    ustar = np.squeeze(frc['ustar'][iCOARE3_Smith,:])
    taum = ustar**2*rho0
    # wind direction [degree, anti-clockwise referenced to the East]
    wdir = np.angle(frc['W'])
    # taux = taum*np.cos(wdir)
    # tauy = taum*np.sin(wdir)
    taux = sot.butter_lpfilt(taum*np.cos(wdir), cutoff=1/6, fs=1, interp=True)
    tauy = sot.butter_lpfilt(taum*np.sin(wdir), cutoff=1/6, fs=1, interp=True)
    ustar = np.sqrt( np.sqrt(taux**2 + tauy**2)/rho0 )
    
    # all positive heat fluxes are into the ocean
    # nsw = frc['HF_shortwave']
    # nlw = frc['HF_longwave']
    with xr.open_dataset(flux_dir+'swnet50n145w_hr.cdf') as ds:
        badvalue = ds.attrs['_FillValue']
        nsw = ds.SWN_1495.where(ds.SWN_1495!=badvalue).rename('nsw').squeeze(['lon','lat','depth'], drop=True) \
                .interpolate_na(dim='time', method='linear').interp(time=pytime).data
    
    with xr.open_dataset(flux_dir+'lwnet50n145w_hr.cdf') as ds:
        badvalue = ds.attrs['_FillValue']
        nlw = -ds.LWN_1136.where(ds.LWN_1136!=badvalue).rename('nlw').squeeze(['lon','lat','depth'], drop=True) \
                 .interpolate_na(dim='time', method='linear').interp(time=pytime).data
    
    hlb = np.squeeze(frc['HF_latent'][iCOARE3_Smith,:])
    hsb = np.squeeze(frc['HF_sensible'][iCOARE3_Smith,:])
    hlb = sot.butter_lpfilt(hlb, cutoff=1/6, fs=1, interp=True)
    hsb = sot.butter_lpfilt(hsb, cutoff=1/6, fs=1, interp=True)
    sst = frc['SST']
    with xr.open_dataset(flux_dir+'sss50n145w_hr.cdf') as ds:
        badvalue = ds.attrs['_FillValue']
        sss = ds.S_41.where(ds.S_41!=badvalue).rename('sss').squeeze(['lon','lat','depth'], drop=True) \
                .interpolate_na(dim='time', method='linear').interp(time=pytime).data
    
    Le = gsw.latentheat_evap_t(sss, sst) # heat of evaporation [J/kg]
    evap = -hlb/Le/rho0*1e3*3600 # [mm/hr], positive mostly
    with xr.open_dataset(flux_dir+'rain_wspd_cor50n145w_hr.cdf') as ds:
        badvalue = ds.attrs['_FillValue']
        rain = ds.RN_485.where(ds.RN_485!=badvalue).rename('rain').squeeze(['lon','lat','depth'], drop=True) \
                 .interpolate_na(dim='time', method='linear').interp(time=pytime).data
    emp = evap - rain
    
    # thermodynamics
    ssSA  = gsw.SA_from_SP(sss, 1, lon, lat)
    ssCT  = gsw.CT_from_t(ssSA, sst, 1)
    alpha = gsw.density.alpha(ssSA, ssCT, 1)
    beta  = gsw.density.beta(ssSA, ssCT, 1)
    
    # Salinity flux
    ws0 = -(emp*1e-3/3600*sss)
    
    # Combine heat fluxes
    hsrf = nlw + hlb + hsb
    hnet = nsw + hsrf
    wt0 = -hsrf/rho0/cp
    
    # Buoyancy fluxes [m^2/s^3]
    wb0 = g*(alpha*wt0 - beta*ws0)
    wbnet_t = -g*alpha*(hnet/rho0/cp)
    wbnet_s = -g*beta*ws0
    wbnet = wbnet_t + wbnet_s
    
    fluxes = xr.Dataset(
        data_vars={'taux': (('time'), taux, {'units': 'N/m2', 'long_name': 'ZONAL WIND STRESS', 'method': 'COARE 3.0 + Smith (default)'}),
                   'tauy': (('time'), tauy, {'units': 'N/m2', 'long_name': 'MERID WIND STRESS', 'method': 'COARE 3.0 + Smith (default)'}),
                   'ustar': (('time'), ustar, {'units': 'm/s', 'long_name': 'FRICTION VELOCITY', 'method': 'COARE 3.0 + Smith (default)'}),
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
                   'wbnet': (('time'), wbnet, {'units': 'm2 s-3', 'long_name': 'KINEMATIC BUOY FLUX'})},
        coords={'yd': (('time'), yd, {'long_name': f'yearday_of_{year}'}),
                'time': (('time'), pytime),
                'depth': (('depth'), [0]),
                'lon': (('lon'), lon, {'units': 'degree', 'long_name': 'longitude'}),
                'lat': (('lat'), lat, {'units': 'degree', 'long_name': 'latitude'})},
        attrs={'description': f'Surface forcing for Ocean Station Papa ({year}) estimated from PMEL mooring data',
               'note': 'Copied from the data file created by Andrey Shcherbina'})
    
    # time array in fluxes has small deviations (<< 1s) from integer hours
    int_hr = pd.date_range(start=fluxes.time[0].dt.round('1H').values,
                           end=fluxes.time[-1].dt.round('1H').values, freq='1H')
    fluxes = fluxes.interp(time=int_hr)
    
    # fill NaN with PMEL data
    fluxes_PMEL = xr.open_dataset(flux_dir+f'OCSP_fluxes_{year}PMEL.nc').load().squeeze() \
                    .drop_vars(['taum','u10','v10','wsp','hrb','tsk','sst2','sss2'])
    fluxes_PMEL.close()
    fluxes = fluxes.fillna(fluxes_PMEL).sel(time=slice(fluxes_PMEL.time[0], fluxes_PMEL.time[-1]))
    fluxes['nsw'] = fluxes.nsw.where(fluxes.nsw>0, other=0)
    
    nc_name = f'OCSP_fluxes_{year}.nc'
    fluxes.to_netcdf(flux_dir + nc_name, engine='netcdf4')
    print(f'Air-sea fluxes: {nc_name} saved at {flux_dir}.\n')

if platform == 'linux':
    Groot = '/media/zhihua/internal1/GDrive/'
elif platform == 'darwin':
    Groot = '/Volumes/GoogleDrive/My Drive/'
_prep_ocsp_fluxes(Groot, 2011)
_prep_ocsp_fluxes(Groot, 2012)