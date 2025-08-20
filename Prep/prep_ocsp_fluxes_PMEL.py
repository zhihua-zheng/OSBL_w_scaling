import xarray as xr
import numpy as np
import sca_osbl_tool as sot
import gsw
from constants import g, cp, rho0
from sys import platform


def _prep_ocsp_fluxes_PMEL(GDrive_root, year):
    ocsp_root = GDrive_root + 'UW/Research/Data/OCSP/'
    moor_dir = ocsp_root + 'Mooring/' + str(year) + '_high_res/'
    flux_dir = moor_dir + 'fluxes/'

    # time span for 2011 ['2011-01-31T00:00:00.000000000' to '2011-05-31T23:00:00.000000000']
    # time span for 2012 ['2012-01-31T00:00:00.000000000' to '2012-05-31T23:00:00.000000000']
    with xr.open_dataset(flux_dir+'tau50n145w_hr.cdf') as ds:
        badvalue = ds.attrs['_FillValue']
        taux = ds.TX_442.where(ds.TX_442!=badvalue).rename('taux').interpolate_na(dim='time', method='linear')
        tauy = ds.TY_443.where(ds.TY_443!=badvalue).rename('tauy').interpolate_na(dim='time', method='linear')
        taum = ds.TAU_440.where(ds.TAU_440!=badvalue).rename('taum').interpolate_na(dim='time', method='linear')
    
    with xr.open_dataset(flux_dir+'wind_10meter_50n145w_hr.cdf') as ds:
        badvalue = ds.attrs['_FillValue']
        u10 = ds.UZS_2422.where(ds.UZS_2422!=badvalue).rename('u10').interpolate_na(dim='time', method='linear')
        v10 = ds.VZS_2423.where(ds.VZS_2423!=badvalue).rename('v10').interpolate_na(dim='time', method='linear')
        wsp = ds.WZS_2401.where(ds.WZS_2401!=badvalue).rename('wsp').interpolate_na(dim='time', method='linear')
    
    with xr.open_dataset(flux_dir+'swnet50n145w_hr.cdf') as ds:
        badvalue = ds.attrs['_FillValue']
        nsw = ds.SWN_1495.where(ds.SWN_1495!=badvalue).rename('nsw').interpolate_na(dim='time', method='linear')
    
    with xr.open_dataset(flux_dir+'lwnet50n145w_hr.cdf') as ds:
        badvalue = ds.attrs['_FillValue']
        nlw = -ds.LWN_1136.where(ds.LWN_1136!=badvalue).rename('nlw').interpolate_na(dim='time', method='linear')
    
    with xr.open_dataset(flux_dir+'qlat50n145w_hr.cdf') as ds:
        badvalue = ds.attrs['_FillValue']
        hlb = -ds.QL_137.where(ds.QL_137!=badvalue).rename('hlb').interpolate_na(dim='time', method='linear')
    
    with xr.open_dataset(flux_dir+'qsen50n145w_hr.cdf') as ds:
        badvalue = ds.attrs['_FillValue']
        hsb = -ds.QS_138.where(ds.QS_138!=badvalue).rename('hsb').interpolate_na(dim='time', method='linear')
    
    with xr.open_dataset(flux_dir+'rf50n145w_hr.cdf') as ds:
        badvalue = ds.attrs['_FillValue']
        hrb = -ds.QR_139.where(ds.QR_139!=badvalue).rename('hrb').interpolate_na(dim='time', method='linear')
    
    with xr.open_dataset(flux_dir+'evap50n145w_hr.cdf') as ds:
        badvalue = ds.attrs['_FillValue']
        evap = ds.E_250.where(ds.E_250!=badvalue).rename('evap').interpolate_na(dim='time', method='linear')
    
    with xr.open_dataset(flux_dir+'rain_wspd_cor50n145w_hr.cdf') as ds:
        badvalue = ds.attrs['_FillValue']
        rain = ds.RN_485.where(ds.RN_485!=badvalue).rename('rain').interpolate_na(dim='time', method='linear')
    
    with xr.open_dataset(flux_dir+'sst50n145w_hr.cdf') as ds:
        badvalue = ds.attrs['_FillValue']
        sst = ds.T_25.where(ds.T_25!=badvalue).rename('sst').interpolate_na(dim='time', method='linear')
    
    with xr.open_dataset(flux_dir+'tsk50n145w_hr.cdf') as ds:
        badvalue = ds.attrs['_FillValue']
        tsk = ds.TSK_1020.where(ds.TSK_1020!=badvalue).rename('tsk').interpolate_na(dim='time', method='linear')
    
    with xr.open_dataset(moor_dir+'t50n145w_hr.cdf') as ds:
        badvalue = ds.attrs['_FillValue']
        sst2 = ds.T_20.where(ds.T_20!=badvalue).sel(depth=['1.0', '5.0']).interp(depth=[2.0])\
                 .rename('sst2').interpolate_na(dim='time', method='linear')
        sst2.attrs.update(long_name='TEMPERATURE AT 2 METER', units='C')
    
    with xr.open_dataset(flux_dir+'sss50n145w_hr.cdf') as ds:
        badvalue = ds.attrs['_FillValue']
        sss = ds.S_41.where(ds.S_41!=badvalue).rename('sss').interpolate_na(dim='time', method='linear')
    
    with xr.open_dataset(moor_dir+'s50n145w_hr.cdf') as ds:
        badvalue = ds.attrs['_FillValue']
        sss2 = ds.S_41.where(ds.S_41!=badvalue).sel(depth=['1.0', '10.0']).interp(depth=[2.0])\
                 .rename('sss2').interpolate_na(dim='time', method='linear')
        sss2.attrs.update(long_name='SALINITY AT 2 METER', units='PSU')
    
    # thermodynamics
    ssSA  = gsw.SA_from_SP(sss, sss.depth, sss.lon, sss.lat)
    ssCT  = gsw.CT_from_t(ssSA, sst, sss.depth)
    alpha = gsw.density.alpha(ssSA, ssCT, 1).values
    beta  = gsw.density.beta(ssSA, ssCT, 1).values
    
    ustar = np.sqrt(taum/rho0).rename('ustar')
    ustar.attrs.update(long_name='FRICTION VELOCITY', units='m/s')

    # Salinity flux
    evap, rain = xr.align(evap, rain, join='override')
    emp = (evap - rain).rename('emp')
    emp.attrs.update(long_name='EVAPORATION - PRECIPITATION', units='MM/HR')
    ws0 = -(emp*1e-3/3600*sss.values).rename('ws0')
    ws0.attrs.update(long_name='KINEMATIC SALT FLUX', units='PSU m/s')
    
    # Combine heat fluxes
    hsrf = (nlw + hlb + hsb + hrb).rename('hsrf')
    hsrf.attrs.update(long_name='SURFACE HEAT FLUX (NO SW)', units='W m-2')
    hnet = (nsw + hsrf).rename('hnet')
    hnet.attrs.update(long_name='SURFACE HEAT FLUX', units='W m-2')
    wt0 = -(hsrf/rho0/cp).rename('wt0')
    wt0.attrs.update(long_name='KINEMATIC TEMP FLUX (NO SW)', units='C m/s')
    
    # Buoyancy fluxes [m^2/s^3]
    wb0 = g*(alpha*wt0 - beta*ws0).rename('wb0')
    wb0.attrs.update(long_name='KINEMATIC BUOY FLUX (NO SW)', units='m2 s-3')
    wbnet_t = -g*alpha*(hnet/rho0/cp).rename('wbnet_t')
    wbnet_t.attrs.update(long_name='KINEMATIC BUOY FLUX (T PART)', units='m2 s-3')
    wbnet_s = -g*beta*ws0.rename('wbnet_s')
    wbnet_s.attrs.update(long_name='KINEMATIC BUOY FLUX (S PART)', units='m2 s-3')
    wbnet = (wbnet_t + wbnet_s).rename('wbnet')
    wbnet.attrs.update(long_name='KINEMATIC BUOY FLUX', units='m2 s-3')
    
    fluxes = xr.merge([taux,tauy,taum,ustar,u10,v10,wsp,nsw,nlw,hlb,hsb,hrb,hsrf,hnet,tsk,
                       sst,sst2,sss,sss2,evap,rain,emp,ws0,wt0,wb0,wbnet_t,wbnet_s,wbnet],
                      join='override').assign_coords(yd=('time', sot.pytime2yd(taux.time)))\
                                      .assign_attrs(ds.attrs)
    del fluxes.attrs['name']
    del fluxes.attrs['long_name']
    del fluxes.attrs['generic_name']
    del fluxes.attrs['FORTRAN_format']
    del fluxes.attrs['units']
    del fluxes.attrs['epic_code']
    
    nc_name = 'OCSP_fluxes_' + str(year) + 'PMEL' + '.nc'
    fluxes.to_netcdf(flux_dir + nc_name, engine='netcdf4')
    print(f'Air-sea fluxes: {nc_name} saved at {flux_dir}.\n')


if platform == 'linux':
    Groot = '/media/zhihua/internal1/GDrive/'
elif platform == 'darwin':
    Groot = '/Volumes/GoogleDrive/My Drive/'
_prep_ocsp_fluxes_PMEL(Groot, 2011)
_prep_ocsp_fluxes_PMEL(Groot, 2012)