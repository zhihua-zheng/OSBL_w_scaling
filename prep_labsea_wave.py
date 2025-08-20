#!/usr/bin/env python
import cdsapi
import numpy as np
import xarray as xr
import pandas as pd
import sca_osbl_tool as sot
from sys import platform
from pathlib import Path
from tqdm.auto import tqdm
from scipy.interpolate import interpn
from constants import nan, pi, rho0, kappa


def dload_wave_spec(dirname, year, first_month, last_month, sday, eday, latlon, label):
    c = cdsapi.Client(timeout=600)
    for i,month in enumerate(range(first_month, last_month+1)):
        print('=========================================================')
        print(f'Downloading {year:04d}-{month:02d}')
        date_range = f'{year:04d}-{month:02d}-{sday[i]:02d}' + '/to/' + f'{year:04d}-{month:02d}-{eday[i]:02d}'
        file_name = dirname + label + f'_dws_{year:04d}-{month:02d}.nc'
        c.retrieve('reanalysis-era5-complete', {
        'date': date_range,
        'direction': '1/to/24',
        'domain': 'g',
        'frequency': '1/to/30',
        'param': '251.140',
        'stream': 'wave', # (hourly, 0.36 degrees) high resolution realization, HRES for waves
        'grid': '0.36/0.36', # required for netcdf
        'area': latlon, # Default: global
        'time': '00/to/23/by/1',
        'type': 'an', # analyses
        'format': 'netcdf',
        }, file_name)
        
        print('=========================================================')
        print(f'Decoding {year:04d}-{month:02d}')
        # decode spectra
        da = xr.open_dataarray(file_name)
        da = da.assign_coords(direction=np.arange(7.5, 352.5 + 15, 15))
        da = da.assign_coords(frequency=np.full(30, 0.03453) * (1.1 ** np.arange(0, 30)))
        da = 10**da
        da = da.fillna(0)
        da.to_netcdf(path=file_name, engine='netcdf4')


def creat_empty_iLa(dftb_mz):
    var1dlist = ['Hsww', 'stds', 'LaSLP2', 'LaSL2', 'LaSSL2', 'Lat2', 'Us0', 'UsSL', 'theta_ww', 'theta_wL', 
                 'xi', 'chim']#, 'LOg', 'Lss', 'LHo']
    var2dlist = ['UsdwshM']
    nan_array1d = np.full((1, dftb_mz.shape[0]), nan)
    nan_array2d = np.full((1, dftb_mz.shape[0], dftb_mz.shape[1]), nan)
    
    vard = dict([(var, (('ID','time'), nan_array1d)) for var in var1dlist])
    vard.update([(var, (('ID','time','z'), nan_array2d)) for var in var2dlist])
    return xr.Dataset(data_vars=vard, coords=dict(ID=('ID', [dftb_mz.ID.data]), 
                      time=('time', dftb_mz.time.data), z=('z', dftb_mz.z.data)))


def apply_float_mask(us_d2f, glon, glat, ilon, ilat):
    us_d2fm = us_d2f.copy()
    if np.isnan(ilon) and np.isnan(ilat):
        us_d2fm[:] = nan
    else:
        dlat = 1.5*np.abs(glat[1]-glat[0])
        dlon = 1.5*np.abs(glon[1]-glon[0])
        Glon, Glat = np.meshgrid(glon, glat)
        mask = ((Glat-ilat)>dlat) | ((Glat-ilat)<-dlat) | \
               ((Glon-ilon)>dlon) | ((Glon-ilon)<-dlon)
        us_d2fm[mask] = nan
    return us_d2fm


def interp2d(lon, lat, z, loni, lati):
    z2d = np.reshape(z, (lat.size, lon.size), order='F')
    if np.isnan(loni) or np.isnan(lati):
        zi = nan
    else:
        zi = interpn((lat,lon), z2d, (lati,loni))
    return zi


def get_La_float(Env, us_d2f, vs_d2f, dftb, ustar, theta_tau, no_gps_id):
    dftb = dftb.drop_vars('mtime')
    fwave = []    
    
    for idf in tqdm(Env.ID.data, desc='Compute wave information for each float'):
        iPos = Env[['lon', 'lat']].sel(ID=idf).rename({'timePOS': 'time'}).interp(time=dftb.time)
        wbf = dftb.wbf.sel(ID=idf)
        wstar = np.copysign(dftb.wstar.sel(ID=idf), wbf)
        bld = dftb.bld.sel(ID=idf)
        mz = dftb.mz.sel(ID=idf)
        zstd = dftb.zstd.sel(ID=idf)
        
        if idf not in no_gps_id[1:]:
            us_d2fm = xr.apply_ufunc(apply_float_mask, us_d2f, us_d2f.lon, us_d2f.lat, iPos.lon, iPos.lat,
                                     input_core_dims=[['lat','lon'], ['lon'], ['lat'], [], []],
                                     vectorize=True, output_core_dims=[['lat','lon']], output_dtypes=[float])
            vs_d2fm = xr.apply_ufunc(apply_float_mask, vs_d2f, us_d2f.lon, us_d2f.lat, iPos.lon, iPos.lat,
                                     input_core_dims=[['lat','lon'], ['lon'], ['lat'], [], []],
                                     vectorize=True, output_core_dims=[['lat','lon']], output_dtypes=[float])
            d2fm = np.sqrt(us_d2fm**2 + vs_d2fm**2)
            
            # significant wave height of wind waves
            Hsww = xr.apply_ufunc(sot.get_Hsww_d2f, d2fm.freq, d2fm, input_core_dims=[['freq'], ['freq']], 
                                  vectorize=True, output_dtypes=[float])
            iHsww = Hsww.interp(lon=iPos.lon, lat=iPos.lat).drop_vars(['lon','lat']).rename('Hsww')
            
            iustar = ustar.interp(lon=iPos.lon, lat=iPos.lat).drop_vars(['lon','lat'])
            
            ust_param = xr.apply_ufunc(sot.get_ust_param_d2f, us_d2f.freq, us_d2fm, vs_d2fm, bld, theta_tau, 
                                       input_core_dims=[['freq'], ['freq'], ['freq'], [], []],
                                       vectorize=True, output_core_dims=[['param']], output_dtypes=[float])
            iust_param = ust_param.interp(lon=iPos.lon, lat=iPos.lat).drop_vars(['lon','lat'])
            Usdw_SL   = iust_param.isel(param=0)
            Uscw_SL   = iust_param.isel(param=1)
            Usdw_ref  = iust_param.isel(param=2)
            Uscw_ref  = iust_param.isel(param=3)
            Usdw0     = iust_param.isel(param=4)
            Uscw0     = iust_param.isel(param=5).rename('Uscw0')
            Usdw_SLa  = iust_param.isel(param=6)
            Uscw_SLa  = iust_param.isel(param=7)
            Usdw_10ph = iust_param.isel(param=8)
            Uscw_10ph = iust_param.isel(param=9)
            Usdw_20ph = iust_param.isel(param=10)
            Uscw_20ph = iust_param.isel(param=11)

            Us_SL  = np.sqrt(Usdw_SL**2 + Uscw_SL**2)
            Us_ref = np.sqrt(Usdw_ref**2 + Uscw_ref**2)
            Us0    = np.sqrt(Usdw0**2 + Uscw0**2).rename('Us0')
            UsSL   = (Us_SL - Us_ref).rename('UsSL')
            
            theta_ww = np.arctan2(Uscw_SL, Usdw_SL).rename('theta_ww')
            z1 = np.maximum(iHsww, 0.02)*4 # Van Roekle et al. 2012, Eq. (29) for alpha_LOW
            theta_wL = np.arctan2(np.sin(theta_ww), iustar/kappa/Us0*np.log(bld/z1) + np.cos(theta_ww))
            idx_noLOW = bld <= z1 # assume Langmuir cells align with wave direction when bld <= z1
            theta_wL = xr.where(idx_noLOW, theta_ww, theta_wL).rename('theta_wL')
            
            LaSLP2 = (iustar*np.cos(theta_wL) / ((Us_SL - Us_ref)*np.cos(theta_ww - theta_wL))).rename('LaSLP2')
            LaSL2 = (iustar/(Usdw_SL - Usdw_ref)).rename('LaSL2')
            LaSSL2 = (iustar/(Usdw_SLa - Usdw_ref)).rename('LaSSL2')
            Lat2 = (iustar/Usdw0).rename('Lat2')

            # Stokes decay scale
            # theta_wave0 = np.arctan2(ust_param.isel(param=5), ust_param.isel(param=4)) + theta_tau
            theta_wave = np.arctan2(ust_param.isel(param=1), ust_param.isel(param=0)) + theta_tau
            stds = xr.apply_ufunc(sot.get_stds_d2f, us_d2f.freq, us_d2fm, vs_d2fm, theta_wave,
                                  input_core_dims=[['freq'], ['freq'], ['freq'], []],
                                  vectorize=True, output_dtypes=[float])
            istds = stds.interp(lon=iPos.lon, lat=iPos.lat).drop_vars(['lon','lat']).rename('stds')

            # Stokes similarity function
            xi = xr.apply_ufunc(sot.get_xi, ust_param.isel(param=4), ust_param.isel(param=6), 
                                ust_param.isel(param=10), ustar, wstar, bld, input_core_dims=[[], [], [], [], [], []], 
                                vectorize=True, output_dtypes=[float])
            ixi = xi.interp(lon=iPos.lon, lat=iPos.lat).drop_vars(['lon','lat']).rename('xi')
            chim = xr.apply_ufunc(sot.get_emp_chi, xi, input_core_dims=[[]], 
                                  kwargs={'var': 'mom'}, vectorize=True, output_dtypes=[float])
            ichim = chim.interp(lon=iPos.lon, lat=iPos.lat).drop_vars(['lon','lat']).rename('chim')
            
            # # stability lengths
            # Lstbl = xr.apply_ufunc(sot.get_stability_length_d2f, us_d2f.freq, us_d2fm, vs_d2fm, theta_tau, ustar, wbf, #chim, 
            #                        input_core_dims=[['freq'], ['freq'], ['freq'], [], [], []],#, []], 
            #                        vectorize=True, output_core_dims=[['param']], output_dtypes=[float])
            # iLstbl = Lstbl.interp(lon=iPos.lon, lat=iPos.lat).drop_vars(['lon','lat'])
            # LOg = iLstbl.isel(param=0).rename('LOg')
            # Lss = iLstbl.isel(param=1).rename('Lss')
            # LHo = iLstbl.isel(param=2).rename('LHo')
            
            # surface proximity function (1-fzS) decay depth
            Ls = xr.apply_ufunc(sot.get_Ls_d2f, us_d2f.freq, us_d2fm, vs_d2fm, 
                                ustar, bld, theta_tau, 
                                input_core_dims=[['freq'], ['freq'], ['freq'], [], [], []],
                                vectorize=True, output_dtypes=[float], kwargs={'dz': 0.5})
            iLs = Ls.interp(lon=iPos.lon, lat=iPos.lat).drop_vars(['lon','lat']).rename('Ls')
            
            # downwind Stokes shear
            UsdwshM2d = xr.apply_ufunc(sot.get_dUsdwdzM_d2f, mz, zstd, us_d2f.freq, us_d2fm, vs_d2fm, theta_tau, 
                                       input_core_dims=[[], [], ['freq'], ['freq'], ['freq'], []],
                                       vectorize=True, output_dtypes=[float])
            UsdwshM2d = UsdwshM2d.isel(lat=slice(None, None, -1)) # reverse lat
            UsdwshM = xr.apply_ufunc(interp2d, UsdwshM2d.lon, UsdwshM2d.lat, UsdwshM2d.stack(xy=('lon','lat')), 
                                     iPos.lon, iPos.lat, input_core_dims=[['lon'], ['lat'], ['xy'], [], []],
                                     vectorize=True, output_dtypes=[float]).rename('UsdwshM')
            
            if idf==no_gps_id[0]:
                # because these floats have the same trajectory, theta_tau and waves are the same 
                # also bld, ustar, wstar are the same for these floats
                # but UsdwshM is not the same since it is calculated on different float mz grid
                us_d2fm_mgps = us_d2fm.copy()
                vs_d2fm_mgps = us_d2fm.copy()
                iLa1d_mgps = xr.merge([iHsww, istds, LaSLP2, LaSL2, LaSSL2, Lat2, Us0, UsSL, theta_ww, theta_wL, ixi, ichim, iLs])
            
            iLa = xr.merge([iHsww, istds, LaSLP2, LaSL2, LaSSL2, Lat2, Us0, UsSL, theta_ww, theta_wL, ixi, ichim, iLs, UsdwshM])
            fwave.append(iLa)
        else:
            # downwind Stokes shear
            UsdwshM2d = xr.apply_ufunc(sot.get_dUsdwdzM_d2f, mz, zstd, us_d2fm_mgps.freq, us_d2fm_mgps, vs_d2fm_mgps, theta_tau, 
                                       input_core_dims=[[], [], ['freq'], ['freq'], ['freq'], []],
                                       vectorize=True, output_dtypes=[float])
            UsdwshM2d = UsdwshM2d.isel(lat=slice(None, None, -1)) # reverse lat
            UsdwshM = xr.apply_ufunc(interp2d, UsdwshM2d.lon, UsdwshM2d.lat, UsdwshM2d.stack(xy=('lon','lat')), 
                                     iPos.lon, iPos.lat, input_core_dims=[['lon'], ['lat'], ['xy'], [], []],
                                     vectorize=True, output_dtypes=[float]).rename('UsdwshM')
            iLa1d = iLa1d_mgps.copy()
            iLa1d['ID'] = ('ID', [idf])
            iLa = xr.merge([iLa1d, UsdwshM])
            fwave.append(iLa)
            # fwave.append(creat_empty_iLa(mz))
    fwave = xr.concat(fwave, dim='ID', combine_attrs='drop').transpose('ID', 'time', 'z')
    return fwave


def _prep_labsea_wave(GDrive_root, year, no_gps_id, crt_flux=True): #, lonlim, latlim):
    site_root = GDrive_root + 'UW/Research/Data/LabSea/'
    nbf_dir = site_root + 'NBF/'
    flux_dir = site_root + 'Met/'
    wave_dir = site_root + 'Wave/'
    if crt_flux:
        nametag = ''
    else:
        nametag = 'E'
    dftb_name = 'LabSea_drifts_binned_'+str(year)+nametag+'.nc'
    env_name = 'LabSea_env_'+str(year)+'.nc'
    # flux_name = 'LabSea_fluxes_'+str(year)+nametag+'.nc'
    
    with xr.open_dataset(nbf_dir+dftb_name) as dftb:
        bldm = dftb.bld.mean('ID')
    
    env = xr.open_dataset(nbf_dir+env_name).load()
    env.close()
    # mlon = env.lon.mean('ID')
    # mlat = env.lat.mean('ID')
    # env['lon'] = env.lon.fillna(value=mlon)
    # env['lat'] = env.lat.fillna(value=mlat)
    
    dws = xr.open_mfdataset(wave_dir+f'LabSea_dws_{year:04d}*.nc').load() \
            .rename({'longitude': 'lon', 'latitude': 'lat', 'frequency': 'freq', 'direction': 'dirc'})#, lon=lonlim, lat=latlim)
    dws.close()
    drad = dws.dirc/180*pi # to, clockwise relative to the North
    bw_drad = drad[1] - drad[0]
    # Hs = 4*np.sqrt(xr.apply_ufunc(np.trapz, (dws.d2fd*bw_drad).sum('dirc',skipna=False), dws.freq,
    #                               input_core_dims=[['freq'], ['freq']],
    #                               vectorize=True, output_dtypes=[float]))
    us_d2f = (np.sin(drad)*dws.d2fd*bw_drad).sum('dirc', skipna=False).shift(time=0).sel(time=dftb.time)
    vs_d2f = (np.cos(drad)*dws.d2fd*bw_drad).sum('dirc', skipna=False).shift(time=0).sel(time=dftb.time)
    
    with xr.open_dataset(site_root+'Reanalysis/ERA5/labsea_surface_hourly_JFM1997-1998.nc') \
                        .rename({'longitude': 'lon', 'latitude': 'lat'}) as F:#, lon=lonlim, lat=latlim
        tau = np.sqrt(F.mntss**2+F.metss**2)
        if crt_flux and year==1997:
            tau = tau - np.maximum(0.72*(tau-0.33), 0)
        ustar = np.sqrt(tau/rho0).interp(lon=dws.lon, lat=dws.lat).rename('ustar').shift(time=0).sel(time=dftb.time)
        theta_tau = np.arctan2(F.mntss, F.metss).interp(lon=dws.lon, lat=dws.lat).rename('theta_tau').shift(time=0).sel(time=dftb.time) 
    
    # bld = bldm.interp(time=dws.time)
    # with xr.open_dataset(flux_dir+flux_name) as F:
    #     taux = F.tauxE.interp(time=dws.time)
    #     tauy = F.tauyE.interp(time=dws.time)
    #     ustar = np.sqrt(np.sqrt(taux**2 + tauy**2)/rho0)
    #     theta_tau = np.arctan2(tauy, taux)
    
    fwave = get_La_float(env, us_d2f, vs_d2f, dftb, ustar, theta_tau, no_gps_id)
    
    # # this creates more nans for UsdwshM since each float has different mz
    # # fill gaps with mean across floats.
    # for var in list(fwave):
    #     var_mean = fwave[var].mean('ID')
    #     dftb[var] = fwave[var].fillna(value=var_mean)
    for var in list(fwave):
        dftb[var] = fwave[var]
    
    dftb['zb'] = -0.6*dftb.Hsww
    dftb['zt'] = -10*dftb.Hsww
    dftb = dftb.drop_vars('Hsww')
    # dftb['zoLOg'] = np.abs(dftb.mz)/dftb.LOg
    # dftb['zoLss'] = np.abs(dftb.mz)/dftb.Lss
    # dftb['zoLHo'] = np.abs(dftb.mz)/dftb.LHo
    
    dftb['SSP'] = dftb.ustar**2*(1+dftb.zoh)*dftb.UsdwshM
    dftb['spf'] = 1 - xr.apply_ufunc(sot.get_fzSM, dftb.mz, dftb.zstd, dftb.Ls, 
                                     input_core_dims=[[], [], []], vectorize=True, output_dtypes=[float])
    
    nc_name = 'LabSea_dftb_wave_'+str(year)+nametag+'.nc'
    dftb.to_netcdf(nbf_dir + nc_name, engine='netcdf4', # encoding makes sure the boolean dtype
                   encoding={'Iequil': {'dtype': 'bool'}, 'Ishoal': {'dtype': 'bool'}, 'Ideepen': {'dtype': 'bool'}})
    print(f'Binned float drift and wave data: {nc_name} saved at {nbf_dir}.')


if platform == 'linux':
    Groot = '/media/zhihua/internal1/GDrive/'
elif platform == 'darwin':
    Groot = '/Volumes/GoogleDrive/My Drive/'
labsea_root = Groot + 'UW/Research/Data/LabSea/'
labsea_wave_dir = labsea_root + 'Wave/'
data97_path = Path(labsea_wave_dir+'LabSea_dws_1997-02.nc')
data98_path = Path(labsea_wave_dir+'LabSea_dws_1998-02.nc')

if not data97_path.is_file():
    #'1997-02-10/to/1997-03-16'
    year = 1997
    first_month = 2
    last_month = 3
    sday = [10, 1]
    eday = [28, 16]
    latlon = '59/-56/56/-52' # North/West/South/East
    label = 'LabSea'
    dload_wave_spec(labsea_wave_dir, year, first_month, last_month, sday, eday, latlon, label)

if not data98_path.is_file():
    #'1998-01-24/to/1998-03-26'
    year = 1998
    first_month = 1
    last_month = 3
    sday = [24, 1, 1]
    eday = [31, 28, 26]
    dload_wave_spec(labsea_wave_dir, year, first_month, last_month, sday, eday, latlon, label)

no_gps_id = [6, 8, 10, 12, 14, 17, 20]
_prep_labsea_wave(Groot, 1997, no_gps_id)#, slice(-54,-52.2), slice(58.6,57))
_prep_labsea_wave(Groot, 1997, no_gps_id, crt_flux=False)
# no_gps_id = [26]
# _prep_labsea_wave(Groot, 1998, no_gps_id)#, slice(-55.4,-53.8), slice(57.8,56.9))