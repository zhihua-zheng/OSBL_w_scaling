import numpy as np
import xarray as xr
import pandas as pd
import sca_osbl_tool as sot
import scipy.io as sio
import gsw
from sys import platform
from constants import nan, pi, rho0, rhoa, g, cp, kappa


def get_Fanapi_Cdn(U10, ifr, ifl, opt='URI'):
    """
    Compute the neutral downwind drag coefficient under typhoon Fanapi (2010),
    according to the results of Hsu et al. (2019) [drag_option='APL'], or Zhou et al. (2022) [drag_option='URI']
    The low wind regime (< 25 m/s) is supplemented by the results of Large and Pond (1981)
    Cdnd: neutral downwind drag coefficient
    phi: angle of stress, clockwise from wind [degree]
    """
    Cdnd = np.full_like(U10, nan)
    phi = np.full_like(U10, nan)
    
    U10_low = np.array([4, 11, 20])
    Cdnd_low = np.array([1.2, 1.2, 1.79])*1e-3
    phi_low = np.zeros_like(U10_low)
    
    # front_right
    if opt == 'APL':
        U10_high = np.array([25, 27.5, 35, 42.5, 60])
        Cdnd_high = np.array([1.13, 1.13, 2.37, 1.63, 1.63])*1e-3
        phi_high = np.array([42, 42, 2, 6, 6])
    elif opt == 'URI':
        U10_high = np.array([25, 26.92, 33.07, 42.27, 60])
        Cdnd_high = np.array([1.319, 1.319, 1.643, 1.781, 1.781])*1e-3
        phi_high = np.array([50.59, 50.59, -9.36, 0.63, 0.63])
    U10_grid = np.hstack((U10_low, U10_high))
    Cdnd_grid = np.hstack((Cdnd_low, Cdnd_high))
    phi_grid = np.hstack((phi_low, phi_high))
    Cdnd[ifr] = np.interp(U10[ifr], U10_grid, Cdnd_grid)
    phi[ifr] = np.interp(U10[ifr], U10_grid, phi_grid)
    
    # front_left
    if opt == 'APL':
        U10_high = np.array([25, 27.5, 35, 60])
        Cdnd_high = np.array([1.69, 1.69, 1.4, 1.4])*1e-3
        phi_high = np.array([-9, -9, -12, -12])
    elif opt == 'URI':
        U10_high = np.array([25, 27.06, 32.5, 60])
        Cdnd_high = np.array([2.254, 2.254, 1.245, 1.245])*1e-3
        phi_high = np.array([-12.19, -12.19, -13.06, -13.06])
    U10_grid = np.hstack((U10_low, U10_high))
    Cdnd_grid = np.hstack((Cdnd_low, Cdnd_high))
    phi_grid = np.hstack((phi_low, phi_high))
    Cdnd[ifl] = np.interp(U10[ifl], U10_grid, Cdnd_grid)
    phi[ifl] = np.interp(U10[ifl], U10_grid, phi_grid)
    
    Cdn = Cdnd*np.sqrt(1 + np.tan(phi/180*pi)**2)
    return Cdnd, phi, Cdn


def get_fetch(theta, r):
    """
    Compute the fetch under a tropical cyclone according to the parametric model of Hwang et al. (2016)
    """
    theta_grid = np.array([-13, 7, 22, 67, 115, 157, 200, 242, 270, 292, 330, 347, 367])
    A_grid = np.array([100, 94.72, 108.79, 77.47, -33.58, 37.75, 43.45, 134.73, 107.65, 149.25, 109.42, 100, 94.72])
    alpha_grid = np.array([0.5, 0.14, -0.01, 0.93, 1.78, 0.4, 0.7, 0.69, 0.46, 0.3, 0.81, 0.5, 0.14])
    
    A = np.interp(theta, theta_grid, A_grid, period=360)
    alpha = np.interp(theta, theta_grid, alpha_grid, period=360)
    return alpha*r + A


def get_psi(Uh, theta):
    """
    Compute the angle between wind and dominant waves under a tropical cyclone according to Moon et al. (2004)
    """
    Uh0 = 1
    psi = 30 + 3*(Uh/Uh0)*(1 + np.tanh(theta-10))
    return psi


def get_ewd_Cdn(U10, theta, r, Uh):
    """
    Compute the effective wind duration based parameterization of drag coefficient according to Hsu et al. (2019)
    """
    fetch = get_fetch(theta, r)
    psi = get_psi(Uh, theta)
    ewd = g*(fetch*1e3/Uh)/(U10*np.cos(psi*pi/180))
    
    Cdnd = np.full_like(U10, nan)
    phi = np.full_like(U10, nan)
    
    idx1 = (ewd >= 5e3) & (ewd <= 9e3)
    idx2 = (ewd >= 9e3) & (ewd <= 22e3)
    Cdnd[idx1] = 2.7*1e-3
    Cdnd[idx2] = (-5.2e-3*(ewd[idx2]/1e3)**2 + 3.8e-3*ewd[idx2]/1e3 + 3.05)*1e-3
    
    idx3 = (ewd >= 5e3) & (ewd <= 12e3)
    idx4 = (ewd >= 12e3) & (ewd <= 22e3)
    phi[idx3] = 0
    phi[idx4] = 0.44*(ewd[idx4]/1e3)**2 - 9.7*ewd[idx4]/1e3 + 52.45
    
    Cdn = Cdnd*np.sqrt(1 + np.tan(phi/180*pi)**2)
    return Cdnd, phi, Cdn


def get_GFDL_Cdn(U10):
    """
    Compute the GFDL (HWRF) parameterization of drag coefficient according to Zhou et al. (2022)
    """
    a1 =  1.044183210405817e-12
    a2 = -5.707116220939218e-11
    a3 =  8.005722172810571e-10
    a4 =  6.322045801589353e-09
    a5 = -2.422002988137712e-07
    a6 =  2.269200594753249e-06
    a7 = -6.029592778169796e-06
    a8 =  8.882284703541603e-06
    a9 = -2.371341185499601e-06
    
    b1 =  1.814407011197660e-15
    b2 = -1.602907562918788e-13
    b3 = -3.351205313520358e-11
    b4 =  6.036179295940524e-09
    b5 = -3.725481686822030e-07
    b6 =  1.059761705898929e-05
    b7 = -1.375241830530252e-04
    b8 =  8.538858261732818e-04
    b9 = -1.936638976963742e-03
    
    idx1 = U10 <= 0.4
    idx2 = (U10 > 0.4) & (U10 <= 9.3)
    idx3 = (U10 > 9.3) & (U10 <  60)
    idx4 = U10 >= 60
    
    z0 = np.full_like(U10, nan)
    z0[idx1] = 4e-7
    z0[idx2] = a9 + a8*U10[idx2] + a7*U10[idx2]**2 + a6*U10[idx2]**3 + a5*U10[idx2]**4 + \
               a4*U10[idx2]**5 + a3*U10[idx2]**6 + a2*U10[idx2]**7 + a1*U10[idx2]**8
    z0[idx3] = b9 + b8*U10[idx3] + b7*U10[idx3]**2 + b6*U10[idx3]**3 + b5*U10[idx3]**4 + \
               b4*U10[idx3]**5 + b3*U10[idx3]**6 + b2*U10[idx3]**7 + b1*U10[idx3]**8
    z0[idx4] = 1.3025e-3
    return (kappa/np.log(10/z0))**2


def get_rdt(U10):
    """
    Compute correction factor for air-sea temperature difference
    Based on Lin et al. 2011, Fig. S8
    """
    return np.minimum(1+0.4*np.tanh((U10-32)/5), 1-0.6*np.tanh((U10-32)/5))


def get_rdq(U10):
    """
    Compute correction factor for air-sea specific humidity difference
    Based on Lin et al. 2011, Fig. S8
    """
    return np.minimum(0.85+0.4*np.tanh((U10-32)/3), 0.85-0.45*np.tanh((U10-32)/5))


def _prep_fanapi_fluxes(GDrive_root, floatID, tau_option='APL'):
    site_root = GDrive_root + 'UW/Research/Data/Hurricanes/ITOP/'
    era5_dir = site_root + 'Reanalysis/ERA5/'
    met_dir = site_root + 'Met/'
    nbf_dir = site_root + 'NBF/'
    env_name = 'Fanapi_env_'+str(floatID)+'.nc'
    year = 2010
    
    # ERA5 fluxes
    era5 = xr.open_dataset(era5_dir+'fanapi_surface_hourly_Sep2010.nc').load()
    era5.close() # reanalysis mean is over the past hour
    era5 = era5.assign_coords({'time': ('time', era5.time.values - pd.Timedelta(minutes=30))})
    era5 = era5.rename({'longitude': 'lon', 'latitude': 'lat'}) #.set_index({'': ''})
    era5['mer'] = -era5.mer # [m/s] ERA5 mer is negative/positive for evaporation/condensation
    era5['sst'] = era5.sst - 273.15 # [C]
        
    # Wind field and drag coefficient from Hsu et al. 2019 (APL), Zhou et al. 2022 (URI)
    if tau_option == 'APL':
        Hsu19 = sio.loadmat(met_dir+'Fanapi_wind_lon_lat.mat', squeeze_me=True, struct_as_record=False)
        pytime = sot.mtime2pytime(Hsu19['Jday_gmt'])
        Fanapi = xr.Dataset(data_vars=dict(u10=(('lat','lon','time'), Hsu19['u10']), 
                                           v10=(('lat','lon','time'), Hsu19['v10']),
                                           uhx=(('time'), Hsu19['uhx']),
                                           uhy=(('time'), Hsu19['uhy']),
                                           trk_lat=(('time'), Hsu19['trk_lat']),
                                           trk_lon=(('time'), Hsu19['trk_lon'])),
                            coords=dict(lon=(('lon'), Hsu19['Lon']),
                                        lat=(('lat'), Hsu19['Lat']),
                                        time=(('time'), pytime)),
                            attrs=dict(description='Typhoon Fanapi surface wind and track from Hsu et al. 2019',
                                       lon2km=Hsu19['lon2km'], lat2km=Hsu19['lat2km'], mrmw=20))
        Fanapi['shfE'] = era5.msshf.interp(lon=Fanapi.lon, lat=Fanapi.lat, time=Fanapi.time)
        Fanapi['lhfE'] = era5.mslhf.interp(lon=Fanapi.lon, lat=Fanapi.lat, time=Fanapi.time)
        Fanapi['evapE'] = era5.mer.interp(lon=Fanapi.lon, lat=Fanapi.lat, time=Fanapi.time)
        Fanapi['u10E'] = era5.u10.interp(lon=Fanapi.lon, lat=Fanapi.lat, time=Fanapi.time)
        Fanapi['v10E'] = era5.v10.interp(lon=Fanapi.lon, lat=Fanapi.lat, time=Fanapi.time)
        Fanapi['U10E'] = np.sqrt(Fanapi.u10E**2 + Fanapi.v10E**2)
        Fanapi['U10'] = np.sqrt(Fanapi.u10**2 + Fanapi.v10**2)
        Fanapi = Fanapi.transpose('lat', 'lon', 'time')
        rwind = Fanapi.U10/Fanapi.U10E
        rdt = get_rdt(Fanapi.U10).where(Fanapi.U10>=15, other=1/rwind)
        rdq = get_rdq(Fanapi.U10).where(Fanapi.U10>=15, other=1/rwind)
        # only modify turbulent fluxes within 34 knot (17.5 m/s) wind radius
        Fanapi['shf'] = Fanapi.shfE*rwind*rdt
        Fanapi['lhf'] = Fanapi.lhfE*rwind*rdq
        Fanapi['evap'] = Fanapi.evapE*rwind*rdq
    elif tau_option == 'URI':
        Fanapi = xr.open_dataset(met_dir+'Fanapi_U10_URI.nc').load()
        Fanapi.close()
        Fanapi = Fanapi.rename({'u': 'u10', 'v': 'v10'})
        track = sio.loadmat(met_dir+'Fanapi_track_URI.mat', squeeze_me=True, struct_as_record=False)
        track_time = sot.mtime2pytime(track['trk_mtime'])
        trk = xr.Dataset(data_vars=dict(lat=(('time'), track['trk_lat']),
                                        lon=(('time'), track['trk_lon'])),
                         coords=dict(time=(('time'), track_time))).interp(time=Fanapi.time)
        Fanapi['trk_lon'] = trk.lon
        Fanapi['trk_lat'] = trk.lat
        Fanapi = Fanapi.dropna('time', how='any')
        Fanapi.attrs['lon2km'] = 101.9127
        Fanapi.attrs['lat2km'] = 111.2
        Fanapi.attrs['mrmw'] = 28 # minimum radius of max wind
        Fanapi['uhx'] = Fanapi.trk_lon.differentiate('time', datetime_unit='s')*Fanapi.lon2km*1e3
        Fanapi['uhy'] = Fanapi.trk_lat.differentiate('time', datetime_unit='s')*Fanapi.lat2km*1e3
        Fanapi = Fanapi.sel(time=slice(None, '2010-09-18 03:09:00'))
        Fanapi['shfE'] = era5.msshf.interp(lon=Fanapi.lon, lat=Fanapi.lat, time=Fanapi.time)
        Fanapi['lhfE'] = era5.mslhf.interp(lon=Fanapi.lon, lat=Fanapi.lat, time=Fanapi.time)
        Fanapi['evapE'] = era5.mer.interp(lon=Fanapi.lon, lat=Fanapi.lat, time=Fanapi.time)
        Fanapi['u10E'] = era5.u10.interp(lon=Fanapi.lon, lat=Fanapi.lat, time=Fanapi.time)
        Fanapi['v10E'] = era5.v10.interp(lon=Fanapi.lon, lat=Fanapi.lat, time=Fanapi.time)
        Fanapi['U10E'] = np.sqrt(Fanapi.u10E**2 + Fanapi.v10E**2)
        Fanapi['U10'] = np.sqrt(Fanapi.u10**2 + Fanapi.v10**2)
        Fanapi = Fanapi.transpose('lat', 'lon', 'time')
        rwind = Fanapi.U10/Fanapi.U10E
        rdt = get_rdt(Fanapi.U10).where(Fanapi.U10>=15, other=1/rwind)
        rdq = get_rdq(Fanapi.U10).where(Fanapi.U10>=15, other=1/rwind)
        # only modify turbulent fluxes within 34 knot (17.5 m/s) wind radius
        Fanapi['shf'] = Fanapi.shfE*rwind*rdt
        Fanapi['lhf'] = Fanapi.lhfE*rwind*rdq
        Fanapi['evap'] = Fanapi.evapE*rwind*rdq
    
    with xr.open_dataset(nbf_dir+env_name) as env:
        flon = env.lon.interp(time=Fanapi.time)
        flat = env.lat.interp(time=Fanapi.time)
        smean = env.S.where(env.P<5, drop=True).mean().values
    
    # interpolate reconstructed wind field to float trajectory
    fwind = Fanapi.interp(lon=flon, lat=flat).dropna('time')
    fwind['uhdir'] = np.arctan2(fwind.uhy, fwind.uhx)*180/pi
    fwind['Uh'] = np.sqrt(fwind.uhy**2 + fwind.uhx**2)
    fwind['U10'] = np.sqrt(fwind.u10**2 + fwind.v10**2)
    fwind['U10E'] = np.sqrt(fwind.u10E**2 + fwind.v10E**2)
    # position of float relative to Fanapi
    fdx = (flon-fwind.trk_lon)*fwind.lon2km
    fdy = (flat-fwind.trk_lat)*fwind.lat2km
    fwind['fdist'] = np.sqrt(fdx**2 + fdy**2) # [km]
    fdir = np.arctan2(fdy, fdx)*180/pi
    frdir = fdir - fwind.uhdir # float direction relative to Fanapi
    frdir[frdir < -180] = frdir[frdir < -180] + 360
    frdir[frdir >  180] = frdir[frdir >  180] - 360
    fwind['frdir'] = frdir
    ifront = np.cos(fwind.frdir/180*pi) >= np.cos((90+20)/180*pi) # only use times before typhoon arrives
    ihfront = (ifront & (fwind.U10 >= 25)).rename('ihfront')
    # identify if float was under the storm
    in_storm = fwind.fdist <= fwind.mrmw
    ifl = ifront & (fwind.frdir > 10) # theta_s = 10 as the boundary between sectors in Fig. 2c of Hsu et al. (2019)
    ifr = ifront & (fwind.frdir <= 10 | in_storm)
    
    Cdnd,phi,Cdn = get_Fanapi_Cdn(fwind.U10, ifr, ifl, opt=tau_option)
    Cdnd_ewd,phi_ewd,Cdn_ewd = get_ewd_Cdn(fwind.U10, fwind.frdir, fwind.fdist, fwind.Uh)
    Cdn_GFDL = get_GFDL_Cdn(fwind.U10)
    
    fwind['taum'] = Cdn*rhoa*fwind.U10**2
    fwind['taum_ewd'] = Cdn_ewd*rhoa*fwind.U10**2
    fwind['taum_GFDL'] = Cdn_GFDL*rhoa*fwind.U10**2
    tau_dir = np.arctan2(fwind.v10, fwind.u10) - phi/180*pi # Hsu's stress direction has large uncertainty
    fwind['taux'] = fwind.taum*np.cos(tau_dir)
    fwind['tauy'] = fwind.taum*np.sin(tau_dir)
    
    # interpolate ERA5 to float trajectory
    varlist = ['msnswrf', 'msnlwrf', 'mtpr', 'sst']
    fmet = era5[varlist].interp(lon=fwind.lon, lat=fwind.lat, time=fwind.time)#.drop_vars(['lon','lat'])
    fmet = fmet.rename({'msnswrf': 'nsw', 'msnlwrf': 'nlw', 'mtpr': 'rain'})
    fmet = fmet.assign(sss=(xr.full_like(fmet.sst, smean)).assign_attrs(longname='SEA SURFACE SALINITY', units='PSU'))
    fmet.sst.attrs.update(longname='SEA SURFACE TEMPERATURE', units='C')
    
    # thermodynamics
    ssSA  = gsw.SA_from_SP(fmet.sss, 1, flon.mean(), flat.mean())
    ssCT  = gsw.CT_from_t(ssSA, fmet.sst, 1)
    alpha = gsw.density.alpha(ssSA, ssCT, 1).values
    beta  = gsw.density.beta(ssSA, ssCT, 1).values
    ustar = np.sqrt(fwind.taum/rho0).rename('ustar')
    ustar.attrs.update(long_name='FsRICTION VELOCITY', units='m/s')
    
    # Salinity flux
    fmet['emp'] = (fwind.evap-fmet.rain)/rho0
    fmet.emp.attrs.update(longname='EVAPORATION - PRECIPITATION', units='m/s')
    ws0 = -(fmet.emp*fmet.sss).rename('ws0')
    ws0.attrs.update(long_name='KINEMATIC SALT FLUX', units='PSU m/s')
    
    # Combine heat fluxes
    hsrf = (fmet.nlw + fwind.lhf + fwind.shf).rename('hsrf')
    hsrf.attrs.update(long_name='SURFACE HEAT FLUX (NO SW)', units='W m-2')
    hnet = (fmet.nsw + hsrf).rename('hnet')
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
    
    fluxes = xr.merge([fwind.taux,fwind.tauy,fwind.taum,fwind.taum_GFDL,fwind.taum_ewd,ihfront,
                       fwind.u10,fwind.v10,fwind.U10,fwind.U10E,fwind.trk_lon,fwind.trk_lat,fwind.uhdir,
                       fmet.nsw,fmet.nlw,fwind.lhf,fwind.lhfE,fwind.shf,fwind.shfE,hsrf,hnet,
                       fmet.sst,fmet.sss,fmet.emp,ustar,ws0,wt0,wb0,wbnet_t,wbnet_s,wbnet],
                      join='inner').assign_coords(yd=('time', sot.pytime2yd(fwind.time)))
    fluxes = fluxes.assign_attrs(lon2km=fwind.lon2km, lat2km=fwind.lat2km, mrmw=fwind.mrmw)
    
    nc_name = 'Fanapi_fluxes_' + str(floatID) + '_' + tau_option + '.nc'
    fluxes.to_netcdf(met_dir + nc_name, engine='netcdf4')
    print(f'Air-sea fluxes: {nc_name} saved at {met_dir}.\n')

if platform == 'linux':
    Groot = '/media/zhihua/internal1/GDrive/'
elif platform == 'darwin':
    Groot = '/Volumes/GoogleDrive/My Drive/'
_prep_fanapi_fluxes(Groot, 60, 'APL')
_prep_fanapi_fluxes(Groot, 61, 'APL')
_prep_fanapi_fluxes(Groot, 62, 'APL')
_prep_fanapi_fluxes(Groot, 64, 'APL')

_prep_fanapi_fluxes(Groot, 60, 'URI')
_prep_fanapi_fluxes(Groot, 61, 'URI')
_prep_fanapi_fluxes(Groot, 62, 'URI')
_prep_fanapi_fluxes(Groot, 64, 'URI')