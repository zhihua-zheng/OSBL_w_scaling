import netCDF4 as nc
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.ndimage as ndimage
import sca_osbl_tool as sot
import gsw
import cmath
from sys import platform
from constants import g, cp, rho0, pi, dnum_unit, nan
from datetime import datetime, timedelta
from scipy.interpolate import Rbf, griddata, RegularGridInterpolator

def crt_nbf_Pfast(site, cyear, pf, ep, ref='pre_dep'):
    """
    Adjust the float's pressure record to remove variations due to atmopsheric pressure changes, 
    using the 12-hour mean of atmopsheric pressure before the deployment, 
    or the overall mean of atmopsheric pressure during the deployment as the reference value; 
    and shift the float's pressure record by a constant to account for the calibration offset.
    """
    if 'atmos_pressure_correction' in pf:
        if sum(pf['atmos_pressure_correction']) == 3:
            print('Atmosphere pressure has already been corrected. Skip.')
            return
    
    # barometric pressure
    t_bp,bp = globals()['read_'+site+'_bp'](cyear)

    # fill missing data in barometric pressure by linear interpolation
    yd_bp = sot.pytime2yd(t_bp)
    bp_is_nan = np.isnan(bp)
    bp_nonan = np.interp(yd_bp, yd_bp[~bp_is_nan],  bp[~bp_is_nan])

    # reference of barometric pressure
    precal_idx = (yd_bp > (pf['ydP'][0]-0.5)) & (yd_bp < pf['ydP'][0])
    bp_premean = bp_nonan[precal_idx].mean()
    bp_allmean = bp_nonan.mean()
    
    if ref == 'pre_dep':
        bp0 = bp_premean
    elif ref == 'all_dep':
        bp0 = bp_allmean
    else:
        print('prescribed <ref> not implemented...')
        return
    
    # interpolate barometic pressure to float time 
    # and compute changes relative to a reference value
    aPair = (np.interp(pf['ydP'], yd_bp, bp_nonan) - bp0)/100 # [dbar], 1 hPa = 0.01 dbar
    pf['P0'],pf['P1'],pf['P2'] = pf['P0']-aPair, pf['P1']-aPair, pf['P2']-aPair
    pf['atmos_pressure_correction'] = [True]*3
    
    # show different choices of Patmos reference
#     plt.figure(figsize=(9.5,2))
#     plt.axvline(pf['ydP'][0],c='r',ls='--',lw=1)
#     plt.axvline(pf['ydP'][0]-0.5,c='r',ls='--',lw=1)
#     plt.plot(yd_bp,bp,ls='-',ms=2);
#     plt.axhline(bp_premean,c='b',ls='--')
#     plt.axhline(bp_allmean,c='g',ls='--')
#     plt.savefig('../Figures/P_atmos_crt.png',dpi=300,bbox_inches='tight')
#     plt.close()
    
    mean_P2mP0 = (pf['P2']-pf['P0']).mean()
    mean_P1mP0 = (pf['P1']-pf['P0']).mean()
    mean_P2mP1 = (pf['P2']-pf['P1']).mean()
#     plt.figure(figsize=(6,5))
#     plt.hist(pf['P2']-pf['P0'],bins='auto',histtype='stepfilled',
#              alpha=0.5,color='r',density=True,label=r'$P_2 - P_0$')
#     plt.hist(pf['P1']-pf['P0'],bins='auto',histtype='stepfilled',
#              alpha=0.5,color='g',density=True,label=r'$P_1 - P_0$')
#     plt.hist(pf['P2']-pf['P1'],bins='auto',histtype='stepfilled',
#              alpha=0.5,color='b',density=True,label=r'$P_2 - P_1$');
#     plt.axvline(mean_P2mP0,ls='--',c='r')
#     plt.axvline(mean_P1mP0,ls='--',c='g')
#     plt.axvline(mean_P2mP1,ls='--',c='b')
#     plt.xlim(0,1.7)
#     plt.ylabel('PDF')
#     plt.legend(loc='best')
#     plt.close
    
    # minium pressure during drifts
    Ndrift = np.nanmax(ep['drift']).astype(int)
    P0_surf = np.full(Ndrift, nan)
    for j in range(Ndrift):
        i = j+1
        iep_drift = np.nonzero(ep['drift']==i)[0]
        if len(iep_drift) < 121: continue
        is_drift = (pf['ydP'] > ep['yd'][iep_drift[0]]) & (pf['ydP'] < ep['yd'][iep_drift[-1]])
        P0_surf[j] = np.nanmin(pf['P0'][is_drift])
    P0_surf[P0_surf > 20] = nan
    
    # show the distribution of Pmin
#     plt.figure()
#     plt.hist(P0_surf,bins='auto',histtype='stepfilled',alpha=0.5)
#     plt.axvline(np.nanmean((P0_surf)),ls='--')
#     plt.xlabel(r'$P_0$ at the surface')
#     plt.annotate('mean: {0:.2f}'.format(np.nanmean(P0_surf)),(-0.075,22))
#     plt.close()
    
    if 'calib_offset_correction' in pf:
        if sum(pf['calib_offset_correction']) == 3:
            print('Calibration offset has already been corrected. Skip.')
            return

    P0_offset = np.around(0.461 - np.nanmean(P0_surf),2)
    P1_offset = P0_offset - (mean_P1mP0 - 0.3)
    P2_offset = P0_offset - (mean_P2mP0 - 0.89)
    pf['P0'] = pf['P0'] + P0_offset
    pf['P1'] = pf['P1'] + P1_offset
    pf['P2'] = pf['P2'] + P2_offset
    pf['calib_offset_correction'] = [True]*3 # coresponding to P0, P1, P2
    
    pf['Pmid'] = pf['P0'] + 0.3 # shift to the position of drogue
    pf['errP'] = 0.124
    ep['Pmid'] = np.interp(ep['yd'], pf['ydP'], pf['Pmid'])


def crt_nbf_pressure(site, cyear, ep):
    """
    Adjust the float's pressure record to remove variations due to atmopsheric pressure changes,
    using the 12-hour mean of atmopsheric pressure before deployment as the pre-calibration value.
    """
    # barometric pressure
    # t_bp,bp = globals()['read_'+site+'_bp'](cyear)
    if platform == 'linux':
        GDrive_root = '/media/zhihua/internal1/GDrive/'
    elif platform == 'darwin':
        GDrive_root = '/Volumes/GoogleDrive/My Drive/'
    site_root = GDrive_root + 'UW/Research/Data/' + site + '/'
    met_dir = site_root + 'Mooring/' + str(cyear) + '_high_res/'
    
    with xr.open_dataset(met_dir+'bp50n145w_10m.cdf') as ds:
        bp = ds.BP_915.where(ds.BP_915<1e5).assign_coords(yd=('time', sot.pytime2yd(ds.time)))
    
    ep['pytime'] = sot.mtime2pytime(ep1['Mtime'])
    # ep['pydnum'] = [nc.date2num(d,dnum_unit,'proleptic_gregorian') for d in ep['pytime']]
    
    # fill missing data in barometric pressure by linear interpolation
    bp.interpolate_na(dim='time', method='linear')
    # dnum_bp = nc.date2num(t_bp,dnum_unit,'proleptic_gregorian')
    # bp_is_nan = np.isnan(bp)
    # bp_nonan = np.interp(dnum_bp,dnum_bp[~bp_is_nan],bp[~bp_is_nan])
    
    # average of barometric pressure, 12-hours before float deployment [pre-calibration]
    precal_idx = (bp.yd > (ep['yd'][0]-0.5)) & (bp.yd <= ep['yd'][0])
    bp_precal  = bp[precal_idx].mean()
    
    # interpolate barometic pressure to float time 
    # and compute changes relative to the pre-calibration value
    aPair = (bp.interp(time=ep['pytime']) - bp_precal)/100 # [dbar], 1 hPa = 0.01 dbar
    ep['Pmid'] = ep['Pav'] - aPair.values


def read_ocsp_bp(cyear):
    """
    """
    bp_dir = '/Users/zhihua/GDrive/UW/Research/Data/Papa/Mooring/' + str(cyear) + \
              '_high_res/'
    h_nc = nc.Dataset(bp_dir+'bp50n145w_10m.cdf',mode='r')

    time_var = h_nc.variables['time']
    time_bp  = nc.num2date(time_var[:],time_var.units,
                           only_use_cftime_datetimes=False,only_use_python_datetimes=True)
    bp = np.squeeze(h_nc.variables['BP_915'][:]) # [hPa]
    bp = np.ma.masked_greater(bp,1e5)
    h_nc.close()

    # masked array to ndarray with nan
    time_bp = np.ma.filled(time_bp,nan)
    bp = np.ma.filled(bp.astype(float),nan)
    return time_bp, bp


def read_lkwa_bp(cyear):
    """
    """
    bp_dir = '/Users/zhihua/GDrive/UW/Research/Data/LakeWashington/Met/'
    met = sio.loadmat(bp_dir+'LkWash'+str(cyear)+'_forcing.mat',
                      squeeze_me=True, struct_as_record=False)
    
    time_bp = np.array([sot.mtime2pytime(d) for d in met['time']])
    bp = met['P'] # [hPa]
    return time_bp, bp


def read_ocsp_flux(cyear):
    """
    """
    flux_dir = '/Users/zhihua/GDrive/UW/Research/Data/Papa/Mooring/' + str(cyear) + \
               '_high_res/fluxes/'

    h_nc = nc.Dataset(flux_dir+'tau50n145w_hr.cdf', mode='r')
    time_var = h_nc.variables['time']
    time = nc.num2date(time_var[:],time_var.units,
                       only_use_cftime_datetimes=False,only_use_python_datetimes=True)
    lon  = np.squeeze(h_nc.variables['lon'][:])
    lat  = np.squeeze(h_nc.variables['lat'][:])
    tau  = np.squeeze(h_nc.variables['TAU_440'][:]) # total wind stress [N/m^2]
    taux = np.squeeze(h_nc.variables['TX_442'][:]) # zonal wind stress [N/m^2]
    tauy = np.squeeze(h_nc.variables['TY_443'][:]) # meridional wind stress [N/m^2]
    h_nc.close()
    
    h_nc = nc.Dataset(flux_dir+'wind_10meter_50n145w_hr.cdf',mode='r')
    wsp = np.squeeze(h_nc.variables['WZS_2401'][:]) # 10 meter wind speed [m/s]
    u10 = np.squeeze(h_nc.variables['UZS_2422'][:]) # 10 meter zonal wind [m/s]
    v10 = np.squeeze(h_nc.variables['VZS_2423'][:]) # 10 meter meridional wind [m/s]
    h_nc.close()
    
    h_nc = nc.Dataset(flux_dir+'swnet50n145w_hr.cdf',mode='r')
    nsw  = np.squeeze(h_nc.variables['SWN_1495'][:]) # net shortwave radiation [W/m^2], into the ocean [+]
    h_nc.close()
    
    h_nc = nc.Dataset(flux_dir+'lwnet50n145w_hr.cdf',mode='r')
    nlw  = -np.squeeze(h_nc.variables['LWN_1136'][:]) # net longwave radiation [W/m^2], into the ocean [+]
    h_nc.close()
    
    h_nc = nc.Dataset(flux_dir+'qlat50n145w_hr.cdf',mode='r')
    hlb  = -np.squeeze(h_nc.variables['QL_137'][:]) # latent heat flux [W/m^2], into the ocean [+]
    h_nc.close()
    
    h_nc = nc.Dataset(flux_dir+'qsen50n145w_hr.cdf',mode='r')
    hsb  = -np.squeeze(h_nc.variables['QS_138'][:]) # sensible heat flux [W/m^2], into the ocean [+]
    h_nc.close()
    
    h_nc = nc.Dataset(flux_dir+'evap50n145w_hr.cdf',mode='r')
    evap = np.squeeze(h_nc.variables['E_250'][:]) # evaporation rate [mm/hr]
    h_nc.close()
    
    h_nc = nc.Dataset(flux_dir+'rain_wspd_cor50n145w_hr.cdf',mode='r')
    rain = np.squeeze(h_nc.variables['RN_485'][:]) # rain rate [mm/hr]
    h_nc.close()
    
    h_nc = nc.Dataset(flux_dir+'sst50n145w_hr.cdf',mode='r')
    sst  = np.squeeze(h_nc.variables['T_25'][:]) # surface temperature [C]
    h_nc.close()
    
    h_nc = nc.Dataset(flux_dir+'sss50n145w_hr.cdf',mode='r')
    sss  = np.squeeze(h_nc.variables['S_41'][:]) # surface salinity [psu]
    h_nc.close()
    
    # mask missing data
    tau  = np.ma.masked_greater(tau,1e5)
    taux = np.ma.masked_greater(taux,1e5)
    tauy = np.ma.masked_greater(tauy,1e5)
    wsp  = np.ma.masked_greater(wsp,1e5)
    u10  = np.ma.masked_greater(u10,1e5)
    v10  = np.ma.masked_greater(v10,1e5)
    nsw  = np.ma.masked_greater(nsw,1e5)
    nlw  = np.ma.masked_greater(nlw,1e5)
    hlb  = np.ma.masked_greater(hlb,1e5)
    hsb  = np.ma.masked_greater(hsb,1e5)
    evap = np.ma.masked_greater(evap,1e5)
    rain = np.ma.masked_greater(rain,1e5)
    sst  = np.ma.masked_greater(sst,1e5)
    sss  = np.ma.masked_greater(sss,1e5)
    
    # convert masked array to ndarray with nan
    time = np.ma.filled(time,nan)
    lon  = np.ma.filled(lon,nan)
    lat  = np.ma.filled(lat,nan)
    tau  = np.ma.filled(tau.astype(float),nan)
    taux = np.ma.filled(taux.astype(float),nan)
    tauy = np.ma.filled(tauy.astype(float),nan)
    wsp  = np.ma.filled(wsp.astype(float),nan)
    u10  = np.ma.filled(u10.astype(float),nan)
    v10  = np.ma.filled(v10.astype(float),nan)
    nsw  = np.ma.filled(nsw.astype(float),nan)
    nlw  = np.ma.filled(nlw.astype(float),nan)
    hlb  = np.ma.filled(hlb.astype(float),nan)
    hsb  = np.ma.filled(hsb.astype(float),nan)
    evap = np.ma.filled(evap.astype(float),nan)
    rain = np.ma.filled(rain.astype(float),nan)
    sst  = np.ma.filled(sst.astype(float),nan)
    sss  = np.ma.filled(sss.astype(float),nan)
    
    # stress direction
    taud = np.arctan2(tauy,taux)*180/pi # [degree, anti-clockwise to the East]
    U10  = np.sqrt(u10**2 + v10**2)
    
    SF = build_SF(time,lon,lat,sss,sst,tau,taud,U10,evap,rain,nlw,nsw,hlb,hsb)
    return SF


def read_lkwa_flux(cyear):
    """
    """
    met_dir = '/Users/zhihua/GDrive/UW/Research/Data/LakeWashington/Met/'
    met = sio.loadmat(met_dir+'LkWash'+str(cyear)+'_forcing.mat',
                      squeeze_me=True,struct_as_record=False)
    
    time = np.array([sot.mtime2pytime(d) for d in met['time']])
    lat,lon = 47.6118, -122.2613
    sss = np.full(len(time), 0)
    
    # stress direction [degree, anti-clockwise to the East]
    taud = np.degrees([cmath.phase(a) for a in met['W'].tolist()])
    
    # heat of evaporation for water
    Le = gsw.latentheat_evap_t(sss, met['SST']) # [J/kg]

    rho_fw = 1000
    evap = -met['HF_latent']/Le/rho_fw*1e3*3600 # [mm/hr], positive
    rain = np.full(len(time), 0)
    
    SF = build_SF(time,lon,lat,sss,met['SST'],met['tau'],taud,met['WSPD_met'],evap,rain,
                  met['HF_longwave'],met['HF_shortwave'],met['HF_latent'],met['HF_sensible'])
    return SF


def read_labsea_flux(cyear):
    """
    """
    met_dir = '/Users/zhihua/GDrive/UW/Research/Data/LabradorSea/Met/Ship_Knorr/'
    if cyear == 1997:
        h_nc = nc.Dataset(met_dir+'dataset1.cdf',mode='r')
    elif cyear == 1998:
        h_nc = nc.Dataset(met_dir+'dataset2.cdf',mode='r')
    
    h_nc.set_auto_mask(False)
    lon = h_nc.variables['lon'][:]
    lat = h_nc.variables['lat'][:]
    
    met_yd   = h_nc.variables['julian_day'][:]
    met_year = h_nc.variables['year'][:]
    met_day  = h_nc.variables['day'][:]
    met_mon  = h_nc.variables['month'][:]
    met_hr   = h_nc.variables['hour'][:]
    met_min  = h_nc.variables['minute'][:]
    met_sec  = h_nc.variables['second'][:]
    
    U10 = h_nc.variables['tw'][:] # not sure about wind measurement height
    tau = h_nc.variables['tau'][:]
    twd = h_nc.variables['twd'][:]
    hsb = -h_nc.variables['shf'][:]
    hlb = -h_nc.variables['lhf'][:]
    lwd = h_nc.variables['lwd'][:]
    lwu = h_nc.variables['lwu'][:]
    swd = h_nc.variables['swd'][:]
    swu = h_nc.variables['swu'][:]
    sst = h_nc.variables['sst'][:]
    h_nc.close()
    
    _,time = sot.tcomps2yd(met_year,met_mon,met_day,met_hr,met_min,met_sec)
    sss = np.full(len(time),35)
    nlw = lwd - lwu
    nsw = swd - swu
    
    # heat of evaporation for water
    Le = gsw.latentheat_evap_t(sss,sst) # [J/kg]

    rho_fw = 1000
    evap = -hlb/Le/rho_fw*1e3*3600 # [mm/hr], positive
    rain = np.full(len(time),0)

    SF = build_SF(time,lon,lat,sss,sst,tau,twd,U10,evap,rain,
                  nlw,nsw,hlb,hsb)
    return SF


# def read_spursii_flux():
#     """
#     """
#     flux_dir = '/Users/zhihua/GDrive/UW/Research/Data/SPURSII/ERA5/'

#     h_nc = nc.Dataset(flux_dir + 'SPURSII_met.nc', mode='r')
        
#     SF = build_SF(time,lon,lat,sss,sst,tau,taud,evap,rain,nlw,nsw,hlb,hsb)
#     return SF


def build_SF(time, lon, lat, sss, sst, tau, taud, U10,
             evap, rain, nlw, nsw, hlb, hsb):
    """
    """
    # time to yearday, time-string
    yd = sot.pytime2yd(time)
    tstr = [a.strftime('%Y-%m-%d %H:%M:%S') for a in time]
    
    # thermodynamics
    ssSA  = gsw.SA_from_SP(sss, 1, lon, lat)
    ssCT  = gsw.CT_from_t(ssSA, sst, 1)
    alpha = gsw.density.alpha(ssSA, ssCT, 1)
    beta  = gsw.density.beta(ssSA, ssCT, 1)
    
    # Momentum flux
    wmom_0 = tau/rho0
    
    # Salinity flux
    emp  = (evap - rain)/1e3/3600 # [m/s]
    ws_0 = -emp*sss
    
    # Combine heat fluxes
    # TODO: penetrative solar radiation
    Qtur = nlw + hlb + hsb
    Qnet = nsw + Qtur
    wt_0 = -Qtur/rho0/cp
    
    # Compute buoyancy fluxes [m^2/s^3]
    wb_0 = g*(alpha*wt_0 - beta*ws_0)
    wb_net_t = -g*alpha*(Qnet/rho0/cp)
    wb_net_s = -g*beta*ws_0
    wb_net = wb_net_t + wb_net_s
    
    # Dictionary
    SF = {'time': {'str': tstr, 'yd': yd},
          'stress': {'mag': tau, 'dir': taud, 'kf': wmom_0, 'U10': U10},
          'heatF': {'net': Qnet, 'tur': Qtur, 'sw': nsw, 'lw': nlw, 'qe': hlb, 'qs': hsb, 'kf': wt_0},
          'saltF': {'emp': emp, 'kf': ws_0},
          'buoyF': {'net': wb_net, 'kf': wb_0, 'sc': wb_net_s, 'tc': wb_net_t}}
    return SF


def interp_field2float(site, flon, flat, fyd):
    """
    """
    data_root = '/Users/zhihua/GDrive/UW/Research/Data/'
    if site=='spursii':
        ERAnc_name  = data_root + 'SPURSII/ERA5/SPURSII_met.nc'
        SMAPnc_name = data_root + 'SPURSII/Sat/SMAP_L3_SSS_CAP_8DAY-RUNNINGMEAN_V5.nc'
    
    h_nc = nc.Dataset(ERAnc_name, mode='r')
    h_nc.set_auto_mask(False)
    time_var = h_nc.variables['time']
    time = nc.num2date(time_var[:], time_var.units,
                       only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    # ERA mean field has timestamp at the end of averaging window
    time = time - timedelta(hours=0.5)
    lon  = h_nc.variables['longitude'][:]
    lat  = h_nc.variables['latitude'][:]
    u10  = h_nc.variables['u10'][:] # instantaneous
    v10  = h_nc.variables['v10'][:] # instantaneous
#     dim  = h_nc.variables['metss'].dimensions
    taux = h_nc.variables['metss'][:] # zonal wind stress [N/m^2]
    tauy = h_nc.variables['mntss'][:] # meridional wind stress [N/m^2]
    nsw  = h_nc.variables['msnswrf'][:] # net shortwave radiation [W/m^2], into the ocean [+]
    nlw  = h_nc.variables['msnlwrf'][:] # net longwave radiation [W/m^2], into the ocean [+]
    hlb  = h_nc.variables['mslhf'][:] # latent heat flux [W/m^2], into the ocean [+]
    hsb  = h_nc.variables['msshf'][:] # sensible heat flux [W/m^2], into the ocean [+]
    evap = -h_nc.variables['mer'][:] * 3600 # evaporation [+] rate [mm/hr]
    rain = h_nc.variables['mtpr'][:] * 3600 # rain rate [mm/hr]
    sst  = h_nc.variables['sst'][:] - 273.15 # surface temperature [C]
    h_nc.close()
    
    h_nc = nc.Dataset(SMAPnc_name, mode='r')
    h_nc.set_auto_mask(False)
#     hpd = getattr(h_nc, 'Gaussian_window_half_power')
    time_var = h_nc.variables['times']
    smap_time = nc.num2date(time_var[:], time_var.units,
                            only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    smap_lon = h_nc.variables['longitude'][:]
    smap_lat = h_nc.variables['latitude'][:]
#     smap_dim = h_nc.variables['smap_sss'].dimensions
    smap_sss = h_nc.variables['smap_sss'][:] # surface salinity [psu]
    h_nc.close()
    
    # transpose to the conventional dimensional order
#     dimc = ['latitude', 'longitude', 'time']
#     dim_order = [dim.index(a) for a in dimc]
    # permutation: the i'th axis will correspond to dim[ dim_order[i] ]
    
    # flip latitude to make it ascending
    lat  = np.flip(lat)
    sst  = np.flip(sst, axis=1)
    taux = np.flip(taux,axis=1)
    tauy = np.flip(tauy,axis=1)
    u10  = np.flip(u10, axis=1)
    v10  = np.flip(v10, axis=1)
    nsw  = np.flip(nsw, axis=1)
    nlw  = np.flip(nlw, axis=1)
    hlb  = np.flip(hlb, axis=1)
    hsb  = np.flip(hsb, axis=1)
    evap = np.flip(evap,axis=1)
    rain = np.flip(rain,axis=1)
    smap_lat = np.flip(smap_lat)
    smap_sss = np.flip(smap_sss, axis=1)
    
    # interpolate SMAP salinity to ERA5 grid
#     sss   = np.zeros(sst.shape)
#     eps_r = hpd/np.sqrt(np.log(2))/110 # [degree], not exactly, but close enough
#     Slon,Slat = np.meshgrid(smap_lon, smap_lat)
#     Lon,Lat   = np.meshgrid(lon, lat)
#     for i in range(smap_time.size):
#         rbfiS = Rbf(Slon, Slat, smap_sss[:,:,i], function='gaussian', epsilon=eps_r)
#         sss[:, :, i*24:(i+1)*24] = np.broadcast_to(rbfiS(Lon, Lat)[...,None],
#                                                    Lon.shape+(24,))
    # interp to float positions
    yd      = sot.pytime2yd(time)
    smap_yd = sot.pytime2yd(smap_time)
#     fsss,fsst,ftaux,ftauy,fu10,fv10,fevap,frain,fnlw,fnsw,fhlb,fhsb = np.zeros((12, fyd.size))
#     lonr,latr = Lon.ravel(), Lat.ravel()
#     for i,idx in enumerate(idx_fyd):

    # default: linear
    rgi_sss  = RegularGridInterpolator((smap_yd, smap_lat, smap_lon), smap_sss)
    rgi_sst  = RegularGridInterpolator((yd, lat, lon), sst)
    rgi_taux = RegularGridInterpolator((yd, lat, lon), taux)
    rgi_tauy = RegularGridInterpolator((yd, lat, lon), tauy)
    rgi_u10  = RegularGridInterpolator((yd, lat, lon), u10)
    rgi_v10  = RegularGridInterpolator((yd, lat, lon), v10)
    rgi_evap = RegularGridInterpolator((yd, lat, lon), evap)
    rgi_rain = RegularGridInterpolator((yd, lat, lon), rain)
    rgi_nlw  = RegularGridInterpolator((yd, lat, lon), nlw)
    rgi_nsw  = RegularGridInterpolator((yd, lat, lon), nsw)
    rgi_hlb  = RegularGridInterpolator((yd, lat, lon), hlb)
    rgi_hsb  = RegularGridInterpolator((yd, lat, lon), hsb)
    
    fsss  = rgi_sss(  np.array([fyd, flat, flon]).T )
    fsst  = rgi_sst(  np.array([fyd, flat, flon]).T )
    ftaux = rgi_taux( np.array([fyd, flat, flon]).T )
    ftauy = rgi_tauy( np.array([fyd, flat, flon]).T )
    fu10  = rgi_u10(  np.array([fyd, flat, flon]).T )
    fv10  = rgi_v10(  np.array([fyd, flat, flon]).T )
    fevap = rgi_evap( np.array([fyd, flat, flon]).T )
    frain = rgi_rain( np.array([fyd, flat, flon]).T )
    fnlw  = rgi_nlw(  np.array([fyd, flat, flon]).T )
    fnsw  = rgi_nsw(  np.array([fyd, flat, flon]).T )
    fhlb  = rgi_hlb(  np.array([fyd, flat, flon]).T )
    fhsb  = rgi_hsb(  np.array([fyd, flat, flon]).T )
    
    # stress direction
    ftaud = np.arctan2(ftauy, ftaux)*180/pi # [degree, anti-clockwise from the East]
    ftau  = np.sqrt(ftauy**2 + ftaux**2)
    fU10  = np.sqrt(fu10**2 + fv10**2)
    idx_fyd = np.where(np.in1d(yd, fyd))[0]
    SF = build_SF(time[idx_fyd], flon, flat, fsss, fsst, ftau, ftaud, fU10,
                  fevap, frain, fnlw, fnsw, fhlb, fhsb)
    return SF


def prep_float_data(fpath, fname):
    """
    Perform quality control for raw Langrangian float data.
    Input file is in netcdf format.
    """
    with xr.open_dataset(fpath+fname) as ds:
        Env = ds.load()
    
        # preliminary QC: remove near-surface salinity spikes
        d_gt_3 = Env.ctd_depth[1] > 3 # top CTD
        d_surf = Env.ctd_depth[1].values.copy()
        d_surf[d_gt_3] = nan
        _,Jsurf_clp = sot.get_clumps(d_surf)

        for j,jsurf in enumerate(Jsurf_clp):
            z = -Env.ctd_depth[:,jsurf]
            S =  Env.ctd_salinity[:,jsurf]
            ndata = jsurf.stop - jsurf.start
        
            if ndata < 3:
                iStiny = np.any(S.values<30, axis=0)
                if iStiny.sum() > 0:
                    print(f'salinity < 30 in short surfacing segment j={j}')
                    Env.float_depth.values[jsurf][ iStiny ] = nan
                if ndata > 1:
                    if S.diff('ctd_time').max().values > 1:
                        print(f'salinity difference > 1 in short surfacing segment j={j}')
                        Env.float_depth.values[jsurf] = nan
                continue

            # remove chunks where depth oscillates
            jzosc_s = sot.first_turn(z[0].values) + 1 # +1 to get the point after turning
            if jzosc_s > 1: # there is at least 1 turning point
                jzosc_e = sot.last_turn(z[0].values)
                if jzosc_e > jzosc_s: # there is more than 1 turning point
                    Env.float_depth.values[jsurf][ jzosc_s : jzosc_e ] = nan

            # remove chunks where Salinity spikes
            dz = z.diff('ctd_time').values
            dS = S.diff('ctd_time').values
            dz[abs(dz)<1e-2] = 1e-2 # minimal resolvable depth difference: 1cm
            dS_dz = dS/dz
            jSspike = np.any(np.round(abs(dS_dz))>=10, axis=0)
            if jSspike.sum() > 0:
                jsspk_s = np.where(jSspike)[0][0] + 1 # +1 -> spike indices in original arrays
                if jSspike.sum() > 1:
                    jsspk_e = np.where(jSspike)[0][-1] + 1
#                     jsspk_affected = np.arange( min(0, jsspk_s-1), max(ndata, jsspk_e+1) )
                    Env.float_depth.values[jsurf][ jsspk_s:jsspk_e ] = nan
                else:
                    jsspk_e = jsspk_s.copy()
                    Env.float_depth.values[jsurf][ :jsspk_e ] = nan

        Jnull = np.where( np.isnan(Env.float_depth) )[0]
        Env_qcd = Env.drop_isel(ctd_time=Jnull)
        print(f'Removed {Env.float_depth.size - Env_qcd.float_depth.size} low-quality measurements')
#         Env_qcd = Env_qcd.where(Env_qcd.ctd_temperature > 1)
#         Env_qcd = Env_qcd.where(Env_qcd.ctd_salinity > 30)
        Env_qcd.update( {'ctd_temperature': Env.ctd_temperature.where(Env.ctd_temperature > 1)} )
        Env_qcd.update( {'ctd_salinity': Env.ctd_salinity.where(Env.ctd_salinity > 30)} )

        new_fname = fname.replace('.nc', '_qcd.nc')
        Env_qcd.to_netcdf(fpath+new_fname, mode='w')
        print(f"Quality controlled file saved as '{new_fname}'")