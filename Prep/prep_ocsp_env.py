import xarray as xr
import numpy as np
import sca_osbl_tool as sot
import scipy.io as sio
import gsw
from constants import g, nan
from sys import platform


def _crt_pressure(GDrive_root, year):
    site_root = GDrive_root + 'UW/Research/Data/OCSP/'
    moor_dir = site_root + 'Mooring/' + str(year) + '_high_res/'
    nbf_dir = site_root + 'NBF/'

    if year==2011:
        env_name = 'EnvPapaIraw.mat'
        FPR_name = 'Papa2011_Pfast.mat'
    elif year==2012:
        env_name = 'EnvPapaIIraw.mat'
        FPR_name = 'Papa2012_Pfast.mat'
    else:
        raise ValueError('Year number not supprted!')

    ep = sio.loadmat(nbf_dir+env_name, squeeze_me=True, struct_as_record=False)
    pytime = sot.yd2pytime(ep['yd'], year)
    # ep['pydnum'] = [nc.date2num(d,dnum_unit,'proleptic_gregorian') for d in ep['pytime']]

    pf = sio.loadmat(nbf_dir+FPR_name, squeeze_me=True, struct_as_record=False)
    # trim fast pressure to just cover Env time
    last_pf = np.where( pf['ydP'] >= ep['yd'][-1] )[0][0] + 1
    for key in pf.keys():
        if not key.startswith('__'):
            pf[key] = pf[key][:last_pf]
    pytimeP = sot.yd2pytime(pf['ydP'], year)

    # interpolate GPS track to the entire operation period, excluding points with less than 5 satellites
    lon = np.interp(ep['yd'], ep['GPS'].yd[ep['GPS'].nsats>4], ep['GPS'].lon[ep['GPS'].nsats>4])
    lat = np.interp(ep['yd'], ep['GPS'].yd[ep['GPS'].nsats>4], ep['GPS'].lat[ep['GPS'].nsats>4])
    
    with xr.open_dataset(moor_dir+'bp50n145w_10m.cdf') as ds:
        bp = ds.BP_915.where(ds.BP_915<1e5).assign_coords(yd=('time', sot.pytime2yd(ds.time)))
        bp = bp.interpolate_na(dim='time', method='linear')
    
    # average of barometric pressure, 12-hours before float deployment [pre-calibration]
    precal_idx = (bp.yd > (pf['ydP'][0]-0.5)) & (bp.yd <= pf['ydP'][0])
    bp_precal  = bp[precal_idx].mean().values
    
    # interpolate to float time, compute changes relative to the pre-calibration value
    aPair = (bp.interp(time=pytimeP) - bp_precal)/100 # [dbar], 1 hPa = 0.01 dbar
    
    # atmospheric pressure correction
    pf['P0'] = pf['P0'] - aPair.squeeze().values
    pf['P1'] = pf['P1'] - aPair.squeeze().values
    pf['P2'] = pf['P2'] - aPair.squeeze().values
    
    mean_P2mP0 = (pf['P2']-pf['P0']).mean()
    mean_P1mP0 = (pf['P1']-pf['P0']).mean()
    mean_P2mP1 = (pf['P2']-pf['P1']).mean()

    # minium pressure during drifts
    Ndrift = np.nanmax(ep['drift']).astype(int)
    P0_surf = np.full(Ndrift, nan)
    for j in range(Ndrift):
        i = j+1
        id_ep = np.nonzero(ep['drift']==i)[0]
        if len(id_ep) < 121: continue
        is_drift = (pf['ydP'] >= ep['yd'][id_ep[0]]) & (pf['ydP'] <= ep['yd'][id_ep[-1]])
        P0_surf[j] = np.nanmin(pf['P0'][is_drift])
    P0_surf[P0_surf > 10] = nan
    
    # nudge pressure differences to geometry
    P0_surf_mean = np.around(np.nanmean(P0_surf), 3)
    print(f'The top pressure sensor is {P0_surf_mean} m below the surface on average.')
    P0_offset = np.around(0.461-P0_surf_mean, 3)
    print(f'Applying offset of {P0_offset} m to the top pressure record.')
    P1_offset = P0_offset - (mean_P1mP0 - 0.3)
    P2_offset = P0_offset - (mean_P2mP0 - 0.89)
    P0 = pf['P0'] + P0_offset
    P1 = pf['P1'] + P1_offset
    P2 = pf['P2'] + P2_offset
    
    Sig0P = 1e3 + np.interp(pf['ydP'], ep['yd'], ep['Sig0']) # potential density (Unesco 1983)
    Pc = P0 + Sig0P*g*0.445/1e4 # hydrostatically shift to the float's geomertic center, Pa to dbar
    Pts = P0 + Sig0P*g*1.15/1e4 # hydrostatically shift to the float's bottom CT sensor, Pa to dbar
    P = np.interp(ep['yd'], pf['ydP'], Pts) # try time averaging?

    env = xr.Dataset(
        data_vars={'P0': (('timeP'), P0, {'units': 'dbar', 'standard_name': 'fast_pressure_top'}),
                   'P1': (('timeP'), P1, {'units': 'dbar', 'standard_name': 'fast_pressure_middle'}),
                   'P2': (('timeP'), P2, {'units': 'dbar', 'standard_name': 'fast_pressure_bottom'}),
                   'Pc': (('timeP'), Pc, {'units': 'dbar', 'standard_name': 'fast_fake_pressure_center'}),
                   'P': (('time'), P, {'units': 'dbar', 'standard_name': 'fake_pressure_for_TS'}),
                   'T': (('time'), ep['T'], {'units': 'C', 'standard_name': 'temperature'}),
                   'S': (('time'), ep['S'], {'units': 'psu', 'standard_name': 'salinity'}),
                   'B': (('time'), ep['B'], {'units': 'cubic centimeter (cc)', 'standard_name': 'float_piston_volume'}),
                   'mode': (('time'), ep['mode'], {'standard_name': 'float_operating_mode'}),
                   'drift': (('time'), ep['drift'], {'standard_name': 'Lagrangian_drift_number'}),
                   'settle': (('time'), ep['settle'], {'standard_name': 'settle_number'}),
                   'up': (('time'), ep['up'], {'standard_name': 'upward_profile_number'}),
                   'down': (('time'), ep['down'], {'standard_name': 'downward_profile_number'}),
                   'lon': (('time'), lon, {'units': 'degree', 'standard_name': 'longitude_linearly_interpolated'}),
                   'lat': (('time'), lat, {'units': 'degree', 'standard_name': 'latitude_linearly_interpolated'})},
        coords={'ydP': (('timeP'), pf['ydP'], {'standard_name': 'fast_yearday_of_'+str(year)}),
                'timeP': (('timeP'), pytimeP),
                'yd': (('time'), ep['yd'], {'standard_name': 'yearday_of_'+str(year)}),
                'time': (('time'), pytime)},
        attrs={'description': 'Raw data from the Lagrangian float deployed in' + str(year) + 'near ocean climate station Papa',
               'note': 'All pressures have been corrected for atmospheric pressure and calibration offsets'})

    nc_name = 'OCSP_env_' + str(year) + '.nc'
    env.to_netcdf(nbf_dir + nc_name, engine='netcdf4')
    print(f'Float data: {nc_name} saved at {nbf_dir}.\n')


if platform == 'linux':
    Groot = '/media/zhihua/internal1/GDrive/'
elif platform == 'darwin':
    Groot = '/Volumes/GoogleDrive/My Drive/'
_crt_pressure(Groot, 2011)
_crt_pressure(Groot, 2012)