import xarray as xr
import numpy as np
import sca_osbl_tool as sot
import scipy.io as sio
import gsw
from constants import g, nan
from sys import platform


def _crt_pressure(GDrive_root, floatID):
    site_root = GDrive_root + 'UW/Research/Data/LakeWA/'
    moor_dir = site_root + 'Mooring/'
    nbf_dir = site_root + 'NBF/'
    env_name = 'Lake2011Env'+str(floatID)+'.mat'
    FPR_name = 'FPR'+str(floatID)+'.mat'
    year = 2011

    ep = sio.loadmat(nbf_dir+env_name, squeeze_me=True, struct_as_record=False)
    pytime = sot.yd2pytime(ep['yd'], year)

    pf = sio.loadmat(nbf_dir+FPR_name, squeeze_me=True, struct_as_record=False)
    # trim fast pressure to just cover Env time
    if pf['ydP'][-1] > ep['yd'][-1]:
        last_pf = np.where( pf['ydP'] >= ep['yd'][-1] )[0][0] + 1
        for key in pf.keys():
            if not key.startswith('__'):
                pf[key] = pf[key][:last_pf]
    pytimeP = sot.yd2pytime(pf['ydP'], year)

    # No GPS for lake floats
    lon = np.full_like(ep['yd'], -122.2613)
    lat = np.full_like(ep['yd'], 47.6118)

    frc = sio.loadmat(moor_dir+'LkWash_forcing.mat', squeeze_me=True, struct_as_record=False)
    bp = xr.DataArray(data=frc['P'], dims=('time'), coords={'time': sot.mtime2pytime(frc['time'])},
                      attrs={'units': 'mbar [hPa]'})
    bp = bp.assign_coords(yd=('time', sot.pytime2yd(bp.time))) \
           .sel(time=slice('2011-10-01', '2011-12-01'))
    # bp = bp.interpolate_na(dim='time', method='linear') # no nan in this dataset
    
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
    
    # mean pressure at the end of upward profiling
    _,Iclump = sot.get_clumps(ep['up'])
    P0_surf = np.full(len(Iclump), nan)
    for i,iup_ep in enumerate(Iclump):
        Pup = ep['P'][iup_ep]
        ydup = ep['yd'][iup_ep]
        if (np.min(Pup) > 5) | (len(Pup) < 10): continue
        ysup = np.around(ydup*24*3600)
        dPdt = np.gradient(Pup, ysup)
        if np.sum(np.abs(dPdt) > 0.02) == 0:
            i_last_up = 0
        else:
            i_last_up = np.where(np.abs(dPdt) > 0.02)[0][-1]
            if i_last_up == len(dPdt)-1: continue    
        isurf_pf = (pf['ydP'] > ep['yd'][iup_ep][i_last_up]) & (pf['ydP'] <= (ep['yd'][iup_ep][-1] + 30/24/3600))
        if np.sum(isurf_pf) < 3: continue
        P0_surf[i] = np.mean(pf['P0'][isurf_pf])

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

    # only use Env data that is also in fast pressure
    env_in_fpr = np.isin(np.around(ep['yd']*24*60), np.around(pf['ydP']*24*60)) # where env time is in FPR (fast pressure record)
    not_in_fpr = np.ones(env_in_fpr.shape)
    not_in_fpr[env_in_fpr] = np.nan
    _,Iextra_env = sot.get_clumps(not_in_fpr)
    mask = np.ones_like(env_in_fpr)
    for i in Iextra_env:
        mask[i] = False

    env = xr.Dataset(
        data_vars={'P0': (('timeP'), P0, {'units': 'dbar', 'standard_name': 'fast_pressure_top'}),
                   'P1': (('timeP'), P1, {'units': 'dbar', 'standard_name': 'fast_pressure_middle'}),
                   'P2': (('timeP'), P2, {'units': 'dbar', 'standard_name': 'fast_pressure_bottom'}),
                   'Pc': (('timeP'), Pc, {'units': 'dbar', 'standard_name': 'fast_fake_pressure_center'}),
                   'P': (('time'), P[mask], {'units': 'dbar', 'standard_name': 'fake_pressure_for_TS'}),
                   'T': (('time'), ep['T'][mask], {'units': 'C', 'standard_name': 'temperature'}),
                   'S': (('time'), ep['S'][mask], {'units': 'psu', 'standard_name': 'salinity'}),
                   'B': (('time'), ep['B'][mask], {'units': 'cubic centimeter (cc)', 'standard_name': 'float_piston_volume'}),
                   'mode': (('time'), ep['mode'][mask], {'standard_name': 'float_operating_mode'}),
                   'drift': (('time'), ep['drift'][mask], {'standard_name': 'Lagrangian_drift_number'}),
                   'settle': (('time'), ep['settle'][mask], {'standard_name': 'settle_number'}),
                   'up': (('time'), ep['up'][mask], {'standard_name': 'upward_profile_number'}),
                   'down': (('time'), ep['down'][mask], {'standard_name': 'downward_profile_number'}),
                   'lon': (('time'), lon[mask], {'units': 'degree', 'long_name': 'longitude'}),
                   'lat': (('time'), lat[mask], {'units': 'degree', 'long_name': 'latitude'})},
        coords={'ydP': (('timeP'), pf['ydP'], {'standard_name': 'fast_yearday_of_'+str(year)}),
                'timeP': (('timeP'), pytimeP),
                'yd': (('time'), ep['yd'][mask], {'standard_name': 'yearday_of_'+str(year)}),
                'time': (('time'), pytime[mask])},
        attrs={'description': 'Raw data from the Lagrangian float '+str(floatID)+' deployed in Lake Washington (2011)',
               'note': 'All pressures have been corrected for atmospheric pressure and calibration offsets'})
    
    # use data before surface boundry layer mixed to the bottom (D'Asaro et al. 2014 supp.)
    env = env.sel(time=slice('2011-10-01', '2011-11-28'), timeP=slice('2011-10-01', '2011-11-28'))
    nc_name = 'LKWA_env_' + str(floatID) + '.nc'
    env.to_netcdf(nbf_dir + nc_name, engine='netcdf4')
    print(f'Float data: {nc_name} saved at {nbf_dir}.\n')


if platform == 'linux':
    Groot = '/media/zhihua/internal1/GDrive/'
elif platform == 'darwin':
    Groot = '/Volumes/GoogleDrive/My Drive/'
_crt_pressure(Groot, 71)
_crt_pressure(Groot, 72)