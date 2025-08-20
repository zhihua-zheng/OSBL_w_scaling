import xarray as xr
import numpy as np
import pandas as pd
import sca_osbl_tool as sot
import scipy.io as sio
import gsw
from tqdm.auto import tqdm
from constants import nan
from sys import platform
from scipy import stats


def _prep_fanapi_gridded(GDrive_root, floatID):
    site_root = GDrive_root + 'UW/Research/Data/Hurricanes/ITOP/'
    nbf_dir = site_root + 'NBF/'
    year = 2010
    depth_max = 120

    with xr.open_dataset(nbf_dir+'Fanapi_env_'+str(floatID)+'.nc') as E:
        mlon = np.mean(E.lon.values)
        mlat = np.mean(E.lat.values)
        S = E.S.values
        T = E.T.values
        P = E.P.values
        mode = E.mode.values
        yd = E.yd.values
        pytime = E.time.values

    SA = gsw.SA_from_SP(S, P, mlon, mlat)
    CT = gsw.CT_from_t(SA, T, P)
    Z = gsw.z_from_p(P, mlat)

    # Extract T-S profiles and compute mixed layer depth
    CTpf, SApf, Zpf, ydprof, prof = sot.prof_from_env(P, T, S, yd, mode, mlon, mlat)

    # Fill the unsampled top and vertically interp to regular grid
    itop = Z >= -5 # np.logical_and(, np.isnan(prof)) # .reduce for more than 2 inputs
    binx = np.arange(np.floor(yd[0]*24), np.ceil(yd[-1]*24)+1)/24 # hourly
    bnd_top = stats.binned_statistic(yd[itop], [yd[itop], CT[itop], SA[itop], Z[itop]], 'mean', bins=binx)[0].T
    mZtop = np.nanmean(bnd_top[:,3])

    # only fill top for those contributed to the previous binning (top >= -5)
    ipf_notop = (Zpf[0,:] < mZtop) & (Zpf[0,:] >= -5)
    ig = ~np.isnan(bnd_top[:,0])
    nt = ydprof.size
    CTpf = np.r_[np.full((1,nt), nan), CTpf] # np.r_ adds one row on top
    SApf = np.r_[np.full((1,nt), nan), SApf]
    Zpf = np.r_[np.full((1,nt), nan), Zpf]
    CTpf[0,ipf_notop] = np.interp(ydprof[ipf_notop], bnd_top[ig,0], bnd_top[ig,1], left=nan, right=nan)
    SApf[0,ipf_notop] = np.interp(ydprof[ipf_notop], bnd_top[ig,0], bnd_top[ig,2], left=nan, right=nan)
    Zpf[0,ipf_notop] = mZtop
    
    # Plot ssTS_fill
    # plt.close()
    # _,axp = plt.subplots(3,1, figsize=(8,5), constrained_layout=True, sharex=True)
    # axp[0].plot(bnd_top[ig,0], bnd_top[ig,1], '-', ms=1)
    # axp[0].plot(ydprof[ipf_notop], CTpf[0,ipf_notop], 'o', ms=3, mfc='none')
    # axp[0].set_ylabel('Temp. [C]')
    # axp[0].legend(['hourly mean (z > -5 m)', 'linear interpolation to profiles'])
    # axp[1].plot(bnd_top[ig,0], bnd_top[ig,2], '-', ms=1)
    # axp[1].plot(ydprof[ipf_notop], SApf[0,ipf_notop], 'o', ms=3, mfc='none')
    # axp[1].set_ylabel('Salinity [g/kg]')
    # axp[2].fill_between(SFp1.yd, SFp1.wb_net.squeeze(), 0, where=SFp1.wb_net.squeeze()>0, color='C0', alpha=0.5)
    # axp[2].fill_between(SFp1.yd, SFp1.wb_net.squeeze(), 0, where=SFp1.wb_net.squeeze()<0, color='C1', alpha=0.5)
    # axp[2].set_ylim(-4.5e-7, 1.2e-7)
    # axp[2].set_ylabel('Buoy. flux [W/kg]')
    # ax2r = axp[2].twinx()
    # ax2r.plot(SFp1.yd, SFp1.taum.squeeze(), color='C4', lw=1)
    # ax2r.set_ylim(0, 1.2)
    # ax2r.set_ylabel('Stress [Pa]', color='C4')
    # ax2r.tick_params(axis='y', color='C4', labelcolor='C4')
    # ax2r.spines['right'].set_color('C4')
    # ax2r.grid(False)
    # axp[2].grid(False)
    # axp[2].set_xlabel('Yearday '+str(year));
    # plt.savefig(outfig_dir + 'ssTS_fill_'+str(year)+'.png');

    # Bin profiles in 3hour-2m bins
    binx = np.arange(np.floor(ydprof[0]*8), np.ceil(ydprof[-1]*8)+1)/8 # 3-hourly
    biny = np.arange(-depth_max-2, 2, 2)
    YDpf = np.tile(ydprof, (Zpf.shape[0],1))
    no_nan = (~np.isnan(CTpf)) & (~np.isnan(SApf))
    bnd2d = stats.binned_statistic_2d(YDpf[no_nan].ravel(), Zpf[no_nan].ravel(), 
                                     [YDpf[no_nan].ravel(), CTpf[no_nan].ravel(), SApf[no_nan].ravel(), Zpf[no_nan].ravel()], 
                                      statistic='mean', bins=(binx,biny))[0].T
    valid_prof = ~(np.isnan(bnd2d).sum(axis=0)>=(bnd2d.shape[0]-3)).any(axis=-1) # at least 3 data points in one profile
    bnd2d = bnd2d[:,valid_prof,:]

    # Vertical interpolation
    ydprof = np.nanmean(bnd2d[:,:,0], axis=0)
    nt = ydprof.size
    MZ = bnd2d[:,:,1:]
    igood = ~np.isnan(bnd2d[:,:,3])
    zprof = -np.linspace(1, depth_max+1, int(depth_max/2)+1)
    Mi = np.full((zprof.size, nt, 2), nan)
    for j,igd in enumerate( tqdm(igood.T, desc='vertically interpolate profile') ):
        df = pd.DataFrame(MZ[igd,j,:], columns=['CT', 'SA', 'Z']).set_index('Z')
        dfi = df.reindex(df.index.union(zprof)) \
                .interpolate(method='pchip', limit_area='inside', limit=8, limit_direction='both') \
                .interpolate(method='nearest', fill_value='extrapolate', limit=1, limit_area='outside',
                             limit_direction='forward') # extrap 1 grid, nearest
        Mi[:,j,:] = dfi.reindex(index=zprof).to_numpy()

    CTprof, SAprof = [np.squeeze(a) for a in np.split(Mi, 2, axis=-1)]
    PTprof = gsw.pt_from_CT(SAprof, CTprof)
    pprof = gsw.p_from_z(zprof, mlat)
    SPprof = gsw.SP_from_SA(SAprof, pprof[:,None], mlon, mlat)
    timeprof = sot.yd2pytime(ydprof, year)
    Sig0prof = gsw.sigma0(SAprof, CTprof)
    mldprof = sot.get_mld(Sig0prof, zprof, 0.005)

    # Interpolate T-S in 2D
    # nzpf = zprof.size
    # HRprof = np.tile(ydprof*24, (nzpf,1))
    # YDprof = HRprof/24
    hds = np.ceil(ydprof[0]*2) # intger half-days
    hde = np.floor(ydprof[-1]*2)
    hrg = np.linspace(hds, hde, int(hde-hds+1))*12 # intger hours
    ydg = hrg/24
    timeg = sot.yd2pytime(ydg, year)
    zg = np.linspace(-depth_max, 0, int(depth_max)+1)
    HRg, Zg = np.meshgrid(hrg, zg)

    hrzCTd = np.stack([yd*24, Z, CT], axis=0)
    hrzSAd = np.stack([yd*24, Z, SA], axis=0)
    # hrzCTd = np.stack([HRprof, Zprof, CTprof], axis=0).reshape(3,-1)
    # hrzSAd = np.stack([HRprof, Zprof, SAprof], axis=0).reshape(3,-1)

    print('2D interpolation of T-S...')
    z_cutoff = 30
    z_scale = 5
    zphr = 1/5
    CTg,CTgerr = sot.Gaussian_interp2d(hrzCTd, HRg, Zg, r=z_scale, cutoff=z_cutoff, yox=zphr) # cutoff: delta_z = 20 m, delta_t = 150 hr
    SAg,SAgerr = sot.Gaussian_interp2d(hrzSAd, HRg, Zg, r=z_scale, cutoff=z_cutoff, yox=zphr) # yox ~ 0.2 m/hr
    Sig0g = gsw.density.sigma0(SAg, CTg)

    # Bin and interpolate mixed layer depth
    mask = ~np.isnan(mldprof)
    yd_bin_edge = np.arange(np.floor(ydprof[0]*8), np.ceil(ydprof[-1]*8)+1)/8 # 3-hour bin
    bnd_mld = stats.binned_statistic(ydprof[mask], [ydprof[mask], mldprof[mask]], 
                                     statistic=sot.nanmean, bins=yd_bin_edge)[0].T
    bnd_mld_std = stats.binned_statistic(ydprof[mask], mldprof[mask],
                                         statistic='std', bins=yd_bin_edge)[0].T
    mask_bnd = ~np.isnan(bnd_mld[:,1])
    mldg = np.interp(ydg, bnd_mld[mask_bnd,0], bnd_mld[mask_bnd,1])

    # 'Zprof': (('zprof', 'timeprof'), Zprof, {'units': 'm', 'standard_name': 'profile_depth'}),
    gridded = xr.Dataset(
        data_vars={'CTprof': (('depth', 'time'), CTprof, {'units': 'C', 'standard_name': 'conservative_temperature_profiles'}),
                   'PTprof': (('depth', 'time'), PTprof, {'units': 'C', 'standard_name': 'potential_temperature_profiles'}),
                   'SAprof': (('depth', 'time'), SAprof, {'units': 'g/kg', 'standard_name': 'absolute_salinity_profiles'}),
                   'SPprof': (('depth', 'time'), SPprof, {'units': 'psu', 'standard_name': 'practical_salinity_profiles'}),
                   'Sig0prof': (('depth', 'time'), Sig0prof, {'units': 'kg/m3', 'standard_name': 'potential_density_profiles'}),
                   'mldprof': (('time'), mldprof, {'units': 'm', 'standard_name': 'mixed_layer_depth'}),
                   'CTg': (('zg', 'timeg'), CTg, {'units': 'C', 'standard_name': 'interpolated_conservative_temperature'}),
                   'SAg': (('zg', 'timeg'), SAg, {'units': 'g/kg', 'standard_name': 'interpolated_absolute_salinity'}),
                   'Sig0g': (('zg', 'timeg'), Sig0g, {'units': 'kg/m3', 'standard_name': 'interpolated_potential_density'}),
                   'CTgerr': (('zg', 'timeg'), CTgerr, {'units': 'C', 'standard_name': 'interpolated_conservative_temperature_error'}),
                   'SAgerr': (('zg', 'timeg'), SAgerr, {'units': 'g/kg', 'standard_name': 'interpolated_absolute_salinity_error'}),
                   'mldg': (('timeg'), mldg, {'units': 'm', 'standard_name': 'interpolated_mixed_layer_depth'})},
        coords={'yd': (('time'), ydprof, {'standard_name': 'profile_yearday_of_'+str(year)}),
                'time': (('time'), timeprof),
                'depth': (('depth'), -zprof, {'standard_name': 'profile_depth'}),
                'ydg': (('timeg'), ydg, {'standard_name': 'interpolation_grid_yearday_of_'+str(year)}),
                'timeg': (('timeg'), timeg),
                'zg': (('zg'), zg, {'standard_name': 'interpolation_grid_z'})},
        attrs={'description': 'Profile and gridded data from the Lagrangian float '+str(floatID)+' deployed under Typhoon Fanapi (2010)',
               'note': '2D gridded fields are from Gaussian interpolation; mixed layer depths are linearly interpolated.'})

    nc_name = 'Fanapi_gridded_'+str(floatID)+'.nc'
    gridded.to_netcdf(nbf_dir + nc_name, engine='netcdf4')
    print(f'Gridded profile data: {nc_name} saved at {nbf_dir}.\n')


if platform == 'linux':
    Groot = '/media/zhihua/internal1/GDrive/'
elif platform == 'darwin':
    Groot = '/Volumes/GoogleDrive/My Drive/'
_prep_fanapi_gridded(Groot, 60)
_prep_fanapi_gridded(Groot, 61)
_prep_fanapi_gridded(Groot, 62)
_prep_fanapi_gridded(Groot, 64)