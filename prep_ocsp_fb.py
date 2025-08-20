import xarray as xr
import numpy as np
import sca_osbl_tool as sot
import scipy.io as sio
import gsw
from sys import platform


def _prep_ocsp_fb(GDrive_root, year):
    site_root = GDrive_root + 'UW/Research/Data/' + 'OCSP/'
    nbf_dir = site_root + 'NBF/'

    # float parameters
    fm = sio.loadmat(nbf_dir+'Papa'+str(year)+'_FM.mat',squeeze_me=True,struct_as_record=False)
    with xr.open_dataset(nbf_dir+'OCSP_env_'+str(year)+'.nc') as E:
        mlon = np.mean(E.lon.values)
        mlat = np.mean(E.lat.values)
        S = E.S.values
        T = E.T.values
        P = E.P.values
        B = E.B.values
        yd = E.yd.values
        pytime = E.time.values

    SA = gsw.SA_from_SP(S, P, mlon, mlat)
    CT = gsw.CT_from_t(SA, T, P)
    rhow = gsw.rho(SA, CT, P)
    b = sot.get_float_buoyancy(fm['Mass'], fm['V0'], fm['T0'], fm['alpha'], fm['gamma'], fm['Air'],
                               P, T, rhow, B/1e6)
    fb = xr.Dataset(
        data_vars={'bq': (('timeP'), fm['V0q'], {'standard_name': 'float_buoyancy_quality'}),
                   'b': (('time'), b, {'units': 'g', 'standard_name': 'float_buoyancy'})},
        coords={'yd': (('time'), yd, {'standard_name': 'yearday_of_'+str(year)}),
                'time': (('time'), pytime)},
        attrs={'description': 'Seawater-relative buoyancy of the Lagrangian float deployed in '+str(year)+' near ocean climate station Papa',
               'note': 'bq: 1 is from logged float parameters, 0 is from diagnosed float parameters with assumptions'})

    nc_name = 'OCSP_fb_'+str(year)+'.nc'
    fb.to_netcdf(nbf_dir + nc_name, engine='netcdf4')
    print(f'Float buoyancy: {nc_name} saved at {nbf_dir}.\n')

if platform == 'linux':
    Groot = '/media/zhihua/internal1/GDrive/'
elif platform == 'darwin':
    Groot = '/Volumes/GoogleDrive/My Drive/'
_prep_ocsp_fb(Groot, 2011)
_prep_ocsp_fb(Groot, 2012)