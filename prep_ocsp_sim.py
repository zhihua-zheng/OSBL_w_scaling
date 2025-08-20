import numpy as np
import xarray as xr
from sys import platform

def select_vars(ds, vars=['zsbl','Bfsfc']):
    return ds[vars].squeeze(['lon', 'lat'], drop=True)

if platform == 'linux':
    GDrive_root = '/media/zhihua/internal1/GDrive/'
elif platform == 'darwin':
    GDrive_root = '/Volumes/GoogleDrive/My Drive/'
site_root = GDrive_root + 'UW/Research/Data/' + 'OCSP/'
sim_dir = site_root + 'NBF/Sim/2011/'

with xr.open_dataset(nbf_dir+'OCSP_dft_selected_2011.nc') as ds:
    drift = ds.drift.values
    timedrift = ds.time.values
Ndrift = int(np.nanmax(drift)) + 1 # starts from 0

# filepaths = [sim_dir+'gotm_out_Drift{:02d}'.format(i)+'.nc' for i in range(Ndrift)]
filepaths = sim_dir+'gotm_out_Drift*.nc'
ds = xr.open_mfdataset(filepaths, combine='by_coords', preprocess=select_vars,
                       data_vars='minimal', coords='minimal', compat='override')