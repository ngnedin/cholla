#!/usr/bin/env python3
# Example file for concatenating 3D hdf5 datasets

import h5py
import numpy as np

ns = 0
ne = 0
n_proc = 16 # number of processors that did the calculations
istart = 0*n_proc
iend = 1*n_proc
dnamein = './hdf5/raw/'
dnameout = './hdf5/'

# loop over outputs
for n in range(ns, ne+1):

  # loop over files for a given output
  for i in range(istart, iend):

    # open the output file for writing (don't overwrite if exists)
    fileout = h5py.File(dnameout+str(n)+'.h5', 'a')
    # open the input file for reading
    filein = h5py.File(dnamein+str(n)+'.h5.'+str(i), 'r')
    # read in the header data from the input file
    head = filein.attrs

    # if it's the first input file, write the header attributes
    # and create the datasets in the output file
    if (i == 0):
      nx = head['dims'][0]
      ny = head['dims'][1]
      nz = head['dims'][2]
      fileout.attrs['dims'] = [nx, ny, nz]
      fileout.attrs['gamma'] = [head['gamma'][0]]
      fileout.attrs['t'] = [head['t'][0]]
      fileout.attrs['dt'] = [head['dt'][0]]
      fileout.attrs['n_step'] = [head['n_step'][0]]

      units = ['time_unit', 'mass_unit', 'length_unit', 'energy_unit', 'velocity_unit', 'density_unit']
      for unit in units:
        fileout.attrs[unit] = [head[unit][0]]

      d  = fileout.create_dataset("density", (nx, ny, nz), chunks=True, dtype=filein['density'].dtype)
      mx = fileout.create_dataset("momentum_x", (nx, ny, nz), chunks=True, dtype=filein['momentum_x'].dtype)
      my = fileout.create_dataset("momentum_y", (nx, ny, nz), chunks=True, dtype=filein['momentum_y'].dtype)
      mz = fileout.create_dataset("momentum_z", (nx, ny, nz), chunks=True, dtype=filein['momentum_z'].dtype)
      E  = fileout.create_dataset("Energy", (nx, ny, nz), chunks=True, dtype=filein['Energy'].dtype)
      try:
        GE = fileout.create_dataset("GasEnergy", (nx, ny, nz), chunks=True, dtype=filein['GasEnergy'].dtype)
      except KeyError:
        print('No Dual energy data present');
      try:
        [nx_mag, ny_mag, nz_mag] = head['magnetic_field_dims']
        bx = fileout.create_dataset("magnetic_x", (nx_mag, ny_mag, nz_mag), chunks=True, dtype=filein['magnetic_x'].dtype)
        by = fileout.create_dataset("magnetic_y", (nx_mag, ny_mag, nz_mag), chunks=True, dtype=filein['magnetic_y'].dtype)
        bz = fileout.create_dataset("magnetic_z", (nx_mag, ny_mag, nz_mag), chunks=True, dtype=filein['magnetic_z'].dtype)
      except KeyError:
        print('No magnetic field data present');
      try:
        HI_d = fileout.create_dataset("HI_density", (nx, ny, nz), chunks=True, dtype=filein['HI_density'].dtype)
        HII_d = fileout.create_dataset("HII_density", (nx, ny, nz), chunks=True, dtype=filein['HII_density'].dtype)
        HeI_d = fileout.create_dataset("HeI_density", (nx, ny, nz), chunks=True, dtype=filein['HeI_density'].dtype)
        HeII_d = fileout.create_dataset("HeII_density", (nx, ny, nz), chunks=True, dtype=filein['HeII_density'].dtype)
        HeIII_d = fileout.create_dataset("HeIII_density", (nx, ny, nz), chunks=True, dtype=filein['HeIII_density'].dtype)
      except KeyError:
        print('No abundance data present');
      try:
        rf1 = fileout.create_dataset("rf1", (nx, ny, nz), chunks=True, dtype=filein['rf1'].dtype)        

    # write data from individual processor file to
    # correct location in concatenated file
    nxl = head['dims_local'][0]
    nyl = head['dims_local'][1]
    nzl = head['dims_local'][2]
    xs = head['offset'][0]
    ys = head['offset'][1]
    zs = head['offset'][2]
    fileout['density'][xs:xs+nxl,ys:ys+nyl,zs:zs+nzl]  = filein['density']
    fileout['momentum_x'][xs:xs+nxl,ys:ys+nyl,zs:zs+nzl] = filein['momentum_x']
    fileout['momentum_y'][xs:xs+nxl,ys:ys+nyl,zs:zs+nzl] = filein['momentum_y']
    fileout['momentum_z'][xs:xs+nxl,ys:ys+nyl,zs:zs+nzl] = filein['momentum_z']
    fileout['Energy'][xs:xs+nxl,ys:ys+nyl,zs:zs+nzl]  = filein['Energy']
    try:
      fileout['GasEnergy'][xs:xs+nxl,ys:ys+nyl,zs:zs+nzl] = filein['GasEnergy']
    except KeyError:
        print('No Dual energy data present');
    try:
      [nxl_mag, nyl_mag, nzl_mag] = head['magnetic_field_dims_local']
      fileout['magnetic_x'][xs:xs+nxl_mag,ys:ys+nyl_mag,zs:zs+nzl_mag] = filein['magnetic_x']
      fileout['magnetic_y'][xs:xs+nxl_mag,ys:ys+nyl_mag,zs:zs+nzl_mag] = filein['magnetic_y']
      fileout['magnetic_z'][xs:xs+nxl_mag,ys:ys+nyl_mag,zs:zs+nzl_mag] = filein['magnetic_z']
    except KeyError:
        print('No magnetic field data present');
    try:
      fileout['HI_density'][xs:xs+nxl,ys:ys+nyl,zs:zs+nzl] = filein['HI_density']
      fileout['HII_density'][xs:xs+nxl,ys:ys+nyl,zs:zs+nzl] = filein['HII_density']
      fileout['HeI_density'][xs:xs+nxl,ys:ys+nyl,zs:zs+nzl] = filein['HeI_density']
      fileout['HeII_density'][xs:xs+nxl,ys:ys+nyl,zs:zs+nzl] = filein['HeII_density']
      fileout['HeIII_density'][xs:xs+nxl,ys:ys+nyl,zs:zs+nzl] = filein['HeIII_density']
    except KeyError:
        print("No abundance data present");
    try:
      fileout['rf1'][xs:xs+nxl,ys:ys+nyl,zs:zs+nzl] = filein['rf1']

    filein.close()

  fileout.close()
