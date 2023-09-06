#ifdef CHEMISTRY_GPU

  #include <cstring>
  #include <fstream>
  #include <iostream>
  #include <sstream>
  #include <string>
  #include <vector>

  #include "../io/io.h"
  #include "../utils/error_handling.h"
  #include "chemistry_gpu.h"

using namespace std;

//Load the UVB ionization and heating rate tables
//from a file provided in the parameter file.
//This routine applies the unit conversion between
//the units the rates are stored in and the internal
//code units.
void Chem_GPU::Load_UVB_Ionization_and_Heating_Rates(struct parameters *P)
{
  char uvb_filename[100];
  // create the filename to read from
  strcpy(uvb_filename, P->UVB_rates_file);
  chprintf(" Loading UVB rates: %s\n", uvb_filename);

  std::fstream in(uvb_filename);
  std::string line;
  std::vector<std::vector<float>> v;
  int i = 0;
  if (in.is_open()) {
    while (std::getline(in, line)) {
      if (line.find("#") == 0) continue;

      float value;
      std::stringstream ss(line);
      // chprintf( "%s \n", line.c_str() );
      v.push_back(std::vector<float>());

      while (ss >> value) {
        v[i].push_back(value);
      }
      i += 1;
    }
    in.close();
  } else {
    chprintf(" Error: Unable to open UVB rates file: %s\n", uvb_filename);
    exit(1);
  }

  int n_lines = i;

  chprintf(" Loaded %d lines in file\n", n_lines);

  rates_z_h         = (float *)malloc(sizeof(float) * n_lines);
  Heat_rates_HI_h   = (float *)malloc(sizeof(float) * n_lines);
  Heat_rates_HeI_h  = (float *)malloc(sizeof(float) * n_lines);
  Heat_rates_HeII_h = (float *)malloc(sizeof(float) * n_lines);
  Ion_rates_HI_h    = (float *)malloc(sizeof(float) * n_lines);
  Ion_rates_HeI_h   = (float *)malloc(sizeof(float) * n_lines);
  Ion_rates_HeII_h  = (float *)malloc(sizeof(float) * n_lines);

  //Real heat_units;
  // heat_units_old = eV_to_ergs / ChemHead.cooling_units;  /// NG 221127: this is incorrect
  //heat_units = eV_to_ergs * 1e-10 * ChemHead.time_units * ChemHead.density_units / MH / MH;
  //heat_units = EV_CGS * 1e-10 * ChemHead.time_units * ChemHead.density_units / MH / MH;

  //chprintf("ChemHead.cooling_units %10.9e\n",ChemHead.cooling_units);
  //chprintf("heat_units %10.9e\n",heat_units);

  Real ion_units  = ChemHead.unitPhotoIonization;
  Real heat_units = ChemHead.heat_units;
  for (i = 0; i < n_lines; i++) {
    rates_z_h[i]         = v[i][0]; //redshift
    Ion_rates_HI_h[i]    = v[i][1] * ion_units;
    Heat_rates_HI_h[i]   = v[i][2] * heat_units;
    Ion_rates_HeI_h[i]   = v[i][3] * ion_units;
    Heat_rates_HeI_h[i]  = v[i][4] * heat_units;
    Ion_rates_HeII_h[i]  = v[i][5] * ion_units;
    Heat_rates_HeII_h[i] = v[i][6] * heat_units;
    // chprintf( " %f  %e  %e  %e   \n", rates_z_h[i], Heat_rates_HI_h[i],  Heat_rates_HeI_h[i],  Heat_rates_HeII_h[i]);
    // chprintf( " %f  %f  \n", rates_z_h[i], Heat_rates_HI_h[i] );
  }

  for (i = 0; i < n_lines - 1; i++) {
    if (rates_z_h[i] > rates_z_h[i + 1]) {
      chprintf(" ERROR: UVB rates must be ordered such that redshift is increasing as the rows increase in the file\n",
               uvb_filename);
      exit(2);
    }
  }

  n_uvb_rates_samples = n_lines;
  scale_factor_UVB_on = 1 / (rates_z_h[n_uvb_rates_samples - 1] + 1);
  chprintf(" Loaded UVB rates: \n");
  chprintf("  N redshift values: %d \n", n_uvb_rates_samples);
  chprintf("  z_min = %f    z_max = %f \n", rates_z_h[0], rates_z_h[n_uvb_rates_samples - 1]);
  chprintf("  UVB on:  a=%f \n", scale_factor_UVB_on);
}


//Print information on the chemistry unit system to
//stdout.
int Chem_GPU::chprintf_chemistry_units( void )
{
  int code;

  code = chprintf("**** Chemistry ****\n\n");
  code = chprintf("Chemistry Header time_units          %10.9e [same as TIME_UNIT].\n",ChemHead.time_units);
  code = chprintf("Chemistry Header length_units        %10.9e [same as LENGTH_UNIT].\n",ChemHead.length_units);
  code = chprintf("Chemistry Header velocity_units      %10.9e [length_units/time_units].\n",ChemHead.velocity_units);
  code = chprintf("Chemistry Header density_units       %10.9e [same as DENSITY_UNIT].\n",ChemHead.density_units);
  code = chprintf("Chemistry Header energy_units        %10.9e [same as ENERGY_UNIT].\n",ChemHead.energy_units);
  code = chprintf("Chemistry Header energy_conversion   %10.9e [v_0_cosmo * * 2 * 1e10].\n",ChemHead.energy_conversion);
  code = chprintf("Chemistry Header dens_number_conv    %10.9e [density_units/MH].\n",ChemHead.dens_number_conv);
  code = chprintf("Chemistry Header reaction_units      %10.9e [MH / (DENSITY_UNIT * TIME_UNIT)].\n",ChemHead.reaction_units);
  code = chprintf("Chemistry Header cooling_units       %10.9e [1e10 * MH * reaction_units].\n",ChemHead.cooling_units);
  code = chprintf("Chemistry Header heat_units          %10.9e [EV_CGS / cooling_units].\n",ChemHead.heat_units);
#ifdef RT
  code = chprintf("Chemistry Header unitPhotoHeating    %10.9e [kb * 1e-10 *time_units*density_units/MH/MH].\n",ChemHead.unitPhotoHeating);
  code = chprintf("Chemistry Header unitPhotoIonization %10.9e [same as TIME_UNIT].\n",ChemHead.unitPhotoIonization);
#endif //RT
#ifdef COSMOLOGY
  code = chprintf("Chemistry Header a_value             %10.9e.\n",ChemHead.a_value);
  code = chprintf("Chemistry Header H0                  %10.9e.\n",ChemHead.H0);
#endif //COSMOLOGY


/* RT photo ionization rates
    photo_i_HI   = pRates[0] * Chem_H.unitPhotoIonization;
    photo_h_HI   = pRates[1] * Chem_H.unitPhotoHeating;
    photo_i_HeI  = pRates[2] * Chem_H.unitPhotoIonization;
    photo_h_HeI  = pRates[3] * Chem_H.unitPhotoHeating;
    photo_i_HeII = pRates[4] * Chem_H.unitPhotoIonization;
    photo_h_HeII = pRates[5] * Chem_H.unitPhotoHeating;
*/

/*
  Non RT photoionization rates
    photo_i_HI   = linear_interpolation(delta_x, indx_l, indx_l + 1, Chem_H.photo_ion_HI_rate_d);
    photo_i_HeI  = linear_interpolation(delta_x, indx_l, indx_l + 1, Chem_H.photo_ion_HeI_rate_d);
    photo_i_HeII = linear_interpolation(delta_x, indx_l, indx_l + 1, Chem_H.photo_ion_HeII_rate_d);
    photo_h_HI   = linear_interpolation(delta_x, indx_l, indx_l + 1, Chem_H.photo_heat_HI_rate_d);
    photo_h_HeI  = linear_interpolation(delta_x, indx_l, indx_l + 1, Chem_H.photo_heat_HeI_rate_d);
    photo_h_HeII = linear_interpolation(delta_x, indx_l, indx_l + 1, Chem_H.photo_heat_HeII_rate_d);
*/

  code = chprintf("\n********\n");

  return code;
}

#endif
