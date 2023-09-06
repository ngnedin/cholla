#ifndef CHEMISTRY_GPU_H
#define CHEMISTRY_GPU_H

#include "../global/global.h"

#define CHEM_TINY 1e-20

// Define the type of a generic rate function.
typedef Real (*Rate_Function_T)(Real, Real);

class Grid3D;

#ifdef RT
  #include "../radiation/alt/photo_rates_csi.ANY.h"
  #include "../radiation/alt/photo_rates_csi_gpu.h"
#endif

// #define TEXTURES_UVB_INTERPOLATION

struct Chemistry_Header {

  Real gamma;
  Real runtime_chemistry_step;
  Real H_fraction;

  // Units system
  Real a_value;             //current scale factor
  Real density_units;       //density units to physical cgs
  Real energy_units;        //
  Real length_units;        //length scale to physical cm
  Real time_units;          //time scale in s
  Real energy_conversion;   //converts between conserved energy field and (cm/s)^2 a^2
  Real current_z;           //current redshift

  //Real dens_base;
  //Real length_base;
  //Real time_base;
  Real velocity_units;
  Real cooling_units;
  Real reaction_units;
  Real dens_number_conv;
  Real heat_units;
  // heat_units_old = eV_to_ergs / ChemHead.cooling_units;  /// NG 221127: this is incorrect
  //   heat_units = eV_to_ergs * 1e-10 * ChemHead.time_units * ChemHead.density_units / MH / MH;

  // Hubble parameter
  Real H0;

  // Interpolation tables for the rates
  int N_Temp_bins;
  Real Temp_start;
  Real Temp_end;

  Real *cool_ceHI_d;
  Real *cool_ceHeI_d;
  Real *cool_ceHeII_d;

  Real *cool_ciHI_d;
  Real *cool_ciHeI_d;
  Real *cool_ciHeII_d;
  Real *cool_ciHeIS_d;

  Real *cool_reHII_d;
  Real *cool_reHeII1_d;
  Real *cool_reHeII2_d;
  Real *cool_reHeIII_d;

  Real *cool_brem_d;

  Real cool_compton;

  Real *k_coll_i_HI_d;
  Real *k_coll_i_HeI_d;
  Real *k_coll_i_HeII_d;
  Real *k_coll_i_HI_HI_d;
  Real *k_coll_i_HI_HeI_d;

  Real *k_recomb_HII_d;
  Real *k_recomb_HeII_d;
  Real *k_recomb_HeIII_d;

  int max_iter;

  int n_uvb_rates_samples;
  float *uvb_rates_redshift_d;
  float *photo_ion_HI_rate_d;
  float *photo_ion_HeI_rate_d;
  float *photo_ion_HeII_rate_d;
  float *photo_heat_HI_rate_d;
  float *photo_heat_HeI_rate_d;
  float *photo_heat_HeII_rate_d;


  Real unitPhotoIonization; //converts per sec to code units
#ifdef RT
  const StaticTableGPU<float, 3, 'x'> *dTables[2];
  const PhotoRateTableStretchCSI *dStretch;
  Real unitPhotoHeating;
#endif
};

#ifdef CHEMISTRY_GPU

class Chem_GPU
{
 public:
  int nx;
  int ny;
  int nz;

  //
  //  0: case A
  //  1: case B
  //  2: special case for Iliev1 test
  //
  int recombination_case;

  Real scale_factor_UVB_on;

  float *cosmo_params_h;
  float *cosmo_params_d;

  int n_uvb_rates_samples;
  float *rates_z_h;
  float *Heat_rates_HI_h;
  float *Heat_rates_HeI_h;
  float *Heat_rates_HeII_h;
  float *Ion_rates_HI_h;
  float *Ion_rates_HeI_h;
  float *Ion_rates_HeII_h;

  float *rates_z_d;
  float *Heat_rates_HI_d;
  float *Heat_rates_HeI_d;
  float *Heat_rates_HeII_d;
  float *Ion_rates_HI_d;
  float *Ion_rates_HeI_d;
  float *Ion_rates_HeII_d;

  struct Chemistry_Header ChemHead;

  struct Fields {
    Real *temperature_h;
  } Fields;

  void Allocate_Array_GPU_Real(Real **array_dev, int size);
  void Copy_Real_Array_to_Device(int size, Real *array_h, Real *array_d);
  void Free_Array_GPU_Real(Real *array_dev);
  void Allocate_Array_GPU_float(float **array_dev, int size);
  void Copy_Float_Array_to_Device(int size, float *array_h, float *array_d);
  void Free_Array_GPU_float(float *array_dev);

  void Initialize(struct parameters *P);

  void Generate_Reaction_Rate_Table(Real **rate_table_array_d, Rate_Function_T rate_function, Real units);

  //creates the cooling rate tables
  //for collisional ionization,
  //collisional excitation, 
  //recombination, compton, and
  //bremsstrahlung
  void Initialize_Cooling_Rates();

  //creates the reaction rate tables
  //for collisions and recombinations
  void Initialize_Reaction_Rates();

  //loads the uvb rates from file and
  //copies them to the gpu
  void Initialize_UVB_Ionization_and_Heating_Rates(struct parameters *P);

  //loads the uvb rates from a file
  void Load_UVB_Ionization_and_Heating_Rates(struct parameters *P);

  //Copies chemistry uvb rate arrays to GPU
  void Copy_UVB_Rates_to_GPU();

  //Free chemistry rate arrays
  void Reset();

  //print the chemistry units to stdout
  int chprintf_chemistry_units();

  #ifdef TEXTURES_UVB_INTERPOLATION
  void Bind_GPU_Textures(int size, float *H_HI_h, float *H_HeI_h, float *H_HeII_h, float *I_HI_h, float *I_HeI_h,
                         float *I_HeII_h);
  #endif
};

/*! \fn void Cooling_Update(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt, Real gamma)
*  \brief When passed an array of conserved variables and a timestep, update the ionization fractions of H and He and
update the internal energy to account for radiative cooling and photoheating from the UV background. */
void Do_Chemistry_Update(Real *dev_conserved, const Real *dev_rf, int nx, int ny, int nz, int n_ghost, int n_fields,
                         Real dt, Chemistry_Header &Chem_H);

#endif
#endif
