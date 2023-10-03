#ifdef COOLING_GRACKLE

  #ifndef INIT_GRACKLE_H
    #define INIT_GRACKLE_H

    #include "../global/global.h"

extern "C" {
    #include <grackle.h>
}

class Cool_GK
{
 public:
  code_units units;
  chemistry_data *data;

  Real energy_conversion; //cosmology units to code units
  Real density_to_CGS;    //code units to comoving CGS incl. h
  Real energy_to_CGS;     //code units to comoving CGS incl. h
  Real length_to_CGS;     //code units to comoving CGS incl. h
  Real velocity_to_CGS;   //code units to comoving CGS incl. h

  Real gamma;

  Real temperature_units;

    #ifdef OUTPUT_TEMPERATURE
  Real *temperature;
    #endif

  Real tiny_number;

  Real scale_factor_UVB_on;

  // Struct for storing grackle field data
  grackle_field_data fields;
  int field_size;

  Cool_GK(void);

  //Create the unit system conversion for Grackle cooling
  void SetUnitsCool(Cosmology &Cosmo);

  //Initialize the Grackle cooling
  void Initialize(struct parameters *P, Cosmology &Cosmo);

  void Free_Memory();
  // void Do_Cooling_Step( Real dt );

  Real Get_Mean_Molecular_Weight(int cell_id);
};

  #endif
#endif
