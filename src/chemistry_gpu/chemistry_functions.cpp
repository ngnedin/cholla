#ifdef CHEMISTRY_GPU

  #include "../grid/grid3D.h"
  #include "../io/io.h"
  #include "chemistry_gpu.h"
  #include "rates.cuh"
  #include "utils/error_handling.h"

  #ifdef DE
    #include "../hydro/hydro_cuda.h"
  #endif

  #define TINY 1e-20


//Set the recombination case and the
//initial hydrogen fraction
void Grid3D::Initialize_Chemistry_Start(struct parameters *P)
{
  Chem.recombination_case = 0; // set case A recombination
  Chem.ChemHead.H_fraction = INITIAL_FRACTION_HI + INITIAL_FRACTION_HII; // initial H fraction = HI + HII
}


//Create the chemistry header,
//the chemistry unit system
//and load the cooling and
//reaction rate tables
void Grid3D::Initialize_Chemistry_Finish(struct parameters *P)
{
  chprintf("Initializing the GPU Chemistry Solver... \n");

  Chem.nx = H.nx;
  Chem.ny = H.ny;
  Chem.nz = H.nz;

  // Initialize the Chemistry Header
  Chem.ChemHead.runtime_chemistry_step = 0;
  Chem.ChemHead.gamma       = gama;
  Chem.ChemHead.N_Temp_bins = 600;
  Chem.ChemHead.Temp_start  = 1.0;
  Chem.ChemHead.Temp_end    = 1000000000.0;
  Chem.ChemHead.max_iter    = 10000;


  // Set up the units system for chemistry
  // This propagates all the units to 
  // ChemHead
  SetUnitsChemistry(P);

  // Initialize all the rate tables and 
  // load them from file
  Chem.Initialize(P);

  // Set photoionization rates
  Chem.ChemHead.n_uvb_rates_samples    = Chem.n_uvb_rates_samples;
  Chem.ChemHead.uvb_rates_redshift_d   = Chem.rates_z_d;
  Chem.ChemHead.photo_ion_HI_rate_d    = Chem.Ion_rates_HI_d;
  Chem.ChemHead.photo_ion_HeI_rate_d   = Chem.Ion_rates_HeI_d;
  Chem.ChemHead.photo_ion_HeII_rate_d  = Chem.Ion_rates_HeII_d;
  Chem.ChemHead.photo_heat_HI_rate_d   = Chem.Heat_rates_HI_d;
  Chem.ChemHead.photo_heat_HeI_rate_d  = Chem.Heat_rates_HeI_d;
  Chem.ChemHead.photo_heat_HeII_rate_d = Chem.Heat_rates_HeII_d;

  #ifdef RT
  Chem.ChemHead.dTables[0] = Rad.photoRates->bTables[0];
  Chem.ChemHead.dTables[1] = (Rad.photoRates->bTables.Count() > 1 ? Rad.photoRates->bTables[1] : nullptr);
  Chem.ChemHead.dStretch   = Rad.photoRates->bStretch.DevicePtr();

  Chem.ChemHead.unitPhotoHeating    = KB * 1e-10 * Chem.ChemHead.time_units * Chem.ChemHead.density_units / MH / MH;
  Chem.ChemHead.unitPhotoIonization = Chem.ChemHead.time_units;
  #endif  // RT

  chprintf("Allocating Memory in Initialize_Chemistry_Finish(). \n\n");
  int n_cells               = H.nx * H.ny * H.nz;
  Chem.Fields.temperature_h = (Real *)malloc(n_cells * sizeof(Real));

  chprintf("Chemistry Solver Successfully Initialized. \n\n");
}

//creates a temperature-dependent
//reaction or cooling rate look-up
//table and copies it to the GPU
void Chem_GPU::Generate_Reaction_Rate_Table(Real **rate_table_array_d, Rate_Function_T rate_function, Real units)
{
  // Host array for storing the rates
  Real *rate_table_array_h = (Real *)malloc(ChemHead.N_Temp_bins * sizeof(Real));

  // Get the temperature spacing.
  Real T, logT, logT_start, d_logT;
  logT_start = log(ChemHead.Temp_start);
  d_logT     = (log(ChemHead.Temp_end) - logT_start) / (ChemHead.N_Temp_bins - 1);

  // Evaluate the rate at each temperature.
  for (int i = 0; i < ChemHead.N_Temp_bins; i++) {
    rate_table_array_h[i] = CHEM_TINY;
    logT                  = logT_start + i * d_logT;
    T                     = exp(logT);
    rate_table_array_h[i] = rate_function(T, units);
  }

  // Allocate the device array for the rate and copy from host
  Allocate_Array_GPU_Real(rate_table_array_d, ChemHead.N_Temp_bins);
  Copy_Real_Array_to_Device(ChemHead.N_Temp_bins, rate_table_array_h, *rate_table_array_d);

  // Free the host array
  free(rate_table_array_h);
}





//Wrapper function to call the
//chemistry update step; sets
//the redshift and optionally
//passes the radiation fields
void Grid3D::Update_Chemistry()
{
  #ifdef COSMOLOGY
  Chem.ChemHead.current_z = Cosmo.current_z;
  #else
  Chem.ChemHead.current_z          = 0;
  #endif

  #ifdef RT
  Do_Chemistry_Update(C.device, Rad.rtFields.dev_rf, H.nx, H.ny, H.nz, H.n_ghost, H.n_fields, H.dt, Chem.ChemHead);
  #else
  Do_Chemistry_Update(C.device, nullptr, H.nx, H.ny, H.nz, H.n_ghost, H.n_fields, H.dt, Chem.ChemHead);
  #endif
}

//Compute the gas temperature field
//from the gas properties stored on
//the grid. Allow for cosmological
//unit conversion. BRANT: complete
void Grid3D::Compute_Gas_Temperature(Real *temperature, bool convert_cosmo_units)
{
  int k, j, i, id;
  Real dens_HI, dens_HII, dens_HeI, dens_HeII, dens_HeIII, dens_e, gamma;
  Real d, vx, vy, vz, E, GE, mu, temp, cell_dens, cell_n;
  Real current_a, a2;
  gamma = gama;

  for (k = 0; k < H.nz; k++) {
    for (j = 0; j < H.ny; j++) {
      for (i = 0; i < H.nx; i++) {
        id = i + j * H.nx + k * H.nx * H.ny;

        d  = C.density[id];
        vx = C.momentum_x[id] / d;
        vy = C.momentum_y[id] / d;
        vz = C.momentum_z[id] / d;
        E  = C.Energy[id];

  #ifdef DE
        GE = C.GasEnergy[id];
  #else
        GE = (E - 0.5 * d * (vx * vx + vy * vy + vz * vz));
  #endif

        dens_HI    = C.HI_density[id];
        dens_HII   = C.HII_density[id];
        dens_HeI   = C.HeI_density[id];
        dens_HeII  = C.HeII_density[id];
        dens_HeIII = C.HeIII_density[id];
        dens_e     = dens_HII + dens_HeII + 2 * dens_HeIII;

        cell_dens = dens_HI + dens_HII + dens_HeI + dens_HeII + dens_HeIII;
        cell_n    = dens_HI + dens_HII + (dens_HeI + dens_HeII + dens_HeIII) / 4 + dens_e;
        mu        = cell_dens / cell_n;

  #ifdef COSMOLOGY
        if (convert_cosmo_units) {
          current_a = Cosmo.current_a;
          a2        = current_a * current_a;
          //GE *= Chem.ChemHead.energy_conversion / a2;
          GE *= Chem.ChemHead.energy_conversion / a2;
        } else {
          GE *= KM_CGS*KM_CGS;  // convert from (km/s)^2 to (cm/s)^2
        }
  #endif

        temp = GE * MP * mu / d / KB * (gamma - 1.0);
        ;
        temperature[id] = temp;
        // chprintf( "mu: %e \n", mu );
        // if ( temp > 1e7 ) chprintf( "Temperature: %e   mu: %e \n", temp, mu );
      }
    }
  }
}

//initializes the cooling and
//reaction rate tables, initializes
//and loads the UVB ionization
//and heating rate tables
void Chem_GPU::Initialize(struct parameters *P)
{

  //create the cooling rate tables
  //and set their units
  Initialize_Cooling_Rates();

  //create the reaction rate tables
  //and set their units
  Initialize_Reaction_Rates();

  #ifndef RT //need to work out how to include this -- BRANT
  Initialize_UVB_Ionization_and_Heating_Rates(P);
  #endif  // RT
}

  //creates the cooling rate tables
  //for collisional ionization,
  //collisional excitation, 
  //recombination, compton, and
  //bremsstrahlung
void Chem_GPU::Initialize_Cooling_Rates()
{
  chprintf(" Initializing Cooling Rates... \n");

  //create the unit system for
  //cooling. Here, cooling units
  //are BRANT: complete
  Real units = ChemHead.cooling_units;


  //create collisional excitation rates
  Generate_Reaction_Rate_Table(&ChemHead.cool_ceHI_d, cool_ceHI_rate, units);
  Generate_Reaction_Rate_Table(&ChemHead.cool_ceHeI_d, cool_ceHeI_rate, units);
  Generate_Reaction_Rate_Table(&ChemHead.cool_ceHeII_d, cool_ceHeII_rate, units);

  //create collisional ionization rates
  Generate_Reaction_Rate_Table(&ChemHead.cool_ciHI_d, cool_ciHI_rate, units);
  Generate_Reaction_Rate_Table(&ChemHead.cool_ciHeI_d, cool_ciHeI_rate, units);
  Generate_Reaction_Rate_Table(&ChemHead.cool_ciHeII_d, cool_ciHeII_rate, units);
  Generate_Reaction_Rate_Table(&ChemHead.cool_ciHeIS_d, cool_ciHeIS_rate, units);

  //create recombination rates
  switch (recombination_case) {
    case 0: {
      Generate_Reaction_Rate_Table(&ChemHead.cool_reHII_d, cool_reHII_rate_case_A, units);
      Generate_Reaction_Rate_Table(&ChemHead.cool_reHeII1_d, cool_reHeII1_rate_case_A, units);
      Generate_Reaction_Rate_Table(&ChemHead.cool_reHeIII_d, cool_reHeIII_rate_case_A, units);
      break;
    }
    case 1:
    case 2: {
      Generate_Reaction_Rate_Table(&ChemHead.cool_reHII_d, cool_reHII_rate_case_B, units);
      Generate_Reaction_Rate_Table(&ChemHead.cool_reHeII1_d, cool_reHeII1_rate_case_B, units);
      Generate_Reaction_Rate_Table(&ChemHead.cool_reHeIII_d, cool_reHeIII_rate_case_B, units);
      break;
    }
  }
  Generate_Reaction_Rate_Table(&ChemHead.cool_reHeII2_d, cool_reHeII2_rate, units);

  //create bremsstrahlung rates
  Generate_Reaction_Rate_Table(&ChemHead.cool_brem_d, cool_brem_rate, units);

  //set compton cooling rates
  ChemHead.cool_compton = 5.65e-36 / units;
}

  //creates the reaction rate tables
  //for collisions and recombinations
void Chem_GPU::Initialize_Reaction_Rates()
{
  chprintf(" Initializing Reaction Rates... \n");

  //create the unit system for
  //reactions. Here, reaction_units
  //are BRANT: complete
  Real units = ChemHead.reaction_units;

  //create reaction rates for collisional
  //ionization
  Generate_Reaction_Rate_Table(&ChemHead.k_coll_i_HI_d, coll_i_HI_rate, units);
  Generate_Reaction_Rate_Table(&ChemHead.k_coll_i_HeI_d, coll_i_HeI_rate, units);
  Generate_Reaction_Rate_Table(&ChemHead.k_coll_i_HeII_d, coll_i_HeII_rate, units);
  Generate_Reaction_Rate_Table(&ChemHead.k_coll_i_HI_HI_d, coll_i_HI_HI_rate, units);
  Generate_Reaction_Rate_Table(&ChemHead.k_coll_i_HI_HeI_d, coll_i_HI_HeI_rate, units);

  //create the reaction rates for
  //recombinations
  switch (recombination_case) {
    case 0: {
      Generate_Reaction_Rate_Table(&ChemHead.k_recomb_HII_d, recomb_HII_rate_case_A, units);
      Generate_Reaction_Rate_Table(&ChemHead.k_recomb_HeII_d, recomb_HeII_rate_case_A, units);
      Generate_Reaction_Rate_Table(&ChemHead.k_recomb_HeIII_d, recomb_HeIII_rate_case_A, units);
      break;
    }
    case 1: {
      Generate_Reaction_Rate_Table(&ChemHead.k_recomb_HII_d, recomb_HII_rate_case_B, units);
      Generate_Reaction_Rate_Table(&ChemHead.k_recomb_HeII_d, recomb_HeII_rate_case_B, units);
      Generate_Reaction_Rate_Table(&ChemHead.k_recomb_HeIII_d, recomb_HeIII_rate_case_B, units);
      break;
    }
    case 2: {
      Generate_Reaction_Rate_Table(&ChemHead.k_recomb_HII_d, recomb_HII_rate_case_Iliev1, units);
      Generate_Reaction_Rate_Table(&ChemHead.k_recomb_HeII_d, recomb_HeII_rate_case_B, units);
      Generate_Reaction_Rate_Table(&ChemHead.k_recomb_HeIII_d, recomb_HeIII_rate_case_B, units);
      break;
    }
  }
}


//loads the uvb rates from file and
//copies them to the gpu
void Chem_GPU::Initialize_UVB_Ionization_and_Heating_Rates(struct parameters *P)
{
  chprintf(" Initializing UVB Rates... \n");

  //loads the uvb rates from a file
  Load_UVB_Ionization_and_Heating_Rates(P);

  //copy uvb rates to the gpu
  Copy_UVB_Rates_to_GPU();

  #ifdef TEXTURES_UVB_INTERPOLATION
  Bind_GPU_Textures(n_uvb_rates_samples, Heat_rates_HI_h, Heat_rates_HeI_h, Heat_rates_HeII_h, Ion_rates_HI_h,
                    Ion_rates_HeI_h, Ion_rates_HeII_h);
  #endif
}

//Copies chemistry uvb rate arrays to GPU
void Chem_GPU::Copy_UVB_Rates_to_GPU()
{
  Allocate_Array_GPU_float(&rates_z_d, n_uvb_rates_samples);
  Allocate_Array_GPU_float(&Heat_rates_HI_d, n_uvb_rates_samples);
  Allocate_Array_GPU_float(&Heat_rates_HeI_d, n_uvb_rates_samples);
  Allocate_Array_GPU_float(&Heat_rates_HeII_d, n_uvb_rates_samples);
  Allocate_Array_GPU_float(&Ion_rates_HI_d, n_uvb_rates_samples);
  Allocate_Array_GPU_float(&Ion_rates_HeI_d, n_uvb_rates_samples);
  Allocate_Array_GPU_float(&Ion_rates_HeII_d, n_uvb_rates_samples);

  Copy_Float_Array_to_Device(n_uvb_rates_samples, rates_z_h, rates_z_d);
  Copy_Float_Array_to_Device(n_uvb_rates_samples, Heat_rates_HI_h, Heat_rates_HI_d);
  Copy_Float_Array_to_Device(n_uvb_rates_samples, Heat_rates_HeI_h, Heat_rates_HeI_d);
  Copy_Float_Array_to_Device(n_uvb_rates_samples, Heat_rates_HeII_h, Heat_rates_HeII_d);
  Copy_Float_Array_to_Device(n_uvb_rates_samples, Ion_rates_HI_h, Ion_rates_HI_d);
  Copy_Float_Array_to_Device(n_uvb_rates_samples, Ion_rates_HeI_h, Ion_rates_HeI_d);
  Copy_Float_Array_to_Device(n_uvb_rates_samples, Ion_rates_HeII_h, Ion_rates_HeII_d);
}


// Free chemistry rate arrays 
void Chem_GPU::Reset()
{
  #ifndef RT //need to work out how to include this -- BRANT
  free(rates_z_h);
  free(Heat_rates_HI_h);
  free(Heat_rates_HeI_h);
  free(Heat_rates_HeII_h);
  free(Ion_rates_HI_h);
  free(Ion_rates_HeI_h);
  free(Ion_rates_HeII_h);

  Free_Array_GPU_float(rates_z_d);
  Free_Array_GPU_float(Heat_rates_HI_d);
  Free_Array_GPU_float(Heat_rates_HeI_d);
  Free_Array_GPU_float(Heat_rates_HeII_d);
  Free_Array_GPU_float(Ion_rates_HI_d);
  Free_Array_GPU_float(Ion_rates_HeI_d);
  Free_Array_GPU_float(Ion_rates_HeII_d);
  #endif
  free(Fields.temperature_h);
}


// Set the unit system for Chemistry
void Grid3D::SetUnitsChemistry(struct parameters *P)
{

  //Note we may need access to cosmology, so we use
  //a Grid3D method rather than Chem_GPU


  //Real Msun_cgs, kpc_cgs;//, kpc_km;//, dens_to_CGS;
  //Msun        = MSUN_CGS;
  //Msun_cgs    = MASS_UNIT;
  //kpc_cgs     = KPC_CGS;
  //kpc_cgs     = LENGTH_UNIT;
  //kpc_km     = KPC_KM;
  //dens_to_CGS = Msun_cgs / kpc_cgs / kpc_cgs / kpc_cgs;
  //dens_to_CGS = DENSITY_UNIT;
  
  //chprintf("dens_to_CGS %e DENSITY_UNIT %e\n",dens_to_CGS,DENSITY_UNIT);

//#ifdef COSMOLOGY
//dens_to_CGS = dens_to_CGS * Cosmo.rho_M_0 * Cosmo.cosmo_h * Cosmo.cosmo_h;

// bruno has
// dens_to_CGS = Cosmo.rho_M_0 * Msun / kpc_cgs / kpc_cgs / kpc_cgs * Cosmo.cosmo_h * Cosmo.cosmo_h;
//#endif  // COSMOLOGY

  //chprintf("dens_to_CGS %e DENSITY_UNIT %e\n",dens_to_CGS,DENSITY_UNIT);


  //density units are solar masses per kpc^3 in grams/cm^3
  Chem.ChemHead.density_units    = DENSITY_UNIT;

  //length units are kpc in cm
  Chem.ChemHead.length_units     = LENGTH_UNIT;

  //time units are kyr in s
  Chem.ChemHead.time_units       = TIME_UNIT;

  //dens_number_conv is number of hydrogen atoms per cm^3
  //for a density of 1 solar masses per kpc^3
  Chem.ChemHead.dens_number_conv = Chem.ChemHead.density_units / MH;
  //Chem.ChemHead.reaction_units   = MH / (Chem.ChemHead.density_units * Chem.ChemHead.time_units);
  

  //set cosmology chemistry unit system if needed
  #ifdef COSMOLOGY

  //scale factor
  Chem.ChemHead.a_value          = Cosmo.current_a;

  //note Chemistry Header H0 is in km/s/Mpc
  Chem.ChemHead.H0               = P->H0;

#define BRUNO_CHEM_UNITS
//
#ifdef BRUNO_CHEM_UNITS
  printf("BRUNO_CHEM_UNITS active\n");
#else
  printf("BRUNO_CHEM_UNITS not active\n");
#endif //BRUNO_CHEM_UNITS

#ifdef  BRUNO_CHEM_UNITS

  //rho_M_0*h^2/a^3 is the physical baryon density at scale factor a in Msun/kpc^3
  //density_units is then physical baryon density in g/cm^3

  Chem.ChemHead.density_units    *= Cosmo.rho_M_0*pow(Cosmo.cosmo_h,2)/pow(Chem.ChemHead.a_value,3); //physical


  //1 physical kpc in cm
  Chem.ChemHead.length_units     *= Chem.ChemHead.a_value/Cosmo.cosmo_h; //LENGTH_UNITS in physical cm

  //this time unit converts between Hubble parameter in
  //km/s/kpc to 1/s
  Chem.ChemHead.time_units       *= (Cosmo.time_conversion/Cosmo.cosmo_h)/TIME_UNIT; //convert Hubble from km/s/kpc to 1/s

  //converts 
  Chem.ChemHead.dens_number_conv *= Cosmo.rho_M_0*pow(Cosmo.cosmo_h,2); //comoving, incl h
  //Real dens_base, length_base, time_base;
  //dens_base   = Chem.ChemHead.density_units * pow(Chem.ChemHead.a_value,3); //comoving incl h?
  //length_base = Chem.ChemHead.length_units / Chem.ChemHead.a_value; //comoving incl h
  //time_base   = Chem.ChemHead.time_units; // kpc in km / h ...
#endif //BRUNO_CHEM_UNITS

  #endif  // COSMOLOGY

  //set the chemistry velocity unit
  Chem.ChemHead.velocity_units = Chem.ChemHead.length_units / Chem.ChemHead.time_units;

  //set the chemsitry energy unit
  Chem.ChemHead.energy_units     = Chem.ChemHead.density_units  * pow(Chem.ChemHead.velocity_units,2);

  //set the chemistry reaction unit
  //Chem.ChemHead.reaction_units = MH / (Chem.ChemHead.density_units * Chem.ChemHead.time_units);

  //without cosmology
  //reaction units are per number of hydrogen atoms in 1 msun per kpc^3
  //per 1 kyr expressed in cm^-3 s^-1
  Chem.ChemHead.reaction_units   = 1 / (Chem.ChemHead.dens_number_conv * Chem.ChemHead.time_units);


#ifdef  BRUNO_CHEM_UNITS
  Chem.ChemHead.reaction_units /= pow(Chem.ChemHead.a_value,3); //comoving incl h?
#endif //BRUNO_CHEM_UNITS

  //set the chemistry cooling units
  Chem.ChemHead.cooling_units  = 1.0e10 * MH * Chem.ChemHead.reaction_units;
//bruno has
/*
  Real Msun, kpc_cgs, kpc_km, dens_to_CGS;
  Msun = MSUN_CGS;
  kpc_cgs = KPC_CGS;
  kpc_km  = KPC_KM;
  dens_to_CGS = Cosmo.rho_M_0 * Msun / kpc_cgs / kpc_cgs / kpc_cgs * Cosmo.cosmo_h * Cosmo.cosmo_h;

  // These are conversions from code units to cgs. Following Grackle
  Chem.H.density_units  = dens_to_CGS / Chem.H.a_value / Chem.H.a_value / Chem.H.a_value ;
  Chem.H.length_units   = kpc_cgs / Cosmo.cosmo_h * Chem.H.a_value;
  Chem.H.time_units     = kpc_km / Cosmo.cosmo_h ;
  Chem.H.velocity_units = Chem.H.length_units /Chem.H.time_units;
  Chem.H.dens_number_conv = Chem.H.density_units * pow(Chem.H.a_value, 3) / MH;
  Real dens_base, length_base, time_base;
  dens_base   = Chem.H.density_units * Chem.H.a_value * Chem.H.a_value * Chem.H.a_value;
  length_base = Chem.H.length_units / Chem.H.a_value;
  time_base   = Chem.H.time_units;
  Chem.H.cooling_units   = ( pow(length_base, 2) * pow(MH, 2) ) / ( dens_base * pow(time_base, 3) );
  Chem.H.reaction_units = MH / (dens_base * time_base );
*/
/*

  #ifdef COSMOLOGY
  Real dens_base, length_base, time_base;
  dens_base   = Chem.ChemHead.density_units;
  length_base = Chem.ChemHead.length_units;
  dens_base   = dens_base * Chem.ChemHead.a_value * Chem.ChemHead.a_value * Chem.ChemHead.a_value;
  length_base = length_base / Chem.ChemHead.a_value;
  time_base = Chem.ChemHead.time_units;
  #endif  // COSMOLOGY
*/

  //BRANT
  //Chem.ChemHead.dens_base   = dens_base;
  //Chem.ChemHead.length_base = length_base;
  //Chem.ChemHead.time_base   = time_base;

  /// Chem.ChemHead.cooling_units   = ( pow(length_base, 2) * pow(MH, 2) ) / ( dens_base * pow(time_base, 3) ); NG 221127 -
  /// this is incorrect
  //Chem.ChemHead.cooling_units  = 1.0e10 * MH * MH / (dens_base * time_base);  // NG 221127 - fixed
  //Chem.ChemHead.reaction_units = MH / (dens_base * time_base);
  //Chem.ChemHead.reaction_units = MH / (Chem.ChemHead.density_units * Chem.ChemHead.time_units);
  //Chem.ChemHead.cooling_units  = 1.0e10 * MH * Chem.ChemHead.reaction_units;
  // printf(" cooling_units: %e\n", Chem.ChemHead.cooling_units );
  // printf(" reaction_units: %e\n", Chem.ChemHead.reaction_units );
  //
//Bruno has
//Chem.H.cooling_units   = ( pow(length_base, 2) * pow(MH, 2) ) / ( dens_base * pow(time_base, 3) );
//Chem.H.reaction_units = MH / (dens_base * time_base );

  //BRANT
  Chem.ChemHead.ion_units  = Chem.ChemHead.time_units;
  Chem.ChemHead.eV_to_ergs = EV_CGS;                                                 //eV in ergs
  Chem.ChemHead.heat_units = Chem.ChemHead.eV_to_ergs / Chem.ChemHead.cooling_units;

  //chprintf("ion_units %e\n",Chem.ChemHead.ion_units);
  //chexit(0);
  //Chem.ChemHead.heat_units = Chem.ChemHead.eV_to_ergs * 1.0e-10 * Chem.ChemHead.time_units * Chem.ChemHead.density_units / MH / MH;
  //    // heat_units_old = eV_to_ergs / ChemHead.cooling_units;  /// NG 221127: this is incorrect
  //      heat_units = eV_to_ergs * 1e-10 * ChemHead.time_units * ChemHead.density_units / MH / MH;
  //        //heat_units = EV_CGS * 1e-10 * ChemHead.time_units * ChemHead.density_units / MH / MH;
  //

// Bruno has
/*
 *   Real eV_to_ergs, heat_units, ion_units;
 *   eV_to_ergs = 1.60218e-12;
 *   heat_units = eV_to_ergs / H.cooling_units;
 *   ion_units  = H.time_units;
 *   in chemistry_io.cpp... 
 *       Ion_rates_HI_h[i]    = v[i][1] * ion_units;
 *           Heat_rates_HI_h[i]   = v[i][2] * heat_units;
 */


  Chem.ChemHead.density_conversion = Chem.ChemHead.density_units;
  #ifdef COSMOLOGY
  // Real kpc_cgs = KPC_CGS;
  //Chem.ChemHead.density_conversion = Cosmo.rho_M_0 * Cosmo.cosmo_h * Cosmo.cosmo_h / pow(kpc_cgs, 3) * MSUN_CGS;
  //Chem.ChemHead.density_conversion = Cosmo.rho_M_0 * Cosmo.cosmo_h * Cosmo.cosmo_h / pow(kpc_cgs, 3) * Msun_cgs;
  //Chem.ChemHead.density_conversion = Cosmo.rho_M_0 * Cosmo.cosmo_h * Cosmo.cosmo_h / pow(LENGTH_UNIT, 3) * Msun_cgs;
  //Chem.ChemHead.density_conversion = Cosmo.rho_M_0 * Cosmo.cosmo_h * Cosmo.cosmo_h / pow(LENGTH_UNIT, 3) * MASS_UNIT;
  
  // density conversion is rho_M_0 in comoving g/cm^3
  Chem.ChemHead.density_conversion *= pow(Chem.ChemHead.a_value,3);

  // energy conversion is (v_0_cosmo)^2 in cm/s
  Chem.ChemHead.energy_conversion   = Cosmo.v_0_cosmo * Cosmo.v_0_cosmo * KM_CGS * KM_CGS;  // km^2 -> cm^2 ;

  //Chem.ChemHead.energy_conversion   = Chem.ChemHead.energy_units / Chem.ChemHead.density_conversion; 

  //Chem.ChemHead.density_units    *= Cosmo.rho_M_0*pow(Cosmo.cosmo_h,2)/pow(Chem.ChemHead.a_value,3); //physical
  //Chem.ChemHead.length_units     *= Chem.ChemHead.a_value/Cosmo.cosmo_h; //LENGTH_UNITS in physical cm
  //Chem.ChemHead.time_units       *= (KPC_KM/Cosmo.cosmo_h)/TIME_UNIT; //WTH
  //Chem.ChemHead.dens_number_conv *= Cosmo.rho_M_0*pow(Cosmo.cosmo_h,2); //comoving, incl h

  #else                                                              // Not COSMOLOGY
  //Chem.ChemHead.density_conversion = DENSITY_UNIT;
  Chem.ChemHead.energy_conversion  = ENERGY_UNIT / DENSITY_UNIT;  // NG: this is energy per unit mass
  #endif


}


#endif //CHEMISTRY_GPU
