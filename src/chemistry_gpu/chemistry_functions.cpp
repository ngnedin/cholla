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

        d  = C.density[id];           //density in units of Msun/kpc^3
        vx = C.momentum_x[id] / d;    //velocity in units of kpc/kyr
        vy = C.momentum_y[id] / d;    //velocity in units of kpc/kyr
        vz = C.momentum_z[id] / d;    //velocity in units of kpc/kyr
        E  = C.Energy[id];            //energy in (msun/kpc^3)(kpc/kyr)^2

  #ifdef DE
        GE = C.GasEnergy[id];
  #else
        GE = (E - 0.5 * d * (vx * vx + vy * vy + vz * vz));
  #endif

        dens_HI    = C.HI_density[id];                        //number density in
        dens_HII   = C.HII_density[id];                       //number density in 
        dens_HeI   = C.HeI_density[id];                       //number density in 
        dens_HeII  = C.HeII_density[id];                      //number density in
        dens_HeIII = C.HeIII_density[id];                     //number density in 
        dens_e     = dens_HII + dens_HeII + 2 * dens_HeIII;   //number density in 

        cell_dens = dens_HI + dens_HII + dens_HeI + dens_HeII + dens_HeIII;
        cell_n    = dens_HI + dens_HII + (dens_HeI + dens_HeII + dens_HeIII) / 4 + dens_e;
        mu        = cell_dens / cell_n; //mean molecular weight

  #ifdef COSMOLOGY
        if (convert_cosmo_units) {
          current_a = Cosmo.current_a;
          a2        = current_a * current_a;
          GE *= Chem.ChemHead.energy_conversion / a2;
        } else {
          GE *= KM_CGS*KM_CGS;  // convert from (km/s)^2 to (cm/s)^2
        }
  #endif

        temp = GE * MP * mu / d / KB * (gamma - 1.0);
        temperature[id] = temp;
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

#ifdef COSMOLOGY
  Initialize_UVB_Ionization_and_Heating_Rates(P);
#endif  // ENDIF
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
  Real cooling_units = ChemHead.cooling_units;


  //create collisional excitation rates
  Generate_Reaction_Rate_Table(&ChemHead.cool_ceHI_d, cool_ceHI_rate, cooling_units);
  Generate_Reaction_Rate_Table(&ChemHead.cool_ceHeI_d, cool_ceHeI_rate, cooling_units);
  Generate_Reaction_Rate_Table(&ChemHead.cool_ceHeII_d, cool_ceHeII_rate, cooling_units);

  //create collisional ionization rates
  Generate_Reaction_Rate_Table(&ChemHead.cool_ciHI_d, cool_ciHI_rate, cooling_units);
  Generate_Reaction_Rate_Table(&ChemHead.cool_ciHeI_d, cool_ciHeI_rate, cooling_units);
  Generate_Reaction_Rate_Table(&ChemHead.cool_ciHeII_d, cool_ciHeII_rate, cooling_units);
  Generate_Reaction_Rate_Table(&ChemHead.cool_ciHeIS_d, cool_ciHeIS_rate, cooling_units);

  //create recombination rates
  switch (recombination_case) {
    case 0: {
      Generate_Reaction_Rate_Table(&ChemHead.cool_reHII_d, cool_reHII_rate_case_A, cooling_units);
      Generate_Reaction_Rate_Table(&ChemHead.cool_reHeII1_d, cool_reHeII1_rate_case_A, cooling_units);
      Generate_Reaction_Rate_Table(&ChemHead.cool_reHeIII_d, cool_reHeIII_rate_case_A, cooling_units);
      break;
    }
    case 1:
    case 2: {
      Generate_Reaction_Rate_Table(&ChemHead.cool_reHII_d, cool_reHII_rate_case_B, cooling_units);
      Generate_Reaction_Rate_Table(&ChemHead.cool_reHeII1_d, cool_reHeII1_rate_case_B, cooling_units);
      Generate_Reaction_Rate_Table(&ChemHead.cool_reHeIII_d, cool_reHeIII_rate_case_B, cooling_units);
      break;
    }
  }
  Generate_Reaction_Rate_Table(&ChemHead.cool_reHeII2_d, cool_reHeII2_rate, cooling_units);

  //create bremsstrahlung rates
  Generate_Reaction_Rate_Table(&ChemHead.cool_brem_d, cool_brem_rate, cooling_units);

  //set compton cooling rates
  ChemHead.cool_compton = 5.65e-36 / cooling_units;
}

  //creates the reaction rate tables
  //for collisions and recombinations
void Chem_GPU::Initialize_Reaction_Rates()
{
  chprintf(" Initializing Reaction Rates... \n");

  //create the unit system for
  //reactions. Here, reaction_units
  //are BRANT: complete
  Real reaction_units = ChemHead.reaction_units;

  //create reaction rates for collisional
  //ionization
  Generate_Reaction_Rate_Table(&ChemHead.k_coll_i_HI_d, coll_i_HI_rate, reaction_units);
  Generate_Reaction_Rate_Table(&ChemHead.k_coll_i_HeI_d, coll_i_HeI_rate, reaction_units);
  Generate_Reaction_Rate_Table(&ChemHead.k_coll_i_HeII_d, coll_i_HeII_rate, reaction_units);
  Generate_Reaction_Rate_Table(&ChemHead.k_coll_i_HI_HI_d, coll_i_HI_HI_rate, reaction_units);
  Generate_Reaction_Rate_Table(&ChemHead.k_coll_i_HI_HeI_d, coll_i_HI_HeI_rate, reaction_units);

  //create the reaction rates for
  //recombinations
  switch (recombination_case) {
    case 0: {
      Generate_Reaction_Rate_Table(&ChemHead.k_recomb_HII_d, recomb_HII_rate_case_A, reaction_units);
      Generate_Reaction_Rate_Table(&ChemHead.k_recomb_HeII_d, recomb_HeII_rate_case_A, reaction_units);
      Generate_Reaction_Rate_Table(&ChemHead.k_recomb_HeIII_d, recomb_HeIII_rate_case_A, reaction_units);
      break;
    }
    case 1: {
      Generate_Reaction_Rate_Table(&ChemHead.k_recomb_HII_d, recomb_HII_rate_case_B, reaction_units);
      Generate_Reaction_Rate_Table(&ChemHead.k_recomb_HeII_d, recomb_HeII_rate_case_B, reaction_units);
      Generate_Reaction_Rate_Table(&ChemHead.k_recomb_HeIII_d, recomb_HeIII_rate_case_B, reaction_units);
      break;
    }
    case 2: {
      Generate_Reaction_Rate_Table(&ChemHead.k_recomb_HII_d, recomb_HII_rate_case_Iliev1, reaction_units);
      Generate_Reaction_Rate_Table(&ChemHead.k_recomb_HeII_d, recomb_HeII_rate_case_B, reaction_units);
      Generate_Reaction_Rate_Table(&ChemHead.k_recomb_HeIII_d, recomb_HeIII_rate_case_B, reaction_units);
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

#ifdef COSMOLOGY
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
#endif //COSMOLOGY
  free(Fields.temperature_h);
}


// Set the unit system for Chemistry
void Grid3D::SetUnitsChemistry(struct parameters *P)
{

  //Note we may need access to cosmology, so we use
  //a Grid3D method rather than Chem_GPU

  //density units are solar masses per kpc^3 in grams/cm^3
  Chem.ChemHead.density_units    = DENSITY_UNIT;

  //length units are kpc in cm
  Chem.ChemHead.length_units     = LENGTH_UNIT;

  //time units are kyr in s
  Chem.ChemHead.time_units       = TIME_UNIT;

  //dens_number_conv is number of hydrogen atoms per cm^3
  //for a density of 1 solar masses per kpc^3
  //multiply to convert to cgs
  //divide to convert from cgs to code units
  Chem.ChemHead.dens_number_conv = Chem.ChemHead.density_units / MH;
  

//set cosmology chemistry unit system if needed
#ifdef COSMOLOGY

  //scale factor
  Chem.ChemHead.a_value          = Cosmo.current_a;

  //note Chemistry Header H0 is in km/s/Mpc
  Chem.ChemHead.H0               = P->H0;

  //rho_M_0*h^2/a^3 is the proper matter density at scale factor a in Msun/kpc^3
  //density_units is then proper matter density in g/cm^3
  Chem.ChemHead.density_units    *= Cosmo.rho_M_0*pow(Cosmo.cosmo_h,2)/pow(Chem.ChemHead.a_value,3); //physical


  //converts 1 comoving kpc to proper cm
  Chem.ChemHead.length_units     *= Chem.ChemHead.a_value/Cosmo.cosmo_h; //1 comoving kpc in proper cm

  //this time unit converts between Hubble parameter in
  //km/s/kpc to 1/s
  Chem.ChemHead.time_units       *= (Cosmo.time_conversion/Cosmo.cosmo_h)/TIME_UNIT; //convert Hubble from km/s/kpc to 1/s

  //converts from cosmological code units to cgs number density
  Chem.ChemHead.dens_number_conv *= Cosmo.rho_M_0*pow(Cosmo.cosmo_h,2); //comoving but incl h^2
#endif  // COSMOLOGY

  //set the chemistry velocity unit
  Chem.ChemHead.velocity_units = Chem.ChemHead.length_units / Chem.ChemHead.time_units;

  //set the chemistry energy unit
  Chem.ChemHead.energy_units     = Chem.ChemHead.density_units  * pow(Chem.ChemHead.velocity_units,2);

  //without cosmology
  //reaction units are per number of hydrogen atoms in 1 msun per kpc^3
  //per 1 kyr expressed in cm^-3 s^-1
  //Converts code units to per cm^3 per s
  Chem.ChemHead.reaction_units   = 1 / (Chem.ChemHead.dens_number_conv * pow(Chem.ChemHead.a_value,3) * Chem.ChemHead.time_units);

  //set the chemistry cooling units, ergs per cm^3 per s
  Chem.ChemHead.cooling_units  = (KM_CGS*KM_CGS) * MH * Chem.ChemHead.reaction_units;

  //define the conversion between the conserved energy field units
  //and (cm/s)^2. In cosmology, the conserved energy field contains
  //a factor of a^2 that is removed inside the chemistry routine
#ifdef COSMOLOGY
  // energy conversion is (v_0_cosmo)^2 in (cm/s)^2
  Chem.ChemHead.energy_conversion   = Cosmo.v_0_cosmo * Cosmo.v_0_cosmo * KM_CGS * KM_CGS;  // km^2 -> cm^2 ;
#else  // Not COSMOLOGY
  // energy conversion is just the ratio of energy and density units
  Chem.ChemHead.energy_conversion  = ENERGY_UNIT / DENSITY_UNIT;  // NG: this is energy per unit mass
#endif // COSMOLOGY

  //define the conversion between photoheating and photoionization rates
  //and the internal code units

  //Photo heating units --    //cm^3 s / K
  Chem.ChemHead.unitPhotoHeating = KB / Chem.ChemHead.cooling_units;

  //Photo ionization units -- convert per second to per kyr in sec
  Chem.ChemHead.unitPhotoIonization = Chem.ChemHead.time_units;
}


#endif //CHEMISTRY_GPU
