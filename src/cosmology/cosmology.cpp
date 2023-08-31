#ifdef COSMOLOGY

  #include "../cosmology/cosmology.h"

  #include "../io/io.h"

//Constructor
Cosmology::Cosmology(void) {}

//Set the cosmological parameters used by the code
void Cosmology::SetCosmologicalParameters(struct parameters *P)
{
  // Initialize the cosmological parameters
  H0      = P->H0;                                //Read H0 in km/s/Mpc from parameter file
  cosmo_h = H0 / 100;                             //Set cosmological "little h"
  H0 /= 1000;                                     //[km/s / kpc]
  Omega_M = P->Omega_M;                           //Set the matter density
  Omega_L = P->Omega_L;                           //Set the dark energy density
  Omega_K = 1 - (Omega_M + Omega_L);              //Set the curvature energy density
  Omega_b = P->Omega_b;                           //Set the baryon density
  Omega_R = 4.166e-5 / ( cosmo_h * cosmo_h );     //Set the radiation density from Planck 2018
}


//Set the unit system for cosmological calculations
void Cosmology::SetUnitsCosmology(struct parameters *P, Grav3D &Grav)
{
  // Gravitational Constant in Cosmological Units
  cosmo_G = G_COSMO; //kpc km^2/s^2 /Msun

  // Set Normalization factors

  //mean baryon density in h^2 Msun/kpc
  rho_b_0 = 3 * H0 * H0 / (8 * M_PI * cosmo_G) * Omega_b / cosmo_h / cosmo_h;

  //length scale in kpc/h
  r_kpch   = 1.0; //kpc/h

  //density scale in h^2 Msun/kpc^3
  rho_M_0 = 3 * H0 * H0 / (8 * M_PI * cosmo_G) * Omega_M / cosmo_h / cosmo_h;

  //time scale in (kpc/h)/(km/2)
  t_H0_inv   = cosmo_h / H0;

  //velocity scale in km/s
  v_0_cosmo   = r_kpch / t_H0_inv;  //km/s

  //potential scale in km^2/s^2
  phi_0_cosmo = v_0_cosmo * v_0_cosmo;  //(km/s)^2

  // Set Scale factor in Gravity
  Grav.current_a = current_a;

  // Set gravitational constant to use for potential calculation
  Grav.Gconst = cosmo_G;

  // Set the maximum change in scale factor
  max_delta_a = MAX_DELTA_A;

  //set the time conversion
  //kpc in km
  time_conversion = KPC_KM;

}

// print the cosmological unit system
int Cosmology::chprintf_cosmology_units(void)
{
  int code;

  code = chprintf("**** Cosmological Parameters ****\n\n");
  code = chprintf("H0               %10.9e [Hubble parameter in km/s/Mpc]\n",H0*1000);
  code = chprintf("Omega_L          %10.9e [Dark energy density]\n",Omega_L);
  code = chprintf("Omega_M          %10.9e [Matter density]\n",Omega_M);
  code = chprintf("Omega_b          %10.9e [Baryon matter density]\n",Omega_b);
  code = chprintf("Omega_K          %10.9e [Curvature energy density]\n",Omega_K);
  code = chprintf("Omega_R          %10.9e [Radiation energy density]\n",Omega_R);
  code = chprintf("Current_a:       %10.9e [Current scale factor]\n", current_a);
  code = chprintf("Current_z:       %10.9e [Current redshift]\n", current_z);

  code = chprintf("**** Cosmological Unit System ****\n\n");
  code = chprintf("cosmo_G:         %10.9e [same as G_COSMO, in kpc km^2/s^2 /Msun]\n", cosmo_G);
  code = chprintf("rho_b_0:         %10.9e [3 Omega_b H0^2/(8piG)/h^2 in h^2 Msun/kpc^3]\n", rho_b_0);
  code = chprintf("rho_M_0:         %10.9e [3 Omega_m H0^2/(8piG)/h^2 in h^2 Msun/kpc^3]\n", rho_M_0);
  code = chprintf("r_kpch:          %10.9e [1 kpc/h]\n", r_kpch);
  code = chprintf("t_H0_inv:        %10.9e [h/H0 in (kpc/h)/(km/s)]\n", t_H0_inv);
  code = chprintf("v_0_cosmo:       %10.9e [r_kpch/t_H0_inv in km/s]\n", v_0_cosmo);
  code = chprintf("phi_0_cosmo:     %10.9e [v_0_cosmo^2 in km^2/s^2]\n", phi_0_cosmo);
  code = chprintf("Max delta_a:     %10.9e [Maximum change in scale factor]\n", max_delta_a);
  code = chprintf("time_conversion: %10.9e [kpc in km, for use with H0]", time_conversion);
  code = chprintf("\n********\n");


  return code;
}


// initialize cosmological simulations
void Cosmology::Initialize(struct parameters *P, Grav3D &Grav, Particles_3D &Particles)
{
  chprintf("Cosmological Simulation\n");

  // Set the cosmological parameters
  SetCosmologicalParameters(P);

  // Set the scale factor and redshift
  if (strcmp(P->init, "Read_Grid") == 0) {
    // Read scale factor value from Particles
    current_z = Particles.current_z;
    current_a = Particles.current_a;
  } else {
    current_z           = P->Init_redshift;
    current_a           = 1. / (current_z + 1);
    Particles.current_z = current_z;
    Particles.current_a = current_a;
  }

  // Set the cosmological unit system
  SetUnitsCosmology(P, Grav);

  // Initialize Time
  t_secs          = 0;

  // Initialize change in scale factor
  delta_a     = max_delta_a;

  //Print the cosmology units to screen
  chprintf_cosmology_units();

  //Set the scale factor when
  //outputs are generated
  Set_Scale_Outputs(P);

  //Generate cosmological initial conditions?

  if ( strcmp(P->init, "Generate_Cosmological_ICs")==0) {
    generate_initial_conditions = true;
  } else { 
    generate_initial_conditions = false;
  }


  if ( generate_initial_conditions ){

    // Initialize parameters for initial conditions generator
    ICs.nx_local = Grav.nx_local;
    ICs.ny_local = Grav.ny_local;
    ICs.nz_local = Grav.nz_local;
    ICs.nx_total = Grav.nx_total;
    ICs.ny_total = Grav.ny_total;
    ICs.nz_total = Grav.nz_total;

  } else {

    ICs.nx_local = 0;
    ICs.nx_local = 0;
    ICs.nx_local = 0;
    ICs.nx_total = 0;
    ICs.nx_total = 0;
    ICs.nx_total = 0;
    ICs.random_fluctuations = NULL;
    ICs.rescaled_random_fluctuations_dm = NULL;
    ICs.rescaled_random_fluctuations_gas = NULL;

  }

}

#endif
