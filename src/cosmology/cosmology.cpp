#ifdef COSMOLOGY

  #include "../cosmology/cosmology.h"

  #include "../io/io.h"

Cosmology::Cosmology(void) {}

void Cosmology::Initialize(struct parameters *P, Grav3D &Grav, Particles_3D &Particles)
{
  chprintf("Cosmological Simulation\n");

  H0      = P->H0;
  cosmo_h = H0 / 100;
  H0 /= 1000;  //[km/s / kpc]
  Omega_M = P->Omega_M;
  Omega_L = P->Omega_L;
  Omega_K = 1 - (Omega_M + Omega_L);
  Omega_b = P->Omega_b;
  Omega_R = 4.166e-5 / ( cosmo_h * cosmo_h ); // From Planck 2018

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

  // Set Scale factor in Gravity
  Grav.current_a = current_a;

  // Gravitational Constant in Cosmological Units
  cosmo_G = G_COSMO;

  // Set gravitational constant to use for potential calculation
  Grav.Gconst = cosmo_G;

  max_delta_a = 0.001;
  delta_a     = max_delta_a;

  // Initialize Time and set the time conversion
  t_secs          = 0;
  time_conversion = KPC_KM;

  // Set Normalization factors
  r_0_dm          = P->xlen / P->nx;
  //t_0_dm          = 1. / H0;
  t_0_dm          = 1. / H0 * cosmo_h;
  //v_0_dm          = r_0_dm / t_0_dm / cosmo_h;
  v_0_dm          = r_0_dm / t_0_dm; 
  rho_0_dm        = 3 * H0 * H0 / (8 * M_PI * cosmo_G) * Omega_M / cosmo_h / cosmo_h;
  rho_mean_baryon = 3 * H0 * H0 / (8 * M_PI * cosmo_G) * Omega_b / cosmo_h / cosmo_h;
  // dens_avrg = 0;

  r_0_gas   = 1.0; //kpc/h
  rho_0_gas = 3 * H0 * H0 / (8 * M_PI * cosmo_G) * Omega_M / cosmo_h / cosmo_h;
  t_0_gas   = 1 / H0 * cosmo_h;
  v_0_gas   = r_0_gas / t_0_gas;
  phi_0_gas = v_0_gas * v_0_gas;
  p_0_gas   = rho_0_gas * v_0_gas * v_0_gas;
  e_0_gas   = v_0_gas * v_0_gas;

  chprintf(" H0: %f\n", H0 * 1000);
  chprintf(" Omega_L: %f\n", Omega_L);
  chprintf(" Omega_M: %f\n", Omega_M);
  chprintf(" Omega_b: %f\n", Omega_b);
  chprintf(" Current_a: %f\n", current_a);
  chprintf(" Current_z: %f\n", current_z);
  chprintf(" rho_0_gas: %e\n", rho_0_gas);
  chprintf(" r_0_gas: %e \n", r_0_gas);
  chprintf(" t_0_gas: %e \n", t_0_gas);
  chprintf(" v_0_gas: %e \n", v_0_gas);
  chprintf(" phi_0_gas: %e \n", phi_0_gas);
  chprintf(" p_0_gas: %e \n", p_0_gas);
  chprintf(" e_0_gas: %e \n", e_0_gas);
  chprintf(" Max delta_a: %f \n", MAX_DELTA_A);

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
    ICs.random_fluctiations = NULL;
    ICs.rescaled_random_fluctiations_dm = NULL;
    ICs.rescaled_random_fluctiations_gas = NULL;

  }

}

#endif
