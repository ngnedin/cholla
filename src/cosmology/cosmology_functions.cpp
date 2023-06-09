#ifdef COSMOLOGY

  #include "../global/global.h"
  #include "../grid/grid3D.h"
  #include "../grid/grid_enum.h"
  #include "../io/io.h"


Real Cosmology::Get_Linear_Growth_Factor_Deriv( Real a )
{
  Real delta_a, D_l, D_r, t_l, t_r;
  delta_a = 1e-5;
  D_l = Get_Linear_Growth_Factor( a - delta_a );
  D_r = Get_Linear_Growth_Factor( a + delta_a );
  t_l = Get_Current_Time( a - delta_a );
  t_r = Get_Current_Time( a + delta_a );
  return ( D_r - D_l ) / ( t_r - t_l );
}

Real Cosmology::Time_Integrand( Real a )
{
  Real H = Get_Hubble_Parameter_Full( a );
  return 1 / ( H * a ) ;
}

Real Cosmology::Get_Current_Time( Real a_end )
{
  int n_iter = 1e8;
  Real a, a_start, delta_a, integral, H;
  a_start = 1e-50;
  delta_a = ( a_end - a_start ) / n_iter;

  // Integrate using Simpon's method
  integral =  Time_Integrand(a_start) + Time_Integrand(a_end);
  for ( int i=1; i<n_iter; i++ ){
    a = a_start + i * delta_a;
    if ( i%2 == 0 ) integral += 2 * Time_Integrand(a);
    else integral += 4 * Time_Integrand(a);
  }
  integral *= delta_a / 3;
  return integral * KPC;
}


Real Cosmology::Growth_Factor_Integrand( Real a )
{
  Real H = Get_Hubble_Parameter_Full( a );
  return 1 / pow( H * a, 3);
}

Real Cosmology::Get_Linear_Growth_Factor( Real a_end )
{
  int n_iter = 1e7;
  Real a, a_start, delta_a, integral, H;
  a_start = 1e-50;
  delta_a = ( a_end - a_start ) / n_iter;

  // Integrate using Simpon's method
  integral =  Growth_Factor_Integrand(a_start) + Growth_Factor_Integrand(a_end);
  for ( int i=1; i<n_iter; i++ ){
    a = a_start + i * delta_a;
    if ( i%2 == 0 ) integral += 2 * Growth_Factor_Integrand(a);
    else integral += 4 * Growth_Factor_Integrand(a);
  }
  integral *= delta_a / 3;
  H = Get_Hubble_Parameter_Full( a );
  return double(5.0)/2 * H0 * H0 * Omega_M * H * integral;
}
void Grid3D::Initialize_Cosmology(struct parameters *P)
{
  chprintf("Initializing Cosmology... \n");
  Cosmo.Initialize(P, Grav, Particles);

  // Change to comoving Cosmological System
  Change_Cosmological_Frame_System(true);

  if (fabs(Cosmo.current_a - Cosmo.next_output) < 1e-5) {
    H.Output_Now = true;
  }

  chprintf("Cosmology Successfully Initialized. \n\n");
}

Real Cosmology::Get_da_from_dt(Real dt)
{
  Real a2    = current_a * current_a;
  Real a_dot = sqrt(Omega_M / current_a + a2 * Omega_L + Omega_K) * H0;
  return a_dot * dt;
}

Real Cosmology::Get_dt_from_da(Real da)
{
  Real a2    = current_a * current_a;
  Real a_dot = sqrt(Omega_M / current_a + a2 * Omega_L + Omega_K) * H0;
  return da / a_dot;
}

Real Cosmology::Get_Hubble_Parameter(Real a)
{
  Real a2     = a * a;
  Real a3     = a2 * a;
  Real factor = (Omega_M / a3 + Omega_K / a2 + Omega_L);
  return H0 * sqrt(factor);
}

Real Cosmology::Get_Hubble_Parameter_Full( Real a )
{
  Real a2 = a * a;
  Real a3 = a2 * a;
  Real a4 = a3 * a;
  Real factor = ( Omega_R/a4 + Omega_M/a3 + Omega_K/a2 + Omega_L );
  return H0 * sqrt(factor);
}

void Grid3D::Change_Cosmological_Frame_System(bool forward)
{
  if (forward) {
    chprintf(" Converting to Cosmological Comoving System\n");
  } else {
    chprintf(" Converting to Cosmological Physical System\n");
  }

  Change_DM_Frame_System(forward);
  #ifndef ONLY_PARTICLES

  Change_GAS_Frame_System_GPU(forward);

  Change_GAS_Frame_System(forward);
  #endif  // ONLY_PARTICLES
}
void Grid3D::Change_DM_Frame_System(bool forward)
{
  #ifdef PARTICLES_CPU

  part_int_t pIndx;
  Real vel_factor;
  vel_factor = 1;

  for (pIndx = 0; pIndx < Particles.n_local; pIndx++) {
    Particles.vel_x[pIndx] *= vel_factor;
    Particles.vel_y[pIndx] *= vel_factor;
    Particles.vel_z[pIndx] *= vel_factor;
  }

  #endif  // PARTICLES_CPU

  // NOTE:Not implemented for PARTICLES_GPU, doesn't matter as long as
  // vel_factor=1
}

void Grid3D::Change_GAS_Frame_System(bool forward)
{
  Real dens_factor, momentum_factor, energy_factor;
  if (forward) {
    dens_factor     = 1 / Cosmo.rho_0_gas;
    momentum_factor = 1 / Cosmo.rho_0_gas / Cosmo.v_0_gas * Cosmo.current_a;
    energy_factor   = 1 / Cosmo.rho_0_gas / Cosmo.v_0_gas / Cosmo.v_0_gas * Cosmo.current_a * Cosmo.current_a;
  } else {
    dens_factor     = Cosmo.rho_0_gas;
    momentum_factor = Cosmo.rho_0_gas * Cosmo.v_0_gas / Cosmo.current_a;
    energy_factor   = Cosmo.rho_0_gas * Cosmo.v_0_gas * Cosmo.v_0_gas / Cosmo.current_a / Cosmo.current_a;
  }
  int k, j, i, id;
  for (k = 0; k < H.nz; k++) {
    for (j = 0; j < H.ny; j++) {
      for (i = 0; i < H.nx; i++) {
        id               = i + j * H.nx + k * H.nx * H.ny;
        C.density[id]    = C.density[id] * dens_factor;
        C.momentum_x[id] = C.momentum_x[id] * momentum_factor;
        C.momentum_y[id] = C.momentum_y[id] * momentum_factor;
        C.momentum_z[id] = C.momentum_z[id] * momentum_factor;
        C.Energy[id]     = C.Energy[id] * energy_factor;

  #ifdef DE
        C.GasEnergy[id] = C.GasEnergy[id] * energy_factor;
  #endif

  #ifdef COOLING_GRACKLE
        C.HI_density[id] *= dens_factor;
        C.HII_density[id] *= dens_factor;
        C.HeI_density[id] *= dens_factor;
        C.HeII_density[id] *= dens_factor;
        C.HeIII_density[id] *= dens_factor;
    #ifdef GRACKLE_METALS
        C.metal_density[id] *= dens_factor;
    #endif
  #endif  // COOLING_GRACKLE

  #ifdef CHEMISTRY_GPU
        C.HI_density[id] *= dens_factor;
        C.HII_density[id] *= dens_factor;
        C.HeI_density[id] *= dens_factor;
        C.HeII_density[id] *= dens_factor;
        C.HeIII_density[id] *= dens_factor;
  #endif
      }
    }
  }
}

#endif
