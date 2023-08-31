#ifdef COSMOLOGY

#ifndef COSMOLOGY_H
#define COSMOLOGY_H

#include <stdio.h>
#include <cmath>
#include "../global/global.h"
#include "../particles/particles_3D.h"
#include "../gravity/grav3D.h"
#include "power_spectrum.h"
#include "../fft/fft_3D.h"

class Cosmology
{
public:

  Real H0;              //Hubble constant in km/s/kpc
  Real Omega_M;         //Matter density
  Real Omega_L;         //Dark energy density
  Real Omega_K;         //Curvature energy density
  Real Omega_b;         //Baryon density
  Real Omega_R;         //Radiation energy density

  Real cosmo_G;         //Gravitational constant in 
  Real cosmo_h;         //Hubble parameter H0 [in km/s/Mpc]/100[km/s/Mpc]
  Real current_z;       //Redshift
  Real current_a;       //Scale factor
  Real max_delta_a;     //Maximum allowable change in scale factor
  Real delta_a;         //Change in scale factor


  Real time_conversion; //kpc in km, for use with H0 as a time unit
  Real dt_secs;
  Real t_secs;

  Real r_kpch;          //1 kpc/h
  Real t_H0_inv;        //h/H0 in (kpc/h)/(km/s)
  Real v_0_cosmo;       //velocity scale (1 kpc/h)/(h/H0) in km/s
  Real phi_0_cosmo;     //potential scale in (km/s)^2
  Real rho_M_0;         //Matter density in h^2 Msun/kpc^3
  Real rho_b_0;         //Mean baryon density in h^2 Msun/kpc^3

  int n_outputs;
  int next_output_indx;
  real_vector_t scale_outputs;
  Real next_output;
  bool exit_now;
  
  bool generate_initial_conditions;
  struct Initial_Conditions{
  
    // For generating cosmological initial conditions
    
    int nx_local;
    int ny_local;
    int nz_local;
    
    int nx_total;
    int ny_total;
    int nz_total;
    
    Real *random_fluctuations;
    Real *rescaled_random_fluctuations_dm;
    Real *rescaled_random_fluctuations_gas;
    
    Cosmo_Power_Spectrum Power_Spectrum;
    
    FFT_3D FFT;
    
  } ICs;
   
  
  
  Cosmology( void );
  void Initialize( struct parameters *P, Grav3D &Grav, Particles_3D &Particles );

  //Set the cosmological parameters used by the code
  void SetCosmologicalParameters(struct parameters *P);

  //Set the unit system for cosmological calculations
  void SetUnitsCosmology(struct parameters *P, Grav3D &Grav);

  // print the cosmological unit system
  int chprintf_cosmology_units(void);

  void Load_Scale_Outputs( struct parameters *P );
  void Set_Scale_Outputs( struct parameters *P );

  void Set_Next_Scale_Output( );

  Real Get_Hubble_Parameter( Real a );
  Real Get_Hubble_Parameter_Full( Real a );

  Real Get_da_from_dt( Real dt );
  Real Get_dt_from_da( Real da );
  
  Real Time_Integrand( Real a );
  Real Get_Current_Time( Real a );
  
  Real Growth_Factor_Integrand( Real a );
  Real Get_Linear_Growth_Factor( Real a );
  Real Get_Linear_Growth_Factor_Deriv( Real a );

};

#endif
#endif
