#ifdef COSMOLOGY

#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include "power_spectrum.h"
#include "../io/io.h"


//NOTE: the input power spectrum should be normalized
//such that \sigma(R)**2 = \int Delta(k)**2 W(k*R)**2 dlnk
//where Delta(k)**2 = k**3 P(k)/(2\pi**2)
//For CDM cosmologies, this means at k = 1.0e-4 h/Mpc
//the value of P(k) will be around 400 or so.at z=0

//If one runs CAMB, the output matter power spectrum
//file will have the correct normalization by default.
//The input k should be in h/Mpc, and the input Pk
//will have units of (Mpc/h)**3.

//All further renormalization for the Fourier
//transforms should be handled elsewhere

// Constructor
Cosmo_Power_Spectrum::Cosmo_Power_Spectrum()
{
}

void Cosmo_Power_Spectrum:: Load_Power_Spectum_From_File( struct parameters *P )
{
  
  char pk_filename[MAXLEN];
  strcpy(pk_filename, P->cosmo_ics_pk_file);
  chprintf( " Loading Power Spectrum File: %s \n", pk_filename );
  
  std::fstream in_file(pk_filename);
  std::string line;
  std::vector<std::vector<float>> v;
  int i = 0;
  if (in_file.is_open()){
    while (std::getline(in_file, line))
    {
       if ( line.find("#") == 0 ) continue;
      
       float value;
       std::stringstream ss(line);
       if (line.length() == 0) continue;
       // chprintf( "%s \n", line.c_str() );
       v.push_back(std::vector<float>());
       
       while (ss >> value){
         // printf( " %d   %f\n", i, value );
         v[i].push_back(value);
       }
       i += 1;    
    }
    
    in_file.close();
  
  } else{
  
    chprintf(" Error: Unable to open the input power spectrum file: %s\n", pk_filename);
    exit(1);
  
  }
  
  int n_lines = i;
  chprintf( " Loaded %d lines in file. \n", n_lines  );
  
  host_size = n_lines;
  host_k      = (Real *)malloc( host_size*sizeof(Real) );
  host_pk_dm  = (Real *)malloc( host_size*sizeof(Real) );
  host_pk_gas = (Real *)malloc( host_size*sizeof(Real) );
  
  
  //Real dx = P->xlen / P->nx; 
  //Real pk_factor = 1 / ( dx * dx * dx );
  
  for (i=0; i<n_lines; i++ ){
    //host_k[i]      = v[i][0] * 1e-3; //Convert from 1/(Mpc/h) to  1/(kpc/h)
    //host_pk_dm[i]  = v[i][1] * pk_factor;  //IMPORTANT: The Power Spectrum has to be rescaled by the resolution volume!! Need to understand this! 
    //host_pk_gas[i] = v[i][2] * pk_factor; 

    host_k[i]      = v[i][0]; // h/Mpc
    host_pk_dm[i]  = v[i][1]; // (Mpc/h)**3
    host_pk_gas[i] = host_pk_dm[i]; // assume Pk_dm = Pk_gas for now
//    host_pk_gas[i] = v[i][2]; // (Mpc/h)**3
  }
  
  CudaSafeCall( cudaMalloc((void**)&dev_size,   sizeof(int)) );
  CudaSafeCall( cudaMalloc((void**)&dev_k,      host_size*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&dev_pk_dm,  host_size*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&dev_pk_gas, host_size*sizeof(Real)) );
  
  CudaSafeCall( cudaMemcpy(dev_size, &host_size, sizeof(int), cudaMemcpyHostToDevice) );
  CudaSafeCall( cudaMemcpy(dev_k, host_k, host_size*sizeof(Real), cudaMemcpyHostToDevice) );
  CudaSafeCall( cudaMemcpy(dev_pk_dm,  host_pk_dm,  host_size*sizeof(Real), cudaMemcpyHostToDevice) );
  CudaSafeCall( cudaMemcpy(dev_pk_gas, host_pk_gas, host_size*sizeof(Real), cudaMemcpyHostToDevice) );

}











#endif //COSMOLOGY
