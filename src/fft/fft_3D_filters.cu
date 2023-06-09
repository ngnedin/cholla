#ifdef PARIS


#include "fft_3D.h"
#include "../utils/gpu.hpp"
#include "../io/io.h"
#include <cassert>
#include <cfloat>
#include <climits>


__host__ __device__ static inline Real sqr(const Real x)
{
  return x*x; 
}

__device__ Real linear_interpolation( Real x, Real *x_vals, Real *y_vals, int N )
{
  if ( x <= x_vals[0] ){
    printf(" x: %f  outside of interpolation range.\n", x );
    return y_vals[0];
  }
  if ( x >= x_vals[N-1] ){
    printf(" x: %f  outside of interpolation range.\n", x );
    return y_vals[N-1];
  }
  int indx = 0;
  while( x_vals[indx] < x ) indx +=1;
  // printf( "%d \n", indx );
  Real xl, xr, yl, yr;
  xl = x_vals[indx-1];
  xr = x_vals[indx];
  yl = y_vals[indx-1];
  yr = y_vals[indx];  
  if ( x < xl || x > xr ) printf(" ##################### Interpolation error:   x: %e  xl: %e  xr: %e   indx: %d\n", x, xl, xr, indx );
  return  yl + ( x - xl ) / ( xr - xl ) * ( yr - yl );
}




void FFT_3D::Filter_rescale_by_k_k2( Real *input, Real *output, bool in_device, int direction, Real D ) const
{
  // Local copies of members for lambda capture
  const int ni = ni_, nj = nj_, nk = nk_; //global grid sizes
  const Real ddi = ddi_, ddj = ddj_, ddk = ddk_;
  const size_t bytes = minBytes_;

  if ( in_device ){
    CHECK( cudaMemcpy( db_, input, inputBytes_, cudaMemcpyDeviceToDevice));
  } else {
    CHECK( cudaMemcpy( db_, input, inputBytes_, cudaMemcpyHostToDevice));
  } 

  // Provide FFT filter with a lambda that multiplies by k / k^2 / D
  henry_->filter(bytes, db_, da_,
    [=] __device__ (const int i, const int j, const int k, const cufftDoubleComplex b) {
      if (i || j || k) {
        // Get the global indices 
        int id_i = i < ni/2 ? i : i - ni;
        int id_j = j < nj/2 ? j : j - nj;
        int id_k = k < nk/2 ? k : k - nk;
        // Compute kx, ky, and kz from the indices
        double kz = id_i * ddi;	//note this has units of h/kpc here
        double ky = id_j * ddj;	
        double kx = id_k * ddk;  
        // Compute the magnitude of k squared
        //double k2 = kx*kx + ky*ky + kz*kz ; //note this has units of (h/kpc)**2 here
        double k2 = kx*kx + ky*ky + kz*kz ; //note this has units of (h/kpc)**2 here
	//k2*=1.0e6;	//(h/Mpc)**2
        if ( k2 == 0 ) k2 = 1.0;	//DOESN't THIS HAVE UNITS?
        Real factor = 0;
        if      (direction == 0) factor = kz / k2 / D;
        else if (direction == 1) factor = ky / k2 / D;
        else if (direction == 2) factor = kx / k2 / D;
        else printf("Wrong direction %d\n", direction ); 
         // multiply b by 1j*factor ( Imaginary Number)
        return cufftDoubleComplex{-factor*b.y,factor*b.x};
      } else {
        return cufftDoubleComplex{0.0,0.0};
      }
    });
    
    if ( in_device ){
      CHECK( cudaMemcpy( output, da_, outputBytes_, cudaMemcpyDeviceToDevice));
    } else {
      CHECK( cudaMemcpy( output, da_, outputBytes_, cudaMemcpyDeviceToHost));
    } 
}

void FFT_3D::Filter_rescale_by_power_spectrum( Real *input, Real *output, bool in_device, int size, Real *dev_k, Real *dev_pk, Real dx3 ) const
{
  // Local copies of members for lambda capture
  const int ni = ni_, nj = nj_, nk = nk_;
  const Real ddi = ddi_, ddj = ddj_, ddk = ddk_;
  const size_t bytes = minBytes_;
  
  if ( in_device ){
    CHECK( cudaMemcpy( db_, input, inputBytes_, cudaMemcpyDeviceToDevice));
  } else {
    CHECK( cudaMemcpy( db_, input, inputBytes_, cudaMemcpyHostToDevice));
  } 
  
  // Provide FFT filter with a lambda that multiplies by P(k)
  henry_->filter(bytes, db_, da_,
    [=] __device__ (const int i, const int j, const int k, const cufftDoubleComplex b) {
      if (i || j || k) {
        // Get the global indices 
        int id_i = i < ni/2 ? i : i - ni;
        int id_j = j < nj/2 ? j : j - nj;
        int id_k = k < nk/2 ? k : k - nk;
        // Compute kx, ky, and kz from the indices
        double kz = id_i * ddi; //fft_3D.cu:  ddi_ = 2.0*M_PI*double(n[0]-1)/(double(n[0])*(hi[0]-lo_[0]));
        double ky = id_j * ddj;	//this has units of h/kpc
        double kx = id_k * ddk;  
        // Compute the magnitude of k 
        const double k_mag_h_mpc = sqrt( kx*kx + ky*ky + kz*kz ) * 1.0e3; // h/Mpc


        //power spectrum in (Mpc/h)**3
	//correct power
        double pk = linear_interpolation( k_mag_h_mpc, dev_k, dev_pk, size );

        double Ampk; //rescaling amplitudes

        double wLow = (1./ni)*(1./nj)*(1./nk); //1/N**3

        if ( i==1 && j==1 && k==1 ) printf("###### kx: %e  ky: %e  kz: %e  k_mag: %e  pk: %e \n", kx, ky, kz, k_mag_h_mpc, pk );  
        if ( i==1 && j==1 && k==1 ) printf("###### ni: %d  nj: %d  nk: %d\n", ni, nj, nk);


        //Rescaling amplitudes
        Ampk = wLow * sqrt(pk/dx3);

        return cufftDoubleComplex{Ampk*b.x,Ampk*b.y};
      } else {
        return cufftDoubleComplex{0.0,0.0};
      }
    });
    
    if ( in_device ){
      CHECK( cudaMemcpy( output, da_, outputBytes_, cudaMemcpyDeviceToDevice));
    } else {
      CHECK( cudaMemcpy( output, da_, outputBytes_, cudaMemcpyDeviceToHost));
    } 
    
}

void FFT_3D::Filter_inv_k2( Real *const input, Real *const output, bool in_device ) const
{
  // Local copies of members for lambda capture
  const int ni = ni_, nj = nj_;
  const Real ddi = ddi_, ddj = ddj_, ddk = ddk_;
  const size_t bytes = minBytes_;
  
  if ( in_device ){
    CHECK( cudaMemcpy( db_, input, inputBytes_, cudaMemcpyDeviceToDevice));
  } else {
    CHECK( cudaMemcpy( db_, input, inputBytes_, cudaMemcpyHostToDevice));
  } 
    
  // Provide FFT filter with a lambda that does 1/k^2 solve in frequency space
  henry_->filter(bytes, db_, da_,
    [=] __device__ (const int i, const int j, const int k, const cufftDoubleComplex b) {
      if (i || j || k) {
        const Real i2 = sqr(double(min(i,ni-i))*ddi);
        const Real j2 = sqr(double(min(j,nj-j))*ddj);
        const Real k2 = sqr(double(k)*ddk);
        const Real d = -1.0/(i2+j2+k2);
        return cufftDoubleComplex{d*b.x,d*b.y};
      } else {
        return cufftDoubleComplex{0.0,0.0};
      }
    });
    
  if ( in_device ){
    CHECK( cudaMemcpy( output, da_, outputBytes_, cudaMemcpyDeviceToDevice));
  } else {
    CHECK( cudaMemcpy( output, da_, outputBytes_, cudaMemcpyDeviceToHost));
  } 
}



#endif

