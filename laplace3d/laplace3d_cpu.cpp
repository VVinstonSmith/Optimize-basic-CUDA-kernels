//
// Program to solve Laplace equation on a regular 3D grid
//

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>


////////////////////////////////////////////////////////////////////////////////
void laplace3d_cpu(int NX, int NY, int NZ, float* u1, float* u2) 
{
  int   i, j, k, ind;
  float sixth=1.0f/6.0f;  

  for (k=0; k<NZ; k++) {
    for (j=0; j<NY; j++) {
      for (i=0; i<NX; i++) {   // i loop innermost for sequential memory access
	   ind = i + j*NX + k*NX*NY;

        if (i==0 || i==NX-1 || j==0 || j==NY-1|| k==0 || k==NZ-1) {
          u2[ind] = u1[ind];          // Dirichlet b.c.'s
        }
        else {
          u2[ind] = ( u1[ind-1    ] + u1[ind+1    ]
                    + u1[ind-NX   ] + u1[ind+NX   ]
                    + u1[ind-NX*NY] + u1[ind+NX*NY] ) * sixth;
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv){

  int    NX, NY, NZ, NITER;
  int    bx, by, i, j, k, ind;
  float  *h_u1, *h_u2, *h_foo;
  double err;
  double time_start,time_stop,time_wall;
  
  if (argc < 5) 
   {
	   printf("\n Usage: laplace3d <NX> <NY> <NZ> <NITER>\n");		
	   return -1;
   }

   NX = atoi(argv[1]);	
   NY = atoi(argv[2]);
   NZ = atoi(argv[3]);
   NITER = atoi(argv[4]);      

  printf("\n Grid dimensions: %d x %d x %d. NITER: %d \n", NX, NY, NZ, NITER);

  // allocate memory for arrays

  h_u1 = (float *)malloc(sizeof(float)*NX*NY*NZ);
  h_u2 = (float *)malloc(sizeof(float)*NX*NY*NZ);

  // initialise h_u1
#pragma omp parallel for private(i,j,k) schedule(static)  
  for (k=0; k<NZ; k++) {
    for (j=0; j<NY; j++) {
      for (i=0; i<NX; i++) {
        ind = i + j*NX + k*NX*NY;

        if (i==0 || i==NX-1 || j==0 || j==NY-1|| k==0 || k==NZ-1)
          h_u1[ind] = 1.0f;           // Dirichlet b.c.'s
        else
          h_u1[ind] = 0.0f;
      }
    }
  }


    // initialise timing
  time_start = omp_get_wtime();  

#pragma omp parallel for private(i,j,k) schedule(static)  
  for (i = 1; i <= NITER; ++i) {
    laplace3d_cpu(NX, NY, NZ, h_u1, h_u2);
	
// swap h_u1 and h_u2	
    h_foo = h_u1; 
	h_u1 = h_u2; 
	h_u2 = h_foo;   
  }

  // stop timing
  time_stop = omp_get_wtime();  
  time_wall = (time_stop - time_start)*1000.0;
  
  printf("\n Laplace3d runs in: %.2f (ms) \n", time_wall);

  err = 0.0;
#pragma omp parallel for private(i,j,k) reduction(+:err)
  for (k=0; k<NZ; k++) {
	  for (j=0; j<NY; j++) {
		  for (i=0; i<NX; i++) {
			  ind = i + j*NX + k*NX*NY;
			  err += (h_u1[ind]-h_u2[ind])*(h_u1[ind]-h_u2[ind]);
		  }
	  }
  }

  printf("\n rms error = %f \n",sqrt(err/ (double)(NX*NY*NZ)));  
  
 // Release memory
  free(h_u1);
  free(h_u2);
}
