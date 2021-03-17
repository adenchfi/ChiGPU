/*
 * How to compile (assume cuda is installed at /usr/local/cuda/)
 *   nvcc -c -I/usr/local/cuda/include batchsyevj_example.cpp 
 *   g++ -o batchsyevj_example batchsyevj_example.o -L/usr/local/cuda/lib64 -lcusolver -lcudart
 *
 */
#define THREADBLOCK_SIZE 32
// should be multiple of 32, here we use 32 since I think my arrays are close to maxing out the available shared memory

/*  TODO
    Outputted bandstructure appears to be not quite correct. Why?
    Is it:
    1) Incorrect reading of TB matrix elements?
    2) Incorrect construction of TB matrix?
       -- Incorrect symmetrization?
 */



#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cusolverDn.h>
#include <math.h>
#include <sstream>
#include <fstream>
#include <string>
#include <iostream>
#include <complex>      // std::complex
#include <iomanip>      // std::setprecision
#include "cuComplex.h"
#include <cstdlib>

__device__ __forceinline__ cuFloatComplex my_cexpd (cuFloatComplex z)
{
    cuFloatComplex res;
    float t = expf (z.x);
    sincosf (z.y, &res.y, &res.x);
    res.x *= t;
    res.y *= t;
    return res;
}

// Forward declaration of the matrix multiplication kernel in case I don't want to put the definition at the beginning
__global__ void MatConstructKernel(const float*, const float*, const int*, const float*, const int*, cuFloatComplex*, const int, const int, const int);

// Matrix multiplication kernel called by MatMul()
__global__ void MatConstructKernel(float* tdeltamn, float* tdeltamncomplex, int* deltapositions, float* kpoints, int* orbset, cuFloatComplex* A0, int m_dim, int mk_batchSize, int nline_tbfile)
{
  
  // want to allocate shared memory array s.t. the whole virtual private array of thread 0 falls into shared memory bank 0, thread 1 into bank 1, etc. Use threadblock_size. see this blog post: https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/
  extern __shared__ cuFloatComplex local_A[]; // the size of this is actually given in the kernel call; should be threadblock_size * sizeof(array) where sizeof(array) is m*m, and note threadblock_size = num_k considered, e.g. 32 here
  // should have space for more things in shared memory, so maybe I could have orbset in here too
  
  // want local_A to look like [k1m1n1 k1m1n2 k1m1n3 km1n4 ... k2m1n1 ...] as each thread will use a different k

  // if each thread block (size 32) deals w/ 32 k-pts, and there are batchSize kpts, then we should have batchSize/blocksize blocks,
  for (int myk = blockIdx.x * blockDim.x + threadIdx.x; myk < mk_batchSize; myk += blockDim.x*gridDim.x)
    {
      int tid = threadIdx.x;
      cuFloatComplex j = make_cuFloatComplex(0.0, 1.0);
  
      float mykx = 2*M_PI*kpoints[3*myk + 0];
      float myky = 2*M_PI*kpoints[3*myk + 1];
      float mykz = 2*M_PI*kpoints[3*myk + 2];

      int tmp3; 
      cuFloatComplex tmp1, tmp2, tmpexparg1, tmpexparg2;

      for (int n1 = 0; n1 < m_dim; n1++)
	{
	  for (int n2 = 0; n2 < m_dim; n2++)
	    {
	      // myk is a global array param, I want to use local array param, e.g. my threadidx?
	      local_A[n1 + m_dim*n2 + m_dim*m_dim*tid] = make_cuFloatComplex(0.0,0.0); // initialize
	    }
	}
  
      for (int line_number = 0; line_number < nline_tbfile; line_number++)
	{
	  tmp1 = make_cuFloatComplex(tdeltamn[line_number], tdeltamncomplex[line_number]);
	  tmpexparg1 = make_cuFloatComplex(0.0, (mykx * deltapositions[3*line_number + 0] + myky*deltapositions[3*line_number + 1] + mykz*deltapositions[3*line_number + 2]));
	  tmp1 = cuCmulf(tmp1, my_cexpd(tmpexparg1));
            
	  tmp3 = (orbset[2*line_number + 0]-1) + m_dim*(orbset[2*line_number + 1]-1) + m_dim*m_dim*tid; 

	  local_A[tmp3] = cuCaddf(local_A[tmp3], tmp1);
	}
      // finished with line_num loop
      // can now write local_A to global memory

      for (int n1 = 0; n1 < m_dim; n1++)
	{
	  for (int n2 = 0; n2 < m_dim; n2++)
	    {
	      A0[n1 + m_dim*n2 + m_dim*m_dim*myk] = local_A[n1 + m_dim*n2 + m_dim*m_dim*tid];
	    }
	}

      // Each thread computes one element of C
      // by accumulating results into Cvalue
      /* 
	 float Cvalue = 0;
	 int row = blockIdx.y * blockDim.y + threadIdx.y;
	 int col = blockIdx.x * blockDim.x + threadIdx.x;
	 for (int e = 0; e < A.width; ++e)
	 Cvalue += A.elements[row * A.width + e]
	 * B.elements[e * B.width + col];
	 C.elements[row * C.width + col] = Cvalue;
      */ 
      
    }

  
    
    // DONE WITH KERNEL
}

void printMatrix(int m, int n, const float*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            float Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
        }
    }
}

int main(int argc, char*argv[])
{
  std::setprecision(10);
  cusolverDnHandle_t cusolverH = NULL;
  cudaStream_t stream = NULL;
  syevjInfo_t syevj_params = NULL;

  cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
  cudaError_t cudaStat1 = cudaSuccess;
  cudaError_t cudaStat2 = cudaSuccess;
  cudaError_t cudaStat3 = cudaSuccess;

  int m = atoi(argv[5]); // m = number of orbitals in TB model; 1<= m <= 32 is supported apparently, probably since 32 = num of threads in thread block
  int lda = m;
  int batchSize = 42000;
  
  
  
  /* READ TB MODEL PARAMS */
    
  std::string infilename;
  std::stringstream(argv[1]) >> infilename;
  //sscanf(argv[1], "%s", &infilename);
  printf("%s\n", argv[1]);
  std::ifstream file(infilename);

  std::string filestr;

  int nline_tb_file = 0; // update when reading the file
  while (std::getline(file, filestr))
    {
      nline_tb_file++; 
    }
  int *deltapos; // contains lattice vector positions in file, corresponding to delta
  int *orbset; // contains the orbitals
  float *tdelta_mn; // contains the t_{delta}^nm values
  float * tdelta_mn_complex; // complex parts
  deltapos = (int*) calloc(3*nline_tb_file, sizeof(int));
  int *d_deltapos = NULL;
  float *d_tdelta_mn = NULL;
  float *d_tdelta_mn_complex = NULL;
  int *d_orbset = NULL;
  
  orbset = (int*) calloc(2*nline_tb_file,sizeof(int));
  tdelta_mn = (float *) calloc(nline_tb_file, sizeof(float));
  tdelta_mn_complex = (float *) calloc(nline_tb_file, sizeof(float));

  nline_tb_file = 0; // reset to reuse for next loop through
  file.clear();
  file.seekg(0, file.beg); // go to beginning of file for next loop

  while (std::getline(file, filestr))
    {
      std::istringstream str (filestr); // load str into string stream
      str >> deltapos[3*nline_tb_file] >> deltapos[3*nline_tb_file + 1] >> deltapos[3*nline_tb_file + 2] >> orbset[2*nline_tb_file] >> orbset[2*nline_tb_file+1] >> tdelta_mn[nline_tb_file] >> tdelta_mn_complex[nline_tb_file]; // parse line into the variable as shown.
      
      nline_tb_file++; 
    }

  // DONE EXTRACTING TB MATRIX
  // ALL GOOD UP TO HERE
  
  
  // READ KMESH FILE
  int numk; // same thing as nk, to fix  
  float kpoint[3];
  kpoint[0] = 0.0;
  kpoint[1] = 0.0;
  kpoint[2] = 0.0;
  // use argv[3] as filename for kpoints
  std::string kfilename;
  std::stringstream(argv[3]) >> kfilename;
  std::ifstream kfile(kfilename);
  std::string kfilestr;
  // I'll use the same format as QE;
  std::getline(kfile, kfilestr); // first line should be label KPOINTS
  std::getline(kfile, kfilestr); // second line should be # of kpts
  std::stringstream(kfilestr) >> numk;
  // DONE WITH EXTRACTING TOTAL NUMKPTS
  
  int maxBatchSize = 30000; // my laptop can handle 250k maybe? but the memory copying over to the gpu doesn't take that long
  // DETERMINE numBatches
  // sscanf(argv[2], "%d", &batchSize);
  int numBatches = (int) numk / maxBatchSize;
  int lastBatchsize = numk % maxBatchSize;
  
  if (numk > maxBatchSize)
    {
      batchSize = maxBatchSize;
    }
  else if (numk == maxBatchSize){
      batchSize = maxBatchSize;
      lastBatchsize = maxBatchSize;
  }
  
  
  float * ks; // initialize and allocate if mesh/path file given
  float *d_ks = NULL;
  ks = (float*) calloc(3*batchSize, sizeof(float));
  cudaStat1 = cudaMalloc ((void**)&d_ks   , sizeof(float) * 3*batchSize);	    
  assert(cudaSuccess == cudaStat1);
  int whichknum = 0;
  
  
  // PREP MATRICES
  // float * A; // TB matrix with all ks

  float * W; // stores TB eigenvalues with all ks
  int * info;
  
  // A = (float*) calloc(lda*m*batchSize, sizeof(float));
  // float * A0 = A;  /* Pointer pointing at beginning of A */
  
  W = (float*) calloc(m*batchSize, sizeof(float));
  info = (int*) calloc(batchSize, sizeof(int));

  cuFloatComplex *d_A  = NULL; /* lda-by-m-by-batchSize */
  
  float *d_W  = NULL; /* m-by-batchSize */
  int* d_info  = NULL; /* batchSize */
  int lwork = 0;  /* size of workspace */
  cuFloatComplex *d_work = NULL; /* device workspace for syevjBatched */

  // we have the m, nline_tb_file constants
  /* 
  int *d_m = NULL;
  int *d_nline_tb = NULL;
  cudaStat1 = cudaMalloc((void**)&d_m, sizeof(int));
  cudaStat1 = cudaMalloc((void**)&d_nline_tb, sizeof(int));
  
  cudaStat1 = cudaMemcpy(d_m, m, sizeof(int), cudaMemcpyHostToDevice);
  cudaStat1 = cudaMemcpy(d_nline_tb, nline_tb_file, sizeof(int), cudaMemcpyHostToDevice);
  */
  
  // we have the orbset, deltapos, tdeltamn, tdeltamncomplex arrays
  // orbset
  cudaStat1 = cudaMalloc ((void**)&d_orbset, sizeof(int) * 2*nline_tb_file);
  assert(cudaSuccess == cudaStat1);
  cudaStat1 = cudaMemcpy(d_orbset, orbset, sizeof(int) * 2*nline_tb_file, cudaMemcpyHostToDevice);
  assert(cudaSuccess == cudaStat1);
  // deltapos
  cudaStat1 = cudaMalloc ((void**)&d_deltapos   , sizeof(int) * 3*nline_tb_file);
  assert(cudaSuccess == cudaStat1);
  cudaStat1 = cudaMemcpy(d_deltapos, deltapos, sizeof(int) * 3*nline_tb_file, cudaMemcpyHostToDevice);
  assert(cudaSuccess == cudaStat1);

  // tdeltamn
  cudaStat1 = cudaMalloc ((void**)&d_tdelta_mn   , sizeof(float) * nline_tb_file);
  assert(cudaSuccess == cudaStat1);
  cudaStat1 = cudaMemcpy(d_tdelta_mn, tdelta_mn, sizeof(float) * nline_tb_file, cudaMemcpyHostToDevice);
  assert(cudaSuccess == cudaStat1);

  // tdeltamn_complex
  cudaStat1 = cudaMalloc ((void**)&d_tdelta_mn_complex   , sizeof(float) * nline_tb_file);
  assert(cudaSuccess == cudaStat1);
  cudaStat1 = cudaMemcpy(d_tdelta_mn_complex, tdelta_mn_complex, sizeof(float) * nline_tb_file, cudaMemcpyHostToDevice);
  assert(cudaSuccess == cudaStat1);
  

  const float tol = 1.e-8;
  const int max_sweeps = 30;
  const int sort_eig  = 1;   /* don't sort eigenvalues if 0 */
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR; /* don't compute eigenvectors */
  const cublasFillMode_t  uplo = CUBLAS_FILL_MODE_LOWER;


  /* OUTPUT FILE */
  std::string outfilename;
  std::stringstream(argv[4]) >> outfilename;
  //sscanf(argv[1], "%s", &infilename);
  std::ofstream myfile;
  myfile.open(outfilename); // we want to open it BEFORE the batch loop
  
  /* residual and executed_sweeps are not supported on syevjBatched */
  //float residual = 0;
  //int executed_sweeps = 0;

  int N = 32;
  int threadsPerBlock = N;
  int numBlocks = 1;

  // for debugging purposes we define some temp variables;

  // TODO: check if the below affects performance; changes shared memory bank size to be float the normal (4 byte is normal), which helps with float precision data
  // cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

  for (int batchnum = 0; batchnum <= numBatches; batchnum++)
    {
      if (batchnum == numBatches )
	{
	  if (lastBatchsize == 0)
	    {
	      break; // we don't want to do the next batch because there is no next batch
	    }
	  batchSize = lastBatchsize;
	  // DEALLOCATE the previous large arrays, and instead re-allocate the smaller arrays in this last batch
	  /* 
	  if (d_A    ) cudaFree(d_A);
	  if (d_W    ) cudaFree(d_W);
	  if (d_info ) cudaFree(d_info);
	  if (d_work ) cudaFree(d_work);
	  free(A); free(W); free(ks); free(info);
	  */
	}
      else if (batchnum < numBatches){
	batchSize = maxBatchSize;
	if (batchnum == 0)
	  {
	    cudaStat1 = cudaMalloc ((void**)&d_A   , sizeof(cuFloatComplex) * lda * m * batchSize);
	    assert(cudaSuccess == cudaStat1);
	  }
      }
      printf(" On batchnum %d with batchsize %d\n", batchnum, batchSize);
      // FOR GIVEN BATCH, GENERATE SOME CUDA PARAMETERS
      numBlocks = batchSize / threadsPerBlock; // maybe? or is there a max numBlocks? there is; 65,000 blocks
      if (numBlocks > 1024){
	numBlocks = 1024; // just to make sure # of active threads is fine; we have a grid Stride loop active 
      }
      //extent = make_cudaExtent(batchSize*sizeof(float),m, m );
      // making a 3D reference volume for the array
      
      // 65k blocks is fine, x 32 threads per block, > maxBatchSize anyway


      for (int mk=0; mk<batchSize; mk++)
	{
	  std::getline(kfile, kfilestr);
	  std::stringstream(kfilestr) >> ks[3*mk+0] >> ks[3*mk + 1] >> ks[3*mk + 2];
	  whichknum++;
	}
      // now ks are loaded into an array
      whichknum=0;
      // START GPU CONTRUCTION OF TB MATRIX??
      // 1) initialize the GPU versions of the inputs
      // 2) copy over values of non-batch-dep. functions above
      // 3) copy over the values of inputs that are different for different batches; only ks needs to be copied over here
      if (batchnum == 0)
	{
	        status = cusolverDnCreate(&cusolverH);
	}

      cudaStat1 = cudaMemcpy(d_ks, ks, sizeof(float) * 3*batchSize, cudaMemcpyHostToDevice);

      // synchronize, make sure all input data is here before calling kernel
      assert(cudaSuccess == cudaStat1);
      cudaStat1 = cudaDeviceSynchronize();
      assert(cudaSuccess == cudaStat1);
      
      // 4) call kernel
      MatConstructKernel<<<numBlocks,THREADBLOCK_SIZE,THREADBLOCK_SIZE*m*m*sizeof(cuFloatComplex)>>>(d_tdelta_mn, d_tdelta_mn_complex, d_deltapos, d_ks, d_orbset, d_A, m, batchSize, nline_tb_file);
      
      cudaStat1 = cudaDeviceSynchronize();
      assert(cudaSuccess == cudaStat1);
       
      // PUT ABOVE TB MAT CONSTRUCTION INTO CUDA KERNEL

      // CALL CUDA KERNEL; DON'T NEED TO COPY H(k) back to CPU just to move it back to GPU

      // 
      
      std::cout << "About to diagonalize H(k) for batch: " << batchnum << "of " << numBatches << std::endl;
      std::cout << std::flush;
    
      /* step 1: create cusolver handle, bind a stream  */


      // assert(CUSOLVER_STATUS_SUCCESS == status);
      if (batchnum == 0)
	{
	  cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	  assert(cudaSuccess == cudaStat1);

	  status = cusolverDnSetStream(cusolverH, stream);
	  assert(CUSOLVER_STATUS_SUCCESS == status);

	  /* step 2: configuration of syevj */
	  status = cusolverDnCreateSyevjInfo(&syevj_params);
	  assert(CUSOLVER_STATUS_SUCCESS == status);

	  /* default value of tolerance is machine zero */
	  status = cusolverDnXsyevjSetTolerance(
						syevj_params,
						tol);
	  assert(CUSOLVER_STATUS_SUCCESS == status);

	  /* default value of max. sweeps is 100 */
	  status = cusolverDnXsyevjSetMaxSweeps(
						syevj_params,
						max_sweeps);
	  assert(CUSOLVER_STATUS_SUCCESS == status);

	  /* disable sorting with this parameter */
	  status = cusolverDnXsyevjSetSortEig(
					      syevj_params,
					      sort_eig);
	  assert(CUSOLVER_STATUS_SUCCESS == status);
	  
	}




      /* step 3: copy A to device */
      if (batchnum == 0)
	{
	  
	  cudaStat2 = cudaMalloc ((void**)&d_W   , sizeof(float) * m * batchSize);
	  cudaStat3 = cudaMalloc ((void**)&d_info, sizeof(int   ) * batchSize);

	}

      assert(cudaSuccess == cudaStat2);
      assert(cudaSuccess == cudaStat3);
      
      // cudaStat1 = cudaMemcpy(d_A, A, sizeof(float) * lda * m * batchSize, cudaMemcpyHostToDevice);
      cudaStat1 = cudaDeviceSynchronize();
      assert(cudaSuccess == cudaStat1);
      
      assert(cudaSuccess == cudaStat2);

      /* step 4: query working space of syevjBatched */
      status = cusolverDnCheevjBatched_bufferSize(
						  cusolverH,
						  jobz,
						  uplo,
						  m,
						  d_A,
						  lda,
						  d_W,
						  &lwork,
						  syevj_params,
						  batchSize
						  );
      printf("lwork: %d\n", lwork);
      assert(CUSOLVER_STATUS_SUCCESS == status);
      if (batchnum == 0)
	{
	  cudaStat1 = cudaMalloc((void**)&d_work, sizeof(cuFloatComplex)*lwork);
	}

      assert(cudaSuccess == cudaStat1);

      /* step 5: compute spectrum of M(k) for all k? */
      // TODO: we can convert this to float by calling instead
      // cusolverDnSsyevjBatched_bufferSize
      // cusolverDnSsyevjBatched
      status = cusolverDnCheevjBatched(
				       cusolverH,
				       jobz,
				       uplo,
				       m,
				       d_A,
				       lda,
				       d_W,
				       d_work,
				       lwork,
				       d_info,
				       syevj_params,
				       batchSize
				       );
      cudaStat1 = cudaDeviceSynchronize();
      assert(cudaSuccess == cudaStat1);
      assert(CUSOLVER_STATUS_SUCCESS == status);


      // cudaStat1 = cudaMemcpy(V    , d_A   , sizeof(float) * lda * m * batchSize, cudaMemcpyDeviceToHost);
      cudaStat2 = cudaMemcpy(W    , d_W   , sizeof(float) * m * batchSize      , cudaMemcpyDeviceToHost);
      cudaStat3 = cudaMemcpy(info, d_info, sizeof(int) * batchSize             , cudaMemcpyDeviceToHost);
      // assert(cudaSuccess == cudaStat1);
      assert(cudaSuccess == cudaStat2);
      assert(cudaSuccess == cudaStat3);


      /* Step 6: show eigenvalues and eigenvectors */
      float *W0 = W; // pointer to the eigenvalues file
      // actually I want to save them to file; the eigenvalues for each k, and then later plot them

      whichknum=0; // reset knums? not sure
      
      // WRITE TO FILE MAKING SURE TO APPEND AND NOT ERASE WHAT WE'VE DONE
      for (int mk=0; mk < batchSize; mk++)
	{
	  /*
	    kpoint[0] = mk*kpath_direction[0];
	    kpoint[1] = mk*kpath_direction[1];
	    kpoint[2] = mk*kpath_direction[2];
	  */
	  kpoint[0] = 2*M_PI*ks[3*mk + 0];
	  kpoint[1] = 2*M_PI*ks[3*mk + 1];
	  kpoint[2] = 2*M_PI*ks[3*mk + 2];
	  myfile << std::setprecision(10) << kpoint[0] << "\t \t";
	  myfile << std::setprecision(10) << kpoint[1] << "\t \t";
	  myfile << std::setprecision(10) << kpoint[2] << "\t \t";
	  for (int i=0; i<m; i++)
	    {
	      myfile << std::setprecision(10) << W0[m*mk + i] << "\t \t"; // the eigenvalues for each kpt
	    }
	  myfile  << std::endl << std::flush;
	}

      printf("End of batch %d\n", batchnum); std::cout << std::flush;
    } // END OF THIS BATCH CALC.
  
  // DONE WITH BATCHES; END OF PROGRAM
  /* free resources */
  if (d_A    ) cudaFree(d_A);
  if (d_W    ) cudaFree(d_W);
  if (d_info ) cudaFree(d_info);
  if (d_work ) cudaFree(d_work);
  if (d_deltapos) cudaFree(d_deltapos);
  if (d_orbset) cudaFree(d_orbset);
  if (d_tdelta_mn) cudaFree(d_tdelta_mn);
  if (d_tdelta_mn_complex) cudaFree(d_tdelta_mn_complex);
  if (d_ks) cudaFree(d_ks);

  if (cusolverH) cusolverDnDestroy(cusolverH);
  if (stream      ) cudaStreamDestroy(stream);
  if (syevj_params) cusolverDnDestroySyevjInfo(syevj_params);

  cudaDeviceReset();

  return 0;  

}



