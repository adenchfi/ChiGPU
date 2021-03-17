#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <sstream>
#include <fstream>
#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cusolverDn.h>
#include "cuComplex.h"

/*  
    This code's purpose is to compute the Lindhard susceptibility in a very efficient and quick manner, and use the GPU to do it.
    We can separate it into two convolutions over energy, being careful about divergences (replacing with derivatives of fermi function)
    Since there are N_orb energies per k_pt and N_k kpts, a convolution over energy using FFTs should take on the order of, using N = N_k * N_orb, 2*N log(N) floating point operations. A direct evaluation would be on the order of N^2 operations. When N ~ 10 billion, this is infeasible and we need the FFT route. 
    I should refer to the CUDA convolution and cufft examples
    
    
 */

__constant__ float dEf, dbeta, domega;
__constant__ int dnum_kpts;

__device__ float myatomicAdd(float* address, float val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ __forceinline__ float fermidist (const float E, const float explim)
{
    float f;
    
    if (abs(E) < 1e-5)
      {
	f = 0.5 + 0.25*E + E*E*E/48.0; // cubic order approximation of nf(x) for small x
	return f;
      }
    if (E > explim)
      {
	f = 0.0;
      }
    else if (E < -explim)
      {
	f = 1.0;
      }
    else
      {
	f = 1.0/(exp(E) + 1);
      }
    
    //float t = expf (z.x);
    //sincosf (z.y, &res.y, &res.x);
    //res.x *= t;
    //res.y *= t;
    return f;
}

// Forward declaration of the matrix multiplication kernel in case I don't want to put the definition at the beginning
__global__ void SuscKernel(const int*, const int*, const unsigned int, const float*, const int, float*);

// Matrix multiplication kernel called by MatMul()
__global__ void SuscKernel( int* tot_el_so_far, int* norb_per_kpt, unsigned int num_kpoints, float* Ekns, int * dqnum, float* chiq)
{
  
  // want to allocate shared memory array s.t. the whole virtual private array of thread 0 falls into shared memory bank 0, thread 1 into bank 1, etc. Use threadblock_size. see this blog post: https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/

  /* 
     Summary of data access:

     {n} = norb_per_kpt[kpt#] = number of nonzero orbitals for that kpt#
     N = tot_el_so_far[kpt#] = number needed to access Ekn[kpt] correctly, as Ekn[N + n] where n is within norb_per_kpt[kpt#]
     Ekn[N[kpt#]+n[kpt#]] = energy of n-th orbital of Nth kpt
     
   */  
  

  // extern __shared__ cuFloatComplex space[];
  
  short int k_norb; // same for whole k-list loop
  int kpq_num;
  short int kpq_norb;
  int N1, N2;
  int qnum = dqnum[0];

  float Ekpqm, Ekn;
  float tmpchiqomeg = 0.0;
  cuFloatComplex numerator, denom; 
  float nfEkpqm, nfEkn;
  float explim = 16.0;
  float delta = 0.00001;
  int tid = threadIdx.x;
  float exparg = 0.0;
  // if (tid == 1)
  //   {
  //     printf("Ef: %lf  beta: %lf  omega: %lf  num_kpts: %d", dEf, dbeta, domega, num_kpoints);
  //   }

  for (int myk = blockIdx.x * blockDim.x + threadIdx.x; myk < dnum_kpts; myk += blockDim.x*gridDim.x)
    {
      tmpchiqomeg = 0.0;      

      if (myk+qnum >= dnum_kpts)
	{
	  kpq_num = myk + qnum - dnum_kpts;
	}
      else if (myk + qnum < dnum_kpts)
	{
	  kpq_num = myk + qnum;
	}
      //kpq_num = (myk+qnum)%dnum_kpts;
      // I pray this fixes it
      
      k_norb = norb_per_kpt[myk];
      kpq_norb = norb_per_kpt[kpq_num];
      
      N1 = tot_el_so_far[myk];
      N2 = tot_el_so_far[kpq_num];
      //if (tid == 1) printf("tid: %d\n", tid);
      //if (tid == 1) printf("kpq_num: %d \t kpq_norb: %d \t myk: %d \t k_norb: %d\n", kpq_num, k_norb, myk, k_norb);
      // if (tid == 1) printf("N1: %d\t N2: %d\n", N1, N2);
      
      
      for (int n = 0; n < k_norb; n++)
	{
	  Ekn = Ekns[N1+n];
	  // calc fermidist of Ekn
	  nfEkn = fermidist(dbeta*(Ekn-dEf), explim);
	  
	  for (int m = 0; m < kpq_norb; m++)
	    {
	      // if (m != n) continue; // only orbital-diag susc, used by Kivelson
	      Ekpqm = Ekns[N2+m];
	      // calc fermidist of Ekpqm
	      nfEkpqm = fermidist(dbeta*(Ekpqm-dEf), explim);
	      
	      exparg = dbeta*(Ekpqm-dEf);
	      if (nfEkpqm == nfEkn) // 
		{
		  if (abs(cuCrealf(denom)) <= 1e-8)
		    {
		      tmpchiqomeg -= fermidist(dbeta*(Ekn-dEf-domega), explim)*dbeta*fermidist(-dbeta*(Ekn-dEf-domega), explim);
		      continue;
		    }
		}

	      numerator = make_cuFloatComplex(nfEkpqm - nfEkn, 0.0);
	      denom = make_cuFloatComplex(domega + Ekpqm - Ekn, delta);
	      tmpchiqomeg -= cuCrealf(cuCdivf(numerator, denom));	   
	      // if (qnum == 0)
	      // 	{
	      // 	  printf("qnum 0, tmpchiqomeg for m,n = %d is: %.10f  for energy 1 =  %lf, nfE1 = %.10f, energy 2 = %lf, nfE2 = %.10f, knum %d and kpqnum %d for beta=%.10f and exparg %.10f\n", m, tmpchiqomeg, Ekn, nfEkn, Ekpqm, nfEkpqm, myk, kpq_num, dbeta, exparg);
	      // 	}
	      // if (qnum == 1)
	      // 	{
	      // 	  printf("qnum 1, tmpchiqomeg for m,n = %d is: %.10f  for energy 1 =  %lf, nfE1 = %.10f, energy 2 = %lf, nfE2 = %.10f, knum %d and kpqnum %d for beta = %.10f and exparg %.10f\n", m, tmpchiqomeg, Ekn, nfEkn, Ekpqm, nfEkpqm, myk, kpq_num, dbeta, exparg);
	      // 	}
	      // if (qnum == 2)
	      // 	{
	      // 	  printf("qnum 2, tmpchiqomeg for m,n = %d is: %.10f  for energy 1 =  %lf, nfE1 = %.10f, energy 2 = %lf, nfE2 = %.10f,  and kpqnum %d for beta = %.10f and exparg %.10f\n", m, tmpchiqomeg, Ekn, nfEkn, Ekpqm, nfEkpqm, kpq_num, dbeta, exparg);
	      // 	}


	      // if (abs(domega + Ekpqm - Ekn) < delta)
	      // 	{
	      // 	  //		  if (tid == 1) printf("Ekn: %lf\t nfEkn: %lf\n", Ekn, nfEkn);
	      // 	  tmpchiqomeg -= fermidist(dbeta*(Ekn-dEf-domega), explim)*dbeta*fermidist(-dbeta*(Ekn-dEf-domega), explim); 
	      // 	}
	      // else
	      // 	{
	      // 	  tmpchiqomeg -= (nfEkpqm - nfEkn)/(domega + Ekpqm - Ekn);
	      // 	}
	      
	      //tmpchiqomeg += 0.01;
	      //	      tmpchiqomeg += stuff;
		    
	    }
	  
	}
      // add to chiqomeg
      __syncthreads();
       atomicAdd(&chiq[qnum], tmpchiqomeg); 
    }     
    // DONE WITH KERNEL
}




int main(int argc, char*argv[])
{
  std::string filename = argv[1];
  if (argv[2])
    {
      std::string outfilename = argv[2];    
    }


  // purpose of this is to compute chi(q,omega) = -2 sum_{kmn} n_f (e_{km} - nf_{ek+q,n})/(omega + e_km - e_{k+q}n)
  
  // the file is small enough now that it's ok if we loop through it a few times

  /* COUNT NUMBER OF LINES IN FILE */
  std::ifstream infile;
  infile.open(filename, std::ios::in);
  
  int lines = 0;
  std::string s;
  int num_Es = 0;
  float oneWord;
  int counter = 0;
  
  while (std::getline(infile, s))
    {
      std::stringstream stream(s);
      while(stream >> oneWord)
	{

	  if (counter >= 3)
	    {
	      ++num_Es;
	    }
	  ++counter;	  
	}
      counter=0;
      lines++;
    }

  // now have info to malloc array big enough to hold the info I want
  // maybe I can count the number of words in each line, too; and have an array with the number of energies in each line

  printf("\n Number of lines in tb file appears to be %d\n", lines);
  int * Esperline;
  Esperline = (int*) malloc (lines*sizeof(int));
  float * allEs;
  // int norb = 8;
  
  allEs = (float*) malloc(num_Es*sizeof(float)); // num_Es makes sure this array only contains the right number of energies
  
  int * els_so_far_this_kpt; // contains how many nonzero elements there have been before the current kpt
  
  els_so_far_this_kpt = (int*) malloc(lines*sizeof(int));
  els_so_far_this_kpt[0] = 0; // initialize

  float * ks;
  ks = (float*)malloc(lines*3*sizeof(float));
  
  infile.clear();
  infile.seekg(0, infile.beg); // go to beginning of file for next loop

  int num_words = 0; // going to use this to count number of Es per line
  lines=0;
  printf("\n Number of energies appears to be %d\n", num_Es);
  float beta =5.0;
  float Ef = 10.44;
  int threadsperblock = 512;
  //int numBlocks = 16384;
  int numBlocks = 256;
  float omega = 0.015;

  if (argv[3])
    {
      sscanf(argv[3], "%f", &omega);
    }
  float explim = 7.0;
  if (argv[4])
    {
      sscanf(argv[4], "%f", &explim);
    }
  float denomlim = 0.00001;
  
  if (argv[5])
    {
      sscanf(argv[5], "%f", &denomlim);
    }
  

  if (argv[6])
    {
      sscanf(argv[6], "%f", &beta);
    }

  if (argv[7])
    {
      sscanf(argv[7], "%f", &Ef);
    }

  while (std::getline(infile, s))
    {
      std::stringstream stream1(s);
      std::stringstream stream(s);
      int tmp_num_Words = 0;
      while (stream1 >> oneWord)
	{
	  ++num_words;
	}
      
      Esperline[lines] = num_words-3;
      
      if (lines >= 1)
	{
	  els_so_far_this_kpt[lines] = els_so_far_this_kpt[lines - 1] + num_words - 3;
	}

      while(stream >> oneWord)
	{
	  
	  if (tmp_num_Words > 2)
	    {
	      allEs[els_so_far_this_kpt[lines] + (tmp_num_Words - 3)] = oneWord;	       				
	    }
	  else if (tmp_num_Words <= 2)
	    {
	      ks[lines*3 + tmp_num_Words] = oneWord;
	    }

	  ++tmp_num_Words;
	}
      tmp_num_Words = 0;
      num_words = 0;
      lines++;
    }
  printf("num_Es: %d\n", num_Es);


  
  // now I have Esperline, allEs, els_so_far_this_kpt, and ks
  // n = Esperline[line#] gives me # of Es for this kpt
  // kx = ks[line# + 0] gives me kx, ky = ks[line# + 1] gives me ky, etc
  // N = els_so_far_this_kpt[line#] = array index for start of ks[line#] for the allEs object
  // allEs[N + n] gives me the n-th eigenenergy with crystal momentum k
  //   -- use A = Esperline[line#] for a sum range for n

  // now I have everything I need to compute chi(q, omega)
  //sscanf("%ld")
  int modlinenum = 0;
  int whichorb1 = 0;
  int numorbsthislinekpq = 0;
  int numorbsthislinek = 0;
  
  float Ekpqm = 0.0;
  float Ekn = 0.0;
  float nfEkpqm = 0.0;
  float nfEkn = 0.0;


  std::cout << allEs[0] << std::endl;
  std::cout << allEs[1] << std::endl;
  std::cout << allEs[2] << std::endl;
  std::cout << allEs[3] << std::endl;
  std::cout << allEs[4] << std::endl;
  std::cout << els_so_far_this_kpt[0] << std::endl;
  std::cout << els_so_far_this_kpt[1] << std::endl;
  std::cout << els_so_far_this_kpt[2] << std::endl;
  
  float chiqomeg = 0.0;
  float dubtmp;
  float * chiqomegs;
  int nomeg = 1;
  chiqomegs = (float*) malloc(lines*nomeg*sizeof(float));
  
  std::ofstream chioutfile;
  chioutfile.open(argv[2], std::ios::out | std::ios::trunc);

  // INITIALIZE, ALLOCATE, COPY arrays onto device global me

  if (num_Es > 100000000){
    std::cout << "Data too large to fit in GPU" << std::flush;
    return 1;
  }
  
  float * chiqomeg_one = (float*)malloc(1*sizeof(float));
  memset(chiqomeg_one, 0, 1*sizeof(float));
  
  int * d_qnum;

  int * d_Esperline;
  float * d_allEs;
  int * d_tot_els_this_kpt;
  float * d_chiqomegs;
  cudaError_t cudaStat1 = cudaSuccess;

  cudaStat1 = cudaMalloc((void**)&d_qnum, sizeof(int));
  
  cudaStat1 = cudaMalloc((void**)&d_Esperline, sizeof(int) * lines);

  assert(cudaSuccess == cudaStat1);

  cudaStat1 = cudaMemcpy(d_Esperline, Esperline, sizeof(int) * lines, cudaMemcpyHostToDevice);
  
  assert(cudaSuccess == cudaStat1);

  
  cudaStat1 = cudaMalloc((void**)&d_allEs, sizeof(float) * num_Es);
  cudaStat1 = cudaMemcpy(d_allEs, allEs, sizeof(float) * num_Es, cudaMemcpyHostToDevice);

  assert(cudaSuccess == cudaStat1);
  cudaStat1 = cudaMalloc((void**)&d_tot_els_this_kpt, sizeof(int) * lines);
  cudaStat1 = cudaMemcpy(d_tot_els_this_kpt, els_so_far_this_kpt, sizeof(int) * lines, cudaMemcpyHostToDevice);

  assert(cudaSuccess == cudaStat1);
  cudaStat1 = cudaMalloc((void**)&d_chiqomegs, sizeof(float) * lines);
  cudaStat1 = cudaMemset(d_chiqomegs, 0, lines*sizeof(float));
  
  assert(cudaSuccess == cudaStat1);
  
  std::cout << "\n Here 1 \n" << std::flush;
  cudaMemcpyToSymbol(dEf, &Ef, sizeof(float), 0, cudaMemcpyHostToDevice);
  std::cout << "\n Here 2 \n" << std::flush;
  cudaMemcpyToSymbol(dbeta, &beta, sizeof(float), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(domega, &omega, sizeof(float), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(dnum_kpts, &lines, sizeof(int), 0, cudaMemcpyHostToDevice);
  std::cout << "\n Here 3 \n" << std::flush;
  // FINISHED COPYING DATA ARRAYS TO GPU

  
  // SEEMS GOOD UP TO HERE ASIDE FROM POSSIBLY GPU KERNEL
  
  for (int qlinenum=0; qlinenum<lines; qlinenum++)
    {
        float qx = ks[3*qlinenum];
	float qy = ks[3*qlinenum+1];
	float qz = ks[3*qlinenum+2];
	// if ((qy > 1e-6) || qx > 1e-6) continue; // want to only do qy = 0.0
	/*  
	    BEGIN GPU ALGORITHM
	    TO CALC SUSC FOR THIS q-pt
	    PLAN BLOCK,GRID,IDX STRUCTURE
	    if I have 65k blocks of 1024 threads each, that's 65mil threads -> more than enough if I have only ~20mil kpts 
	    let's say I do numkpts/num_threads_per_block kpts then
	    doing 256 threads per block sounds good
	    ~ ceil(numkpts/thread_per_block) ~ ceil(20mil/256) ~ 200k blocks
	    ~ exceeds block threshold, but we can just assign more work to the threads
	    ~ let's say then I do 2^14 = 16384 blocks
	    ~ would need each thread to be responsible for like 10+ kpts
	    ~ which sounds ok to me
	*/
	if (qlinenum % 3 == 0)
	  {
	    std::cout << "\n Here at qlinenum " << qlinenum << "\n" << std::flush;
	  }

	cudaStat1 = cudaMemcpy(d_qnum, &qlinenum,  sizeof(int), cudaMemcpyHostToDevice);

	assert(cudaSuccess == cudaStat1);
	
	SuscKernel<<<numBlocks,threadsperblock>>>(d_tot_els_this_kpt, d_Esperline, lines, d_allEs, d_qnum, d_chiqomegs);

	cudaStat1 = cudaDeviceSynchronize();
	
	assert(cudaSuccess == cudaStat1);

	
	cudaStat1 = cudaMemcpy(chiqomeg_one, &d_chiqomegs[qlinenum+0], sizeof(float), cudaMemcpyDeviceToHost);

	assert(cudaSuccess == cudaStat1);

	
	chioutfile << ks[3*qlinenum+0] << "\t" << ks[3*qlinenum+1] << "\t" << ks[3*qlinenum+2] << "\t" << chiqomeg_one[0] << std::endl;	
	memset(chiqomeg_one, 0, 1*sizeof(float));
	
    }

  
  
  return 0;
}
