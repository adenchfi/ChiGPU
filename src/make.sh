## assuming you've installed the CUDA toolkit, below you can specify your own include directory.

nvcc --use_fast_math -c -I/usr/lib/cuda-10.2/include bandstruct_TB_GPU_complex.cu

g++ -O3 -o bandstruct_TB_batch_gpu_complex bandstruct_TB_GPU_complex.o -L/usr/lib/cuda-10.2/lib64 -lcusolver -lcudart

nvcc --use_fast_math -c -I/usr/lib/cuda-10.2/include susc_calc_GPU.cu

g++ -O3 -o susc_calc_GPU susc_calc_GPU.o -L/usr/lib/cuda-10.2/lib64 -lcusolver -lcudart


#./bandstruct_TB_batch_gpu_complex LaTe3_tb_filtered.dat 25000 kpath.dat LaTe3_test_BS_complex
