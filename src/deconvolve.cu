#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

extern "C" void deconvolve (int N1, int N2, double *uIni, double *srcImg, int itertime, double h, double lambda, double delta, double epsilon, double * dataNow);

__global__ void nablaIni_kernel(int N1, int N2, double *nablaU){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int j=blockIdx.y*blockDim.y+threadIdx.y;
  if (i<N1&&j<N2){
    nablaU[i*N2+j]=0;
  }
}

__global__ void normNabla_kernel(int N1, int N2, double *u, double h, double * nablaU){
  double ex, ey;
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int j=blockIdx.y*blockDim.y+threadIdx.y;

  if (i<N1-1&&j<N2-1&&i>0&&j>0){
    ex=(u[(i+1)*N2+j]-u[(i-1)*N2+j])/2/h;
    ey=(u[i*N2+j+1]-u[i*N2+j-1])/2/h;
    nablaU[i*N2+j]=sqrt(ex*ex+ey*ey);
  }
}

__global__ void normMinus_kernel(int N1, int N2, double * a, double * b, double * result){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int j=blockIdx.y*blockDim.y+threadIdx.y;

  if (i<N1&&j<N2){
    result[i*N2+j]=abs(a[i*N2+j]-b[i*N2+j]);
  }
}

__global__ void fcal_kernel(int N1, int N2, double *srcImg, double *f, double *nablaU, double *uMinusf, double lambda, double delta, double epsilon){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int j=blockIdx.y*blockDim.y+threadIdx.y;

  if (i<N1&&j<N2){
    f[i*N2+j]=lambda*srcImg[i*N2+j]/sqrt(uMinusf[i*N2+j]*uMinusf[i*N2+j]+delta)*sqrt(nablaU[i*N2+j]*nablaU[i*N2+j]+epsilon);
  }
}

__global__ void deblur_kernel(int N1, int N2, double *u, double *srcImg, double *deblurU, double *f, double *nablaU, double *uMinusf, double h, double lambda, double delta, double epsilon){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int j=blockIdx.y*blockDim.y+threadIdx.y;

  if (i<N1-1&&j<N2-1&&i>0&&j>0){
    deblurU[i*N2+j]=(h*h*f[i*N2+j]+u[(i-1)*N2+j]+u[i*N2+j-1]+u[(i+1)*N2+j]+u[i*N2+j+1])/(4+lambda*sqrt(nablaU[i*N2+j]*nablaU[i*N2+j]+epsilon)*h*h/sqrt(uMinusf[i*N2+j]*uMinusf[i*N2+j]+delta));
    if (deblurU[i*N2+j]>255){deblurU[i*N2+j]=255;}
    if (deblurU[i*N2+j]<0){deblurU[i*N2+j]=0;}
  }
}

__global__ void deblur_kernel(int N1, int N2, double *nabla){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int j=blockIdx.y*blockDim.y+threadIdx.y;
  if(i<N1&&j<N2){
    nabla[i*N2+j]=0;
  }
}

double residual(int N1, int N2, double* u, double* f) {
        double residual_norm_sq = 0.0;
        for (int i = 0; i < N1 * N2; i++) {
        residual_norm_sq += (u[i] - f[i]) * (u[i] - f[i]);
        }
        return sqrt(residual_norm_sq);
}

void deconvolve (int N1, int N2, double *uIni, double *srcImg, int itertime, double h, double lambda, double delta, double epsilon, double * u) {
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(N1/BLOCK_SIZE+1,N2/BLOCK_SIZE+1);

  double *u_d,*u_prev_d, *f, *nablaU, *uMinusf;
  double *srcImg_d;
  cudaMalloc(&u_d,N1*N2*sizeof(double));
  cudaMalloc(&u_prev_d,N1*N2*sizeof(double));
  cudaMalloc(&f,N1*N2*sizeof(double));
  cudaMalloc(&nablaU,N1*N2*sizeof(double));
  cudaMalloc(&uMinusf,N1*N2*sizeof(double));
  cudaMalloc(&srcImg_d, N1*N2*sizeof(double));

  cudaMemcpy(srcImg_d, srcImg, N1*N2*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(u_d,uIni,N1*N2*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(u_prev_d,uIni,N1*N2*sizeof(double),cudaMemcpyHostToDevice);

  clock_t t;
  t=clock();
  printf("Initial residule: %f\n", residual(N1,N2, uIni, srcImg));
  printf("time= %f seconds\n", ((float)t)/CLOCKS_PER_SEC);
  for (int i=0; i<itertime;i++){
    //printf("itertime=%d\n",i);
    nablaIni_kernel<<<dimGrid,dimBlock>>>(N1,N2,nablaU);
    normNabla_kernel<<<dimGrid,dimBlock>>>(N1, N2, u_d, h, nablaU);
    normMinus_kernel<<<dimGrid,dimBlock>>>(N1, N2, u_d, srcImg_d, uMinusf);
    fcal_kernel<<<dimGrid,dimBlock>>>(N1,N2,srcImg_d,f,nablaU,uMinusf,lambda, delta,epsilon);
    deblur_kernel<<<dimGrid,dimBlock>>>(N1,N2, u_prev_d,srcImg_d,u_d,f, nablaU, uMinusf, h, lambda, delta, epsilon);
    cudaMemcpy(u_prev_d,u_d,N1*N2*sizeof(double),cudaMemcpyDeviceToDevice);
  }
  cudaMemcpy(u,u_prev_d,N1*N2*sizeof(double),cudaMemcpyDeviceToHost);

  //cudaMemcpy(u, srcImg_d, N1*N2*sizeof(double), cudaMemcpyDeviceToHost);
  
  t=clock()-t;
  printf("Final residule: %f\n", residual(N1,N2, u, srcImg));
  printf("time= %f seconds\n", ((float)t)/CLOCKS_PER_SEC);
  cudaFree(u_d);
  cudaFree(u_prev_d);
  cudaFree(f);
  cudaFree(nablaU);
  cudaFree(uMinusf);
  cudaFree(srcImg_d);
}

