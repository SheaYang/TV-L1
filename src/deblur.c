/****************************************
 deblur.c
 ****************************************/

#include <deblur.h>
#include "include.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
// cuda
#include <cuda.h>
#include <cuda_runtime.h>

void normNabla(int N1, int N2, double *u, double h, double * nablaU){
  double ex, ey;
  for (int i=0; i<N1*N2; i++) nablaU[i]=0;
  for (int i=1; i<N1-1;i++){
    for (int j=1; j<N2-1; j++){
      ex=(u[(i+1)*N2+j]-u[(i-1)*N2+j])/2/h;
      ey=(u[i*N2+j+1]-u[i*N2+j-1])/2/h;
      nablaU[i*N2+j]=sqrt(ex*ex+ey*ey);
    }
  }
}

void normMinus(int N1, int N2, double * a, double * b, double * result){
  for (int i=0; i<N1*N2; i++){
    result[i]=abs(a[i]-b[i]);
  }
}

double residual(int N1, int N2, double* u, double* f) {
        double residual_norm_sq = 0.0;
        for (int i = 0; i < N1 * N2; i++) {
        residual_norm_sq += (u[i] - f[i]) * (u[i] - f[i]);
        }
        return sqrt(residual_norm_sq);
}

void deblur(int N1, int N2, double *uIni, double *srcImg,int itertime, double h, double lambda, double delta, double epsilon, double * u){
  double * deblurU=(double*) malloc(N1*N2*sizeof(double));
  double * f=(double*) malloc(N1*N2*sizeof(double));
  double * nablaU=(double*) malloc(N1*N2*sizeof(double));
  double * uMinusf=(double*) malloc(N1*N2*sizeof(double));

  for (int i=0;i<N1*N2;i++) {u[i]=uIni[i];}
  printf("Initial residule: %f\n", residual(N1,N2, uIni, srcImg));
  clock_t t;
  t=clock();
  for (int iter=0;iter<itertime;iter++){
    normNabla(N1,N2, u, h,nablaU);
    normMinus(N1,N2, u, srcImg, uMinusf);
    for (int i=0; i<N1*N2; i++){
      f[i]=lambda*srcImg[i]/sqrt(uMinusf[i]*uMinusf[i]+delta)*sqrt(nablaU[i]*nablaU[i]+epsilon);
    }

    for (int i=1; i<N1-1;i++){
      for (int j=1; j<N2-1;j++){
        deblurU[i*N2+j]=(h*h*f[i*N2+j]+u[(i-1)*N2+j]+u[i*N2+j-1]+u[(i+1)*N2+j]+u[i*N2+j+1])/(4+lambda*sqrt(nablaU[i*N2+j]*nablaU[i*N2+j]+epsilon)*h*h/sqrt(uMinusf[i*N2+j]*uMinusf[i*N2+j]+delta));
        if (deblurU[i*N2+j]>255){deblurU[i*N2+j]=255;}
        if (deblurU[i*N2+j]<0){deblurU[i*N2+j]=0;}
      }
    }
    for (int i=0;i<N1*N2;i++) {u[i]=deblurU[i];}
    if (iter%400==0){
      printf("%d %f\n", iter, residual(N1,N2, u, srcImg));
    }
  }
  t=clock()-t;
  //printf("Final residule: %f\n", residual(N1,N2, u, srcImg));
  printf("time= %f seconds\n", ((float)t)/CLOCKS_PER_SEC);
  free(deblurU);
  free(f);
  free(nablaU);
  free(uMinusf);
}

void deblurGPU(int N1, int N2, double *uIni, double *srcImg, int itertime, double h, double lambda, double delta, double epsilon, double * dataNow)
{
  deconvolve(N1, N2, uIni,srcImg, itertime, h, lambda, delta, epsilon, dataNow);
  //printf("residule: %f\n", residual(N1,N2, dataNow, srcImg));
}
