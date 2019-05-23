#ifndef __DEBLUR__
#define __DEBLUR__

#include <stdio.h>
#include <math.h>

#include <cv.h>
#include <highgui.h>

void normNabla(int N1, int N2, double *u, double h, double * nablaU);
void normMinus(int N1, int N2, double * a, double * b, double * result);
void deblur(int N1, int N2, double *uIni, double *srcImg,int itertime, double h, double lambda, double delta, double epsilon, double * dataNow);
double residual(int N1, int N2, double* u, double* f);
void deblurGPU(int N1, int N2, double *uIni, double *srcImg, int itertime, double h, double lambda, double delta, double epsilon, double * dataNow);


#endif
