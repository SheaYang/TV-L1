#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <getopt.h>
#include <cv.h>
#include <highgui.h>

#include "deblur.h"
#include "include.h"

int main( int argc, char* argv[]){

    char c;
    char *filename;
    int gpuFlag=0;
    int itertime=10000;//100000;
    double h, lambda=0.3, delta=0.005, epsilon=0.005;

    while ((c = getopt(argc, argv, ":bgk:s:d:f:p:x:y:")) != -1) {
        switch(c) {
            case 'f':
                filename = optarg;
                printf("Processing file: %s\n", filename);
                break;
            case 'g':
                printf("Use GPU Kernel\n");
                gpuFlag = 1;
                break;
        }
    }

    IplImage* img = cvLoadImage(filename, CV_LOAD_IMAGE_COLOR);
    
    int side=img->height;
    int N1=img->height+2;
    int N2=img->width+2;

    printf("Height: %d, Width: %d, N1: %d, N2: %d\n", img->height, img->width, N1,N2);
   
    IplImage* imgSplit[3];
    IplImage* dblSplit[3];
    IplImage* diffSplit[3];
    for(int i = 0; i < 3; i++){
      imgSplit[i] = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
      dblSplit[i] = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
      diffSplit[i] = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
    }
    cvSplit(img, imgSplit[0], imgSplit[1], imgSplit[2], NULL);
    
    for (int i=0;i<3;i++){
    //for (int i=2;i<3;i++){
      double* datasrc=(double*) malloc (N1*N2*sizeof(double));
      double* dataIni=(double*) malloc (N1*N2*sizeof(double));
      double* dataNow=(double*) malloc (N1*N2*sizeof(double));
      for(int h = 0 ; h < N1; h++){
        for( int w = 0; w < N2; w++){
          datasrc[h*N2+w]=0;
          dataIni[h*N2+w]=0;
          dataNow[h*N2+w]=0;
        }
      }
    for(int h = 1 ; h < N1-1; h++){
      for( int w = 1; w < N2-1; w++){
        datasrc[h*N2+w]=(double) IMG_ELEM(imgSplit[i], h-1, w-1);
        //dataIni[h*N2+w]=(double) IMG_ELEM(imgSplit[i], h-1, w-1);
      }
    }
    
    if (!gpuFlag){
        deblur(N1,N2, dataIni, datasrc, itertime, 1, lambda, delta, epsilon,dataNow);
      } else{
        deblurGPU(N1, N2, dataIni, datasrc, itertime, 1, lambda, delta, epsilon, dataNow);
    }

    for(int h = 1 ; h < N1-1; h++){
      for( int w = 1; w < N2-1; w++){
        IMG_ELEM(dblSplit[i], h-1, w-1) = dataNow[h * N2 + w];
        IMG_ELEM(diffSplit[i], h-1, w-1) = dataNow[h * N2 + w]-datasrc[h * N2 + w];
	}
    }
    free(datasrc);
    free(dataIni);
    free(dataNow);
  }
  IplImage* dbl = cvClone(img);
  IplImage* diff = cvClone(img);
  cvMerge(imgSplit[0], imgSplit[1], imgSplit[2], NULL, img);
  cvMerge(dblSplit[0], dblSplit[1], dblSplit[2], NULL, dbl);
  cvMerge(diffSplit[0], diffSplit[1], diffSplit[2], NULL, diff);

  cvSaveImage("/scratch/sy1823/HPC/final_project/mydeblur_GPU/img.png", img, 0);
  cvSaveImage("/scratch/sy1823/HPC/final_project/mydeblur_GPU/imgSplit_blue.png", imgSplit[2], 0);
  cvSaveImage("/scratch/sy1823/HPC/final_project/mydeblur_GPU/deblur.png", dbl, 0);
  //cvSaveImage("deblur_red.png", dblSplit[0], 0);
  //cvSaveImage("deblur_green.png", dblSplit[1], 0);
  cvSaveImage("/scratch/sy1823/HPC/final_project/mydeblur_GPU/deblur_blue.png", dblSplit[2], 0);
  cvSaveImage("/scratch/sy1823/HPC/final_project/mydeblur_GPU/diff.png", diff, 0);

  cvReleaseImage(&imgSplit[0]);
  cvReleaseImage(&imgSplit[1]);
  cvReleaseImage(&imgSplit[2]);
  cvReleaseImage(&dblSplit[0]);
  cvReleaseImage(&dblSplit[1]);
  cvReleaseImage(&dblSplit[2]);
  cvReleaseImage(&diffSplit[0]);
  cvReleaseImage(&diffSplit[1]);
  cvReleaseImage(&diffSplit[2]);
  cvReleaseImage(&img);
  cvReleaseImage(&dbl);
  cvReleaseImage(&diff);

  return 0;

ERROR:
    fprintf(stderr, "Usage: -f [/path/to/image]                path to the image file\n"); 
    return 1;

}
