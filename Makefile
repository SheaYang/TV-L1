# Makefile of research program

OBJ_DIR = bin
SRC_DIR = src
INCLUDE_DIR = src
LIB_DIR = lib

PKG_CONFIG_PATH := /usr/local/lib/pkgconfig/:$(PKG_CONFIG_PATH)

MAINFILE = main.c
TARGET = deblur
SRCS = ${MAINFILE} deblur.c #expsystem.c fourier.c
OBJS := ${SRCS:.c=.o}
OBJS := ${addprefix ${OBJ_DIR}/, ${OBJS}}
CUDA_SRCS = deconvolve.cu
CUDA_OBJS := ${CUDA_SRCS:.cu=.o}
CUDA_OBJS := ${addprefix ${OBJ_DIR}/, ${CUDA_OBJS}}
INCLUDE_HEADER = ${INCLUDE_DIR}/include.h

#CVFLAGS = `pkg-config --cflags opencv` 
#CVLIBS = `pkg-config --libs opencv`
CVFLAGS = `pkg-config --cflags opencv` 
CVLIBS = `pkg-config --libs opencv`

CUDAFLAGS = -I/usr/local/cuda/include
CUDALIBS = -L/usr/local/cuda/lib -lcufft
NVCCFLAGS = -arch=sm_30 --use_fast_math

CC = gcc
NVCC = nvcc
CFLAGS =-std=gnu99 \
		-fopenmp \
	-I${INCLUDE_DIR}
DEBUG = -g -O2
CLIBFLAGS = -lm -lstdc++ 

${TARGET}:${OBJS} ${CUDA_OBJS}
	${CC} ${CFLAGS} -o $@ ${DEBUG} ${CLIBFLAGS} ${CVLIBS} ${CUDALIBS} \
${OBJS} ${CUDA_OBJS} 

${OBJ_DIR}/${MAINFILE:.c=.o}:${MAINFILE}
	${CC} $< ${CFLAGS} -c -o $@ ${DEBUG} ${CVFLAGS} ${CUDAFLAGS}


${OBJ_DIR}/%.o:${SRC_DIR}/%.c ${INCLUDE_HEADER} ${SRC_DIR}/%.h
	${CC} $<  ${CFLAGS} -c -o $@  ${DEBUG} ${CVFLAGS} ${CUDAFLAGS}

${OBJ_DIR}/%.o:${SRC_DIR}/%.cu
	${NVCC} $< -c -o $@  ${DEBUG} ${NVCCFLAGS}

clean:
	rm -f ${TARGET} ${OBJS} ${CUDA_OBJS}
