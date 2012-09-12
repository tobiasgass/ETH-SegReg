// Copyright (c) 2006-2009 Wataru Kasai
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENCE file.

#include "mem.h"
#include <algorithm>

void nrerror(char *error_text){
  fprintf(stderr,"Numerical Recipes run-time error...\n");
  fprintf(stderr,"%s\n",error_text);
  fprintf(stderr,"...now exiting to system...\n");
  exit(1);
}


Int1D AllocInt_1D(int n)
{
  int *v= new int [n];
  if (!v) nrerror("allocation failure in ivector()");
  return v;
}

Int2D AllocInt_2D(int nr, int nc)
{
  int **m = new int* [nr];
  if (!m) nrerror("allocation failure 1 in imatrix()");

  for(int i=0; i<nr; i++) {
    m[i] = new int [nc];
    if (!m[i]) nrerror("allocation failure 2 in imatrix()");
  }
  return m;
}

Float1D AllocFloat_1D(int n)
{
  float *v = new float [n];
  if (!v) nrerror("allocation failure in vector()");
  return v;
}

Float2D AllocFloat_2D(int nr, int nc)
{
  float **m = new float* [nr];
  if (!m) nrerror("allocation failure 1 in matrix()");

  for(int i=0; i<nr; i++) {
    m[i] = new float[nc];
    if (!m[i]) nrerror("allocation failure 2 in matrix()");
  }
  return m;
}

Double1D AllocDouble_1D(int n)
{
  double *v = new double [n];
  if (!v) nrerror("allocation failure in dvector()");
  return v;
}

Double2D AllocDouble_2D(int nr, int nc)
{
  double **m = new double* [nr];
  if (!m) nrerror("allocation failure 1 in dmatrix()");

  for(int i=0; i<nr; i++) {
    m[i] = new double [nc];
    std::fill_n(m[i],0.0,nc);
    if (!m[i]) nrerror("allocation failure 2 in dmatrix()");
  }
  return m;
}

Double3D  AllocDouble_3D(int nh, int nr, int nc)
{
  double ***xi;
  xi = new double** [nh];
  for (int i = 0; i < nh; i++)
    xi[i] = AllocDouble_2D(nr, nc);
  return xi;
}

Double4D  AllocDouble_4D(int t, int nh, int nr, int nc)
{
  double ****x;
  x = new double*** [t];
  for (int i = 0; i < t; i++)
    x[i] = AllocDouble_3D(nh, nr, nc);
  return x;
}


void FreeFloat_1D(Float1D v){  delete[] v; }

void FreeInt_1D(Int1D v){  delete[] v; }

void FreeDouble_1D(Double1D v){  delete[] v; }

void FreeFloat_2D(Float2D m, int nr, int nc)
{
  for(int i=nr-1; i>=0; i--) delete[] m[i];
  delete[] m;
}

void FreeDouble_2D(Double2D m,int nr, int nc)
{
  for(int i=nr-1; i>=0; i--) delete[] m[i];
  delete[] m;
}

void FreeInt_2D(Int2D m, int nr, int nc)
{
  for(int i=nr-1; i>=0; i--) delete[] m[i];
  delete[] m;
}

void FreeDouble_3D(Double3D x, int nh, int nr, int nc)
{
  for (int i = 0; i < nh; i++)
    FreeDouble_2D(x[i], nr, nc);
  delete[] x;
}

void FreeDouble_4D(Double4D x, int t,int nh, int nr, int nc)
{
  for (int i = 0; i < t; i++)
    FreeDouble_3D(x[i], nh, nr, nc);
  delete[] x;
}
