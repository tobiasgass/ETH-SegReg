// Copyright (c) 2006-2009 Wataru Kasai
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENCE file.

#include <stdio.h>
#include <stdlib.h>

typedef double **** Double4D;
typedef double  *** Double3D;
typedef double   ** Double2D;
typedef double    * Double1D;
typedef float   *** Float3D;
typedef float    ** Float2D;
typedef float     * Float1D;
typedef int     *** Int3D;
typedef int      ** Int2D;
typedef int       * Int1D;

extern void nrerror(char *error_text);
extern Int1D AllocInt_1D(int n);
extern Int2D AllocInt_2D(int nr, int nc);
extern Float1D AllocFloat_1D(int n);
extern Float2D AllocFloat_2D(int nr, int nc);
extern Double1D AllocDouble_1D(int n);
extern Double2D AllocDouble_2D(int nr, int nc);
extern Double3D  AllocDouble_3D(int nh, int nr, int nc);
extern Double4D  AllocDouble_4D(int t, int nh, int nr, int nc);
extern void FreeFloat_1D(Float1D v);
extern void FreeInt_1D(Int1D v);
extern void FreeDouble_1D(Double1D v);
extern void FreeFloat_2D(Float2D m, int nr, int nc);
extern void FreeDouble_2D(Double2D m,int nr, int nc);
extern void FreeInt_2D(Int2D m, int nr, int nc);
extern void FreeDouble_3D(Double3D x, int nh, int nr, int nc);
extern void FreeDouble_4D(Double4D x, int t,int nh, int nr, int nc);

