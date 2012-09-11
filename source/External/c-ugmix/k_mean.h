// Copyright (c) 2006-2009 Wataru Kasai
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENCE file.

#include    <stdio.h>
#include    <stdlib.h>
#include    <string.h>
#include    <math.h>
#include    "mem.h"

#define CHECK 1000

typedef struct clusterRec {
  int    c;
  int    patterns;
  double* vec;
  double* acc;
} Cluster;

int random_no(int n);
  
/*--------------- Update the centers of clusters -----------------*/
int update_cluster(Cluster *cl, int cluster_no,int dim);
  
  
/*--------------- Clustering by K-Mean Method ----------------------*/
bool k_Mean(int cluster_no, Double2D obs, int* obs_c, int dim, int pattern_no);
  
