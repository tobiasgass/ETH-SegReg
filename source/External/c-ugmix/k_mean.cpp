// Copyright (c) 2006-2009 Wataru Kasai
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENCE file.

#include "k_mean.h"

int random_no(int n)
{
    double r1;

    r1=(double)rand(); 
    return((int)(r1/RAND_MAX * (double)n));
}

/*--------------- Update the centers of clusters -----------------*/
int update_cluster(Cluster *cl, int cluster_no, int dim)
{
    int    flag;
    double mean;

    flag = 1;
    for(int i=0; i<cluster_no; i++){
        for(int j=0; j<dim; j++){
            if (cl[i].patterns) mean = cl[i].acc[j]/(double)cl[i].patterns;
            if(mean != cl[i].vec[j]){
                flag = 0;
                cl[i].vec[j] = mean;
            }
        }
    }
    return(flag);
}


/*--------------- Clustering by K-Mean Method ----------------------*/
bool k_Mean(int cluster_no, Double2D obs, int* obs_c, int dim, int pattern_no)
{
    int s,t;
    int check;
    int flag,no=0,n;
    double dist,dmin;
    bool success=true;

    if(cluster_no > pattern_no){
        obs_c[pattern_no] = random_no(cluster_no);
    }
    else{
        Cluster *cl = new Cluster[cluster_no];
        for(int i=0;i<cluster_no;i++){
            cl[i].vec = AllocDouble_1D(dim);
            cl[i].acc = AllocDouble_1D(dim);
        }

        for(int i=0; i<cluster_no; i++){
            cl[i].c = i;
            n = random_no(pattern_no);
            for(int j=0; j< dim; j++) cl[i].vec[j] = obs[j][n];
        }
    
        flag = 0;
        check = 0;
        while(!flag && check<CHECK){
            check++;
            for(int i=0; i<cluster_no; i++){
                cl[i].patterns = 0;
                for(int k=0; k< dim; k++) cl[i].acc[k]= 0.0;
            }
      
            for(n=0; n < pattern_no; n++){
                dmin=0.0;
                for(s=0; s<cluster_no; s++){
                    for(t=0, dist=0.0; t < dim; t++) 
                        dist += 
                            (obs[t][n]-cl[s].vec[t])*(obs[t][n]-cl[s].vec[t]);
                    if(!s || (dist < dmin)){
                        dmin = dist; 
                        no = s;
                    }
                }
                obs_c[n] = cl[no].c;
                for(int k=0; k<dim; k++) cl[no].acc[k] += obs[k][n];
                cl[no].patterns++;
            }
            flag = update_cluster(cl,cluster_no,dim);
        }
        for(int i=0;i<cluster_no;i++){
            FreeDouble_1D(cl[i].vec);
            FreeDouble_1D(cl[i].acc);
        }
        for(s=0; s<cluster_no; s++){
            //printf("%d %d %f\n",s,cl[s].patterns,cl[s].vec[0]);
            if (cl[s].patterns==0)
                success=false;
        }
        delete [] cl;
    }
    return success;
}
  
