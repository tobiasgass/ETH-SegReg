#include "unsupervised.h"
#include <iostream>
#include <cmath>
#ifdef use_namespaces
using namespace NEWMAT;
#endif
void unsupervised::init(){
	FLAG=false;
	best_k_nz=0;
	Dim=0;
	best_alpha = NULL;
	mu = NULL;
	sigma = NULL;
}

unsupervised::unsupervised(){
	init();
}

unsupervised::~unsupervised(){
	if(FLAG) CleanUp();
}

void unsupervised::CleanUp(){
	FLAG=false;
	Dim=0;
	FreeDouble_1D(best_alpha);
	for(int k=0;k<best_k_nz;k++){
		mu[k].CleanUp();
		sigma[k].CleanUp();
	}
	delete [] mu;
	delete [] sigma;
	best_k_nz=0;
}

void unsupervised::display(){
	if(FLAG==true){
		fprintf(stderr, "Dim:%d  Num-of-components:%d\n", Dim, best_k_nz);
		for(int i=0;i<best_k_nz;i++){
			fprintf(stderr, 
                    "For %d-th component, \nmixture coefficient\n%lf\naverage\n",i, best_alpha[i]);
			for(int j=0;j<Dim;j++) fprintf(stderr, "%lf ", mu[i].element(j));
			fprintf(stderr, "\nvariance\n");
			for(int j=0;j<Dim;j++){
				for(int k=0;k<Dim;k++) fprintf(stderr, "%lf ", sigma[i].element(j,k));
				fprintf(stderr, "\n");
			}
		}
	}
	else fprintf(stderr, "estimation is not completed!\n");
}

int unsupervised::get_best_k(){
	if(FLAG==true) return best_k_nz;
	else return 0;
}

double unsupervised::max(double p, double q){
	if(p>q) return p;
	else return q;
}

double unsupervised::gaussian(int k, const ColumnVector& ob){
	ColumnVector tmp(Dim);
	tmp = ob-mu[k];
	double x = (tmp.t() * sigma[k].i() * tmp).AsScalar();
	if( x<0.0 || std::isnan(x) ) x = 0.0; //for avoiding the failure from calculation error of newmat library.
    
	tmp.CleanUp();
	return (exp(x*(-0.5)) / (pow(2.0*M_PI,ob.Nrows()*0.5)*pow(sigma[k].Determinant(),0.5)));
}

double unsupervised::gaussian(const ColumnVector& ob, const ColumnVector& mix_mu, const Matrix& mix_sigma){
	ColumnVector tmp(Dim);
	tmp = ob-mix_mu;
	double x = (tmp.t() * mix_sigma.i() * tmp).AsScalar();
	if(x<0.0) x = 0.0; //for avoiding the failure from calculation error of newmat library.
    
	tmp.CleanUp();
	return (exp(x*(-0.5)) / (pow(2.0*M_PI,ob.Nrows()*0.5)*pow(mix_sigma.Determinant(),0.5)));
}

double unsupervised::likelihood(const ColumnVector& ob){
	double tmp=0.0;
	if(!FLAG) nrerror("estimation is not done!\n");
	for(int i=0;i<best_k_nz;i++) tmp += best_alpha[i]*gaussian(i,ob);
	return tmp;
}

int unsupervised::estimate(int k_max, const Matrix& obs){
	if(FLAG) return 0;
	Dim = obs.Nrows();
	best_k_nz = k_max;
	int n=obs.Ncols();
	int k_min=1;
	int k_nz=k_max;
	int t=0;
	int tmp_int = 0;
	int N = Dim+Dim*(Dim+1)/2;
	double L_min = L_MAX;
	double tmp = 0.0;
	double tmp2 = 0.0;
	double criterion = 0.0;
	double previous = 0.0;
	Double2D u = AllocDouble_2D(n, k_max);
	Double2D w = AllocDouble_2D(n, k_max);
	Double1D alpha = AllocDouble_1D(k_max);
	best_alpha = AllocDouble_1D(k_max);
	bool flag = false;
	bool first_flag = true;
	mu = new ColumnVector [k_max];
	sigma = new Matrix [k_max];
	for(int i=0;i<k_max;i++){
		mu[i].ReSize(Dim);
		sigma[i].ReSize(Dim,Dim);
	}
 
    
	/*Initialization*/
	Double2D tmp4 = AllocDouble_2D(Dim,n);
	Double2D tmp_vectors = AllocDouble_2D(n,Dim);

	Int1D tmp5 = AllocInt_1D(n);
	for(int j=0;j<n;j++){  
		for(int i=0;i<Dim;i++){
			tmp4[i][j] = obs.element(i,j);
            tmp_vectors[j][i]=0.0;
		}
		tmp5[j] = 0;
	}

    //initial estimate. reestimate in case a cluster is empty, and reduce maximum number of clusters
    k_Mean(k_max,tmp4,tmp5,Dim,n);
    
	int r,roop;
	/*Initialization of average and variance*/
	for(int j=0;j<k_max;j++){
		r=0;
		mu[j] = 0.0;
		sigma[j] = 0.0;
	
		for(int k=0;k<n;k++){
			if(tmp5[k] == j){
				for(int l=0;l<Dim;l++){
					mu[j].element(l) += tmp4[l][k];
					tmp_vectors[k][l] = tmp4[l][k];
				}
				r++;
			}
		}
		/*The routine for avoiding the non-positive definite matrix, in case of data<dimension,*/
		roop = 0;
		while(r <= Dim){
			r++;
			for(int check=1; roop<n || check; roop++){
				if(tmp5[roop]!=j){
					for(int l=0;l<Dim;l++){
						mu[j].element(l) += tmp4[l][roop];
						tmp_vectors[r][l] = tmp4[l][roop];
						check = 0;
					}
				}
			}
		}
	
		best_alpha[j] = alpha[j] = (double) r / n;
        //std::cout<<"j "<<j<<" "<<alpha[j]<<std::endl;
		if (r) mu[j] /= r;
		for(int k=0;k<Dim;k++){
			for(int l=0;l<Dim;l++){
				for(int m=0;m<r;m++) 
                    sigma[j].element(k,l) += (tmp_vectors[m][k] - mu[j].element(k)) * 	(tmp_vectors[m][l] - mu[j].element(l));
				if (r) sigma[j].element(k,l) /= r;
				sigma[j].element(k,l) *= 2;
			}
		}
		/*previous routine is not perfect scheme, 
          finally we check if the covariance is positive definite*/
		if(sigma[j].Determinant()<=0){
			for(int k=0;k<Dim;k++){
				for(int l=0;l<Dim;l++){
					if(k==l) sigma[j].element(k,l) = 1.0;
					else sigma[j].element(k,l) = 0.0;
				}
			}
		}
	}
	FreeDouble_2D(tmp4,Dim,n);
	FreeInt_1D(tmp5);
	FreeDouble_2D(tmp_vectors,n,Dim);
    
    
	ColumnVector *mix_mu = new ColumnVector [k_max];
	Matrix *mix_sigma = new Matrix [k_max];
	for(int m=0;m<k_max;m++){
		mix_mu[m].ReSize(Dim);
		mix_sigma[m].ReSize(Dim,Dim);
		for(int d=0;d<Dim;d++){
			mix_mu[m].element(d) = mu[m].element(d);
			for(int dd=0;dd<Dim;dd++){
				mix_sigma[m].element(d,dd) = sigma[m].element(d,dd);
			}
		}
	}
    
    
	for(int i=0;i<n;i++){
		for(int m=0;m<k_max;m++){
			u[i][m] = gaussian(m, obs.Column(i+1));
			if(u[i][m]<UNDER_PROB) u[i][m] = UNDER_PROB;
            w[i][m]=0.0;
		}
	}
#if 1 
	do{
		if(!t) criterion = L_MAX;
		first_flag = true;
	
		do{
			if(!first_flag) previous = criterion;
			else{
				previous = L_MAX;
				first_flag = false;
			}
			flag = false;
			t += 1;
	    
			for(int m=0;m<k_max;m++){
                //std::cout<<"m "<<m<<" "<<alpha[m]<<std::endl;
				if(!alpha[m]) flag = true;
				for(int i=0;i<n;i++){
					tmp = 0.0;
					for(int j=0;j<k_max;j++) {
                        tmp += alpha[j]*u[i][j];
                        //std::cout<<j<<" "<<tmp<<" "<<alpha[j]<<" "<<u[i][j]<<std::endl;
                    }
					if(tmp  ) {
                        //std::cout<<"tmp2 "<<tmp<<" "<< alpha[m] * u[i][m] / tmp<<std::endl;
                        w[i][m] = alpha[m] * u[i][m] / tmp;
                    }else{
                        w[i][m]=0;
                    }
                }
		
                if(!flag){
                    alpha[m] = 0.0;
                    for(int i=0;i<n;i++) {
                        alpha[m] += w[i][m];
                        //std::cout<<i<<" "<<alpha[m]<<" "<<w[i][m]<<std::endl;
                    }
                    alpha[m] = max(0.0, alpha[m] - (double) N / 2.0);
                    tmp = 0.0;
                    for(int j=0;j<k_max;j++){
                        tmp2 = 0.0;
                        for(int i=0;i<n;i++) tmp2 += w[i][j];
                        //std::cout<<j<<" "<<tmp2<<" "<<N<<std::endl;
                        tmp += max(0.0, tmp2 - (double) N / 2.0);
                    }
                    //std::cout<<"tmp "<<tmp<<" "<<alpha[m]<<std::endl;
                    if(tmp ) alpha[m] = alpha[m] / tmp;
                    else alpha[m] = 0.0;
                }
                
                tmp = 0.0;
                for(int l=0;l<k_max;l++) tmp += alpha[l];
                for(int l=0;l<k_max;l++) alpha[l] /= tmp;
		
                if(alpha[m] ){
                    mix_mu[m] = 0.0;
                    mix_sigma[m] = 0.0;
                    tmp = 0.0;
		    
                    for(int i=0;i<n;i++){
                        tmp += w[i][m];
                        mix_mu[m] += w[i][m] * obs.Column(i+1);
                    }
                    if(tmp  ) mix_mu[m] /= tmp;
                    for(int i=0;i<n;i++)
                        mix_sigma[m] += w[i][m] * (obs.Column(i+1)-mix_mu[m])*(obs.Column(i+1)-mix_mu[m]).t();
                    if(tmp  ) mix_sigma[m] /= tmp;
		    
                    for(int i=0;i<n;i++){
                        u[i][m] = gaussian(obs.Column(i+1), mix_mu[m], mix_sigma[m]);
                        if(u[i][m]<UNDER_PROB) u[i][m] = UNDER_PROB;
                    }
                }
                else{
                    if(!flag) k_nz--;
                }
            }
	    
            tmp = 0.0;
            flag = false;
            criterion = (double) k_nz*log((double) n/12.0) / 2.0 + (double) k_nz*(N+1)/2.0;
            for(int m=0;m<k_max;m++){
                //std::cout<<tmp<<" "<<alpha[m]<<" "<<n<<" "<<log((double) n*alpha[m] / 12.0) << std::endl;
                if(alpha[m]) tmp += log((double) n*alpha[m] / 12.0);
            }
            //std::cout<<tmp<<" "<<criterion<<std::endl;
            criterion += (double) N * tmp / 2.0;
            tmp = 0.0;
            for(int i=0;i<n;i++){
                tmp2 = 0.0;
                for(int m=0;m<k_max;m++) tmp2 += alpha[m]*u[i][m];
                if(tmp2>UNDER_PROB ) tmp += log(tmp2);
                else flag = true;
            }
            if(flag) criterion = previous; 
            else criterion -= tmp;
	    
            /*output the progression
              if(criterion==L_MAX) fprintf(stderr,"%d L_MAX(%d)\n",t, best_k_nz);
              else if(criterion<L_min) fprintf(stderr,"%d %lf(%d)\n",t, criterion, best_k_nz);
              else fprintf(stderr,"%d %lf(%d)\n",t, L_min, best_k_nz);
            */

            //fprintf(stderr,"%d %lf(%d)\n",t, criterion, k_nz);
	    
        }while((previous - criterion > THRESHOLD*fabs(previous)));
	
	
        if(criterion < L_min){
            L_min = criterion;
	    
            for(int k=0;k<best_k_nz;k++){
                mu[k].CleanUp();
                sigma[k].CleanUp();
            }
            delete [] mu;
            delete [] sigma;
            FreeDouble_1D(best_alpha);
	    
            best_alpha = AllocDouble_1D(k_nz);
            mu = new ColumnVector [k_nz];
            sigma = new Matrix [k_nz];
            for(int k=0, l=0;k<k_max;k++){
                if(alpha[k] ){
                    best_alpha[l] = alpha[k];
                    mu[l].ReSize(Dim);
                    sigma[l].ReSize(Dim,Dim);
                    mu[l] = mix_mu[k];
                    sigma[l] = mix_sigma[k];
                    l++;
                }
            }
            best_k_nz = k_nz;
        }
	
        tmp = 1e306;
        tmp2 = 0.0;
        tmp_int = 0;
        for(int m=0;m<k_max;m++){
            if(alpha[m] ){
                tmp2 += alpha[m];
                if(tmp>alpha[m]){
                    tmp = alpha[m];
                    tmp_int = m;
                }
            }
        }
        alpha[tmp_int] = 0.0;
        k_nz = 0;
        tmp2 -= tmp;
        if(tmp2  ){
            for(int m=0;m<k_max;m++){
                if(alpha[m]){
                    alpha[m] /= tmp2;
                    k_nz++;
                }
            }
        }
    }while(k_nz>=k_min);
    
    
    FreeDouble_2D(u, n, k_max);
    FreeDouble_2D(w, n, k_max);
    FreeDouble_1D(alpha);
    for(int m=0;m<k_max;m++){
        mix_mu[m].CleanUp();
        mix_sigma[m].CleanUp();
    }
    delete [] mix_mu;
    delete [] mix_sigma;
#endif    
    FLAG = true;  //estimation is done.
    return best_k_nz;
}
