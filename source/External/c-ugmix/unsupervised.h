/////////////////////////////////////////////////////////////////////
//                                                                 //
// unsupervised.h                                                  //
//                                                                 //
//Reference: "Unsupervised learning of finite mixture models"      //
//            IEEE Trans. Pattern Analysis and Machine Intelligence//
//            vol. 24, no. 3, pp. 381-396, March 2002.             //
//                                                                 //
//C++ ver. of source code is distributed by W.Kasai                // 
//                     with the author's permission --  2010.2     //
/////////////////////////////////////////////////////////////////////

#define _USE_MATH_DEFINES

#include    <stdlib.h>
#include    <math.h>
#include    "newmat.h"
#include    "newmatap.h"
#include    "newmatrm.h"
#include    "precisio.h"
#include    "k_mean.h"
#include    "include.h"
#define THRESHOLD 1.0e-5
#define UNDER_PROB 1.0e-300
#define L_MAX 1.0e100
#ifdef use_namespace
using namespace NEWMAT;
#endif
class unsupervised{
 private:
  bool FLAG;
  int Dim;
  int best_k_nz;
  Double1D best_alpha;
  ColumnVector* mu;
  Matrix* sigma;
  
  void init();
  double max(double p,double q);
  double gaussian(const ColumnVector& ob, 
		  const ColumnVector& mix_mu, const Matrix& mix_sigma);
 public:
  unsupervised();
  ~unsupervised();

  /*display the model parameters*/
  void display();
  /*get the optimized number of components*/
  int get_best_k();
  /*calculate the probability from a k-th component*/
  double gaussian(int k, const ColumnVector& ob);
  /*estimation: k_max->upper limit of the number of components*/
  int estimate(int k_max, const Matrix& obs);
  /*calculate he likelihood of unseen data from the model*/
  double likelihood(const ColumnVector& ob);
  /*memory release*/
  void CleanUp();
};
  


