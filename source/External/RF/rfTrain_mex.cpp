#include <string> 
#include <set> 
#include <math.h> 
#include <mex.h> 
 
#include <libxml/tree.h>
#include <libxml/parser.h> 
 
#include <boost/numeric/ublas/matrix.hpp> 
#include <boost/numeric/ublas/vector.hpp> 
 
#include "hyperparameters.h" 
#include "forest.h" 
 
 
 
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) 
{ 
/* 
   if(nrhs < 3) { 
      mexErrMsgTxt("Three input arguments needed! \nUSAGE: model = rfTrain(data, labels, parameters)"); 
   } 
    
   if( nrhs == 4 ) { 
      long int objRef = (long int) mxGetScalar(prhs[3]);
      Forest* rf = reinterpret_cast<Forest*>(objRef); 
      delete rf; 
   } 
 
   // Read the data and labels
   double* pData = (double*) mxGetData( prhs[0] );
   double* pLabels = (double*) mxGetData( prhs[1] );
   int* dataSize = (int*) mxGetDimensions( prhs[0] );
   int nSamples = dataSize[0];
   int nFeatures = dataSize[1];

   matrix<float> data( nSamples, nFeatures );
   std::vector<int> labels( nSamples );

   for ( register int cRow = 0; cRow < nSamples; cRow++ ) {
      for ( register int cCol = 0 ; cCol < nFeatures; cCol++ ) {
         data( cRow, cCol )  = (float) pData[ cRow + cCol*nSamples ];
      }
      labels[cRow] = (int) pLabels[cRow];
   }

   // Read parameters
   HyperParameters hp;
   hp.numTrees = (int) mxGetScalar(mxGetField(prhs[2], 0, "numTrees"));
   hp.maxTreeDepth = (int) mxGetScalar(mxGetField(prhs[2], 0, "maxTreeDepth"));
   hp.numRandomFeatures = (int) mxGetScalar(mxGetField(prhs[2], 0, "numRandomHyperplanes"));
   hp.numProjFeatures = (int) mxGetScalar(mxGetField(prhs[2], 0, "numProjFeatures"));
   hp.bagRatio = mxGetScalar(mxGetField(prhs[2], 0, "bagRatio"));          
   hp.useGPU = (int) mxGetScalar(mxGetField(prhs[2], 0, "useGPU"));
   hp.useSubSamplingWithReplacement = (int) mxGetScalar(mxGetField(prhs[2], 0, "selectFeatureWR"));         
   hp.useSoftVoting = (int) mxGetScalar(mxGetField(prhs[2], 0, "useSoftVoting"));
   hp.classifierFile = mxArrayToString(mxGetField(prhs[2], 0, "classifierFile"));
   

   // Only used, when using CPU implementation
   if( ~hp.useGPU ) {
      if( hp.numProjFeatures > 1 )
         hp.useRandProj = 1;
      else
         hp.useRandProj = 0;

      hp.verbose = 0;
      hp.useInfoGain = 0;
      //hp.savePath = mxArrayToString(mxGetField(prhs[2], 0, "savePath"));
      //hp.saveName = mxArrayToString(mxGetField(prhs[2], 0, "saveName"));
   }
                
   hp.saveForest = 0;
   hp.useSVM = 0;

   hp.numLabeled = nSamples;
   
   // Determine number of unique classes
   std::set<int> uniqueClasses;
   for( int i = 0; i < labels.size(); i++ ) {
      uniqueClasses.insert(labels[i]);    // only unique elements are added to a set container
   }      
   hp.numClasses = uniqueClasses.size();


   // Check input parameters
   if( hp.numRandomFeatures > nFeatures ) {
      mexErrMsgTxt("Number of random features for selection must not be greater than the number of features.");
   }
 
   //Forest rf(hp);
   Forest* rf;
   rf = new Forest(hp);


   rf->train(data, labels); 
 
   // Save forest 
   if (hp.saveForest) {
       rf->save(hp.classifierFile);      
   } 
 
   //*************************************************************************************************** 
   // Return to Matlab 
   // TODO:    model to Matlab struct 
 
   //void mxSetField(mxArray *pm, mwIndex index, const char *fieldname, mxArray *value); 
    
 
 
 
 
 
 
 
   plhs[0] = mxCreateNumericMatrix( 1, 1, mxDOUBLE_CLASS, mxREAL );
   double* pLhs0 = mxGetPr( plhs[0] );
   pLhs0[0] = rf->getTrainingErr(); 
 
   plhs[1] = mxCreateNumericMatrix( 1, 1, mxINT64_CLASS, mxREAL );
   long int* pLhs1 = (long int*) mxGetPr( plhs[1] );
   pLhs1[0] = reinterpret_cast<long int>(rf); 
*/   
   return; 
} 
 
 
 
