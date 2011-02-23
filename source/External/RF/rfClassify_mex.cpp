#include <string>

#include <mex.h>

#include "forest.h"

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) 
{ 

/* 
   if(nrhs < 3) { 
      mexErrMsgTxt("Three input arguments needed! \nUSAGE: model = rfClassify(data, labels, classifierFile)"); 
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
   
   std::string filename = mxArrayToString(prhs[2]);

   long int objRef = (long int) mxGetScalar(prhs[3]);
   Forest* rf = reinterpret_cast<Forest*>(objRef);
   rf->eval(data, labels);

   //Forest rf(filename);
   //rf.eval(data, labels);


   std::vector<int> predictedLabels = rf->getPredictions();
   std::vector<float> predictedConfidences = rf->getPredictionConfidences();
   double predictionErr = rf->getPredictionErr();

   // Return to MATLAB
   plhs[0] = mxCreateNumericMatrix( nSamples, 1, mxINT32_CLASS, mxREAL );
   int* pLhs0 = (int*) mxGetPr( plhs[0] );
   for( int i=0; i < predictedLabels.size(); i++ ) {
      pLhs0[i] = predictedLabels[i];
   }
   
   plhs[1] = mxCreateNumericMatrix( nSamples, 1, mxSINGLE_CLASS, mxREAL );
   float* pLhs1 = (float*) mxGetPr( plhs[1] );
   for( int i=0; i < predictedLabels.size(); i++ ) {
      pLhs1[i] = predictedConfidences[i];
   }

   plhs[2] = mxCreateNumericMatrix( 1, 1, mxDOUBLE_CLASS, mxREAL );
   double* pLhs2 = mxGetPr( plhs[2] );
   pLhs2[0] = predictionErr;
*/

   return;
}
