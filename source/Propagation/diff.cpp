diff --git a/source/Propagation/CLERCIndependentDimensions.h b/source/Propagation/CLERCIndependentDimensions.h
index 542fff2..89d21d6 100644
--- a/source/Propagation/CLERCIndependentDimensions.h
+++ b/source/Propagation/CLERCIndependentDimensions.h
@@ -48,6 +48,7 @@ protected:
     map<string,ImagePointerType> * m_segmentationList;
 
     map< int, map <int, GaussEstimatorType > > m_pairwiseInconsistencyStatistics;
+    map< int, map <int, FloatImagePointerType > > m_pairwiseLocalWeightMaps;
 
 private:
     int m_nPixels;// number of pixels/voxels
@@ -117,7 +118,7 @@ public:
         m_linearInterpol=false;
         m_haveDeformationEstimate=false;
         m_updatedDeformationCache = new  map< string, map <string, DeformationFieldPointerType> > ; 
-        m_results = std::vector<mxArray * >(D);
+        m_results = std::vector<mxArray * >(D,NULL);
         //m_updateDeformations=true;
         m_updateDeformations=false;
         m_exponent=1.0;
@@ -146,36 +147,54 @@ public:
 
         m_nEqFullCircleEnergy  = (m_wFullCircleEnergy>0.0)* internalD * m_nPixels *  m_numImages*(m_numImages-1)*(m_numImages-2); //there is one equation for each component of every pixel of every triple of images
         m_nVarFullCircleEnergy = 2*(interpolationFactor+2); //two variables per uniqe pair in the triple (2*2), plus linear interpolation for the third pair (2*2^D)
-
+        if ( m_nEqFullCircleEnergy  )
+            m_wFullCircleEnergy /=m_nEqFullCircleEnergy ;
+        
         m_nEqCircleNorm =  (m_wCircleNorm>0.0)* m_nPixels * internalD * m_numImages*(m_numImages-1)*(m_numImages-2); //again all components of all triples
         m_nVarCircleNorm = interpolationFactor+2 ; // only one/2^D variables per pair
-        LOG<<VAR(D)<<" "<<VAR(interpolationFactor)<<" "<<VAR(m_nVarCircleNorm)<<std::endl;
+        if (m_nEqCircleNorm)
+            m_wCircleNorm/=m_nEqCircleNorm;
+        
         
         m_nEqErrorInconsistency =  (m_wErrorInconsistency>0.0)* m_nPixels * internalD * m_numImages*(m_numImages-1)*(m_numImages-2); //again all components of all triples
         m_nVarErrorInconsistency = interpolationFactor+2; // only one/2^D variables per pair
+        if (m_nEqErrorInconsistency)
+            m_wErrorInconsistency/=m_nEqErrorInconsistency;
 
         m_nEqDeformationSmootheness =  (m_wDeformationSmootheness>0.0)* D* m_nPixels * internalD * m_numImages*(m_numImages-1); //every pixel in each registration has D neighbors (in one direction), and each component separately
         m_nVarDeformationSmootheness = 3; //for piecewise linear regularization, 2 for piecewise constant
+        if (m_nEqDeformationSmootheness)
+            m_wDeformationSmootheness/=m_nEqDeformationSmootheness;
 
         m_nEqErrorSmootheness =  (m_wErrorSmootheness>0.0)* D* m_nPixels * internalD * m_numImages*(m_numImages-1); //every pixel in each registration has D neighbors (in one direction), and each component separately
         m_nVarErrorSmootheness = 3; //for piecewise linear regularization, 2 for piecewise constant
+        if (m_nEqErrorSmootheness)
+            m_wErrorSmootheness/=m_nEqErrorSmootheness;
       
         m_nEqErrorNorm = (m_wErrorNorm>0.0)*  m_nPixels * internalD * m_numImages*(m_numImages-1); //every error at every location in each image pair
         m_nVarErrorNorm = 1;
+        if (m_nEqErrorNorm)
+            m_wErrorNorm/=m_nEqErrorNorm;
 
         m_nEqErrorStatistics = (m_wErrorStatistics>0.0)*  m_nPixels * internalD * m_numImages*(m_numImages-1); //every error at every location in each image pair
         m_nVarErrorStatistics = 1;
+        if (m_nEqErrorStatistics)
+            m_wErrorStatistics/=m_nEqErrorStatistics;
 
         m_nEqTransformationSimilarity =  (m_wTransformationSimilarity>0.0)*m_nPixels * internalD * m_numImages*(m_numImages-1); //same as ErrorNorm
         m_nVarTransformationSimilarity= 1;
+        if (m_nEqTransformationSimilarity)
+            m_wTransformationSimilarity/=m_nEqTransformationSimilarity;
 
         int m_nEqSUM=(m_wSum>0.0)*m_nPixels * internalD * m_numImages*(m_numImages-1);
+        if (m_nEqSUM)
+            m_wSum/=m_nEqSUM;
         int m_nVarSUM=2;
 
         m_nEqs=  m_nEqErrorStatistics+  m_nEqFullCircleEnergy + m_nEqCircleNorm+ m_nEqDeformationSmootheness + m_nEqErrorNorm+ m_nEqTransformationSimilarity + m_nEqSUM +  m_nEqErrorSmootheness + m_nEqErrorInconsistency; // total number of equations
         
-        m_estError= m_nEqFullCircleEnergy ||  m_nEqErrorNorm || m_nEqSUM||m_nEqErrorInconsistency ||  m_nEqErrorStatistics;
-        m_estDef = m_nEqFullCircleEnergy || m_nEqTransformationSimilarity ||  m_nEqDeformationSmootheness  ||  m_nEqCircleNorm || m_nEqSUM  ;
+        m_estError= m_nEqFullCircleEnergy ||  m_nEqErrorNorm || m_nEqSUM||m_nEqErrorInconsistency ;
+        m_estDef = m_nEqFullCircleEnergy || m_nEqTransformationSimilarity ||  m_nEqDeformationSmootheness  ||  m_nEqCircleNorm || m_nEqSUM  ||  m_nEqErrorStatistics;
         m_nVars= m_numImages*(m_numImages-1)*m_nPixels*internalD *(m_estError + m_estDef); // total number of free variables (error and deformation)
         
         m_nNonZeroes=  m_nEqErrorStatistics+ m_nEqErrorSmootheness*m_nVarErrorSmootheness +m_nEqFullCircleEnergy *m_nVarFullCircleEnergy + m_nEqCircleNorm * m_nVarCircleNorm + m_nEqDeformationSmootheness*m_nVarDeformationSmootheness + m_nEqErrorNorm*m_nVarErrorNorm + m_nEqTransformationSimilarity*m_nVarTransformationSimilarity + m_nEqSUM*m_nVarSUM + m_nVarErrorInconsistency*m_nEqErrorInconsistency; //maximum number of non-zeros
@@ -190,6 +209,12 @@ public:
         this->m_ROI->TransformIndexToPhysicalPoint(nullIdx,startPoint);
         (*m_downSampledDeformationCache)[(*m_imageIDList)[0]][(*m_imageIDList)[1]]->TransformPhysicalPointToIndex(startPoint,startIndex);
         m_regionOfInterest.SetIndex(startIndex);
+#ifdef SEPENGINE
+        if (this->m_ep){
+            engClose(this->m_ep);
+        }
+#endif
+
 
     }
 
@@ -218,7 +243,12 @@ public:
         double totalInconsistency = 0.0;
         int totalCount = 0;
         for (unsigned int d = 0; d< D; ++d){
-
+#ifdef SEPENGINE       
+            if (!(this->m_ep = engOpen("matlab -nodesktop -nodisplay -nosplash -nojvm"))) {
+                fprintf(stderr, "\nCan't start MATLAB engine\n");
+                exit(EXIT_FAILURE);
+            }
+#endif
             mxArray *mxX=mxCreateDoubleMatrix((mwSize)m_nNonZeroes,1,mxREAL);
             mxArray *mxY=mxCreateDoubleMatrix((mwSize)m_nNonZeroes,1,mxREAL);
             mxArray *mxV=mxCreateDoubleMatrix((mwSize)m_nNonZeroes,1,mxREAL);
@@ -260,6 +290,9 @@ public:
         
             
             computeTripletEnergies( x,  y, v,  b, c,  eq,d);
+            if (m_updateDeformations){
+                computePairwiseSimilarityWeights();
+            }
             computePairwiseEnergiesAndBounds( x,  y, v,  b, init, lb, ub, c,  eq,d);
 
 
@@ -316,12 +349,11 @@ public:
 
             
             if (1){
-                
-                engEvalString(this->m_ep, "options=optimset(optimset('lsqlin'),'Display','iter','TolFun',1e-54,'PrecondBandWidth',Inf);");//,'Algorithm','active-set' );");
+                engEvalString(this->m_ep, "options=optimset(optimset('lsqlin'),'Display','iter','TolFun',1e-54,'PrecondBandWidth',Inf,'LargeScale','on');");//,'Algorithm','active-set' );");
                 //solve using trust region method
                 TIME(engEvalString(this->m_ep, "tic;[x resnorm residual flag  output lambda] =lsqlin(A,b,[],[],[],[],lb,ub,init);toc"));
                 //solve using active set method (backslash)
-                //TIME(engEvalString(this->m_ep, "tic;[x resnorm residual flag output lambda] =lsqlin(A,b,[],[],[],[],[],[],init);toc"));
+                //TIME(engEvalString(this->m_ep, "tic;[x resnorm residual flag output lambda] =lsqlin(A,b,[],[],[],[],[],[],[]);toc"));
                 printf("%s", buffer+2);
                 engEvalString(this->m_ep, " resnorm");
                 printf("%s", buffer+2);
@@ -335,13 +367,18 @@ public:
             if ((m_results[d] = engGetVariable(this->m_ep,"x")) == NULL)
                 printf("something went wrong when getting the variable.\n Result is probably wrong. \n");
             engEvalString(this->m_ep,"clear A b init lb ub x;" );
+
+            
+#ifdef SEPENGINE
+            engClose(this->m_ep);
+#endif
         }//dimensions
 
     }
     virtual void solve(){}
 
     virtual void storeResult(string directory){
-        std::vector<double> result(m_nVars);
+        //std::vector<double> result(m_nVars);
         std::vector<double*> rData(D);
         for (int d= 0; d<D ; ++d){
             rData[d]=mxGetPr(this->m_results[d]);
@@ -362,9 +399,9 @@ public:
             for (int t=0;t<m_numImages;++t){
                 if (s!=t){
                     //slightly(!!!) stupid creation of empty image
-                    DeformationFieldPointerType estimatedError=TransfUtils<ImageType>::createEmpty(this->m_ROI);//ImageUtils<DeformationFieldType>::createEmpty((*m_downSampledDeformationCache)[(*m_imageIDList)[s]][(*m_imageIDList)[t]]);
+                    DeformationFieldPointerType estimatedError=TransfUtils<ImageType>::createEmpty(this->m_ROI);
                     DeformationFieldIterator itErr(estimatedError,estimatedError->GetLargestPossibleRegion());
-                    DeformationFieldPointerType estimatedDeform=TransfUtils<ImageType>::createEmpty(this->m_ROI);//ImageUtils<DeformationFieldType>::createEmpty((*m_downSampledDeformationCache)[(*m_imageIDList)[s]][(*m_imageIDList)[t]]);
+                    DeformationFieldPointerType estimatedDeform=TransfUtils<ImageType>::createEmpty(this->m_ROI);
                     DeformationFieldIterator itDef(estimatedDeform,estimatedDeform->GetLargestPossibleRegion());
                     itErr.GoToBegin();
                     itDef.GoToBegin();
@@ -386,15 +423,16 @@ public:
                         originalDeformation=origIt.Get();
                         for (unsigned int d=0;d<D;++d,++p){
                             // minus 1 to correct for matlab indexing
-                            if (m_estError)
+                            if (m_estError){
                                 estimatedError[d]=rData[d][edgeNumError(s,t,idx,d)-1];
+                            }
                             if (m_estDef)
                                 estimatedDeformation[d]=rData[d][edgeNumDeformation(s,t,idx,d)-1];
-                            else
-                                estimatedDeformation[d]= originalDeformation[d]-estimatedError[d];
-                            if (!m_estError && ! m_estError){
+                            
+                            if (!m_estError && m_estDef){
                                 estimatedError[d]= originalDeformation[d]-estimatedDeformation[d];
-                            }
+                            } else if (!m_estDef && m_estError)
+                                estimatedDeformation[d]= originalDeformation[d]-estimatedError[d];
                         }
                         itErr.Set(estimatedError);
                         itDef.Set(estimatedDeformation);
@@ -482,6 +520,7 @@ public:
         inc=computeInconsistency(m_trueDeformations,mask);
         //inc=computeInconsistency(m_deformationCache,mask);
 
+     
       
         
     }
@@ -652,7 +691,7 @@ protected:
 
     void computeTripletEnergies(double * x, double * y, double * v, double * b, long int &c,long int & eq, unsigned int d){
         double maxAbsDisplacement=0.0;
-        
+        m_pairwiseInconsistencyStatistics=map<int, map <int,  GaussEstimatorType > >();
         //0=min,1=mean,2=max,3=median,-1=off,4=gauss;
         int accumulate=4;
         double manualResidual=0.0;
@@ -668,8 +707,10 @@ protected:
                     
                     if (m_pairwiseInconsistencyStatistics[s].find(t)==m_pairwiseInconsistencyStatistics[s].end())
                         m_pairwiseInconsistencyStatistics[s][t]=GaussEstimatorType();
-
                     
+                    m_pairwiseInconsistencyStatistics[s][t].addImage(TransfUtils<ImageType>::getComponent(dSourceTarget,d));
+
+                    //#define ORACLE         
                     //triplet energies
                     for (int i=0;i<m_numImages;++i){ 
                         if (t!=i && i!=s){
@@ -679,15 +720,20 @@ protected:
                             if (true && m_haveDeformationEstimate && (*m_updatedDeformationCache)[(*m_imageIDList)[intermediate]][(*m_imageIDList)[target]].IsNotNull()){
                                 dIntermediateTarget=(*m_updatedDeformationCache)[(*m_imageIDList)[intermediate]][(*m_imageIDList)[target]];
                             }
+#ifdef ORACLE
+                            dIntermediateTarget=(*m_trueDeformations)[(*m_imageIDList)[intermediate]][(*m_imageIDList)[target]];
+#endif
                             DeformationFieldPointerType dSourceIntermediate = (*m_downSampledDeformationCache)[(*m_imageIDList)[source]][(*m_imageIDList)[intermediate]];
 
                             
                             //compute indirect deform
                             DeformationFieldPointerType indirectDeform = TransfUtils<ImageType>::composeDeformations(dIntermediateTarget,dSourceIntermediate);
+                            //DeformationFieldPointerType indirectDeform = TransfUtils<ImageType>::composeDeformations(dSourceIntermediate,dIntermediateTarget);
                             //compute difference of direct and indirect deform
                             DeformationFieldPointerType difference = TransfUtils<ImageType>::subtract(indirectDeform,dSourceTarget);
                             
                             FloatImagePointerType directionalDifference = TransfUtils<ImageType>::getComponent(difference,d);
+                            FloatImagePointerType directionalDeform = TransfUtils<ImageType>::getComponent(indirectDeform,d);
                             
                             //check if all accumulators exist
                             if (m_pairwiseInconsistencyStatistics.find(i)==m_pairwiseInconsistencyStatistics.end())
@@ -695,8 +741,9 @@ protected:
                             if (m_pairwiseInconsistencyStatistics[i].find(t)==m_pairwiseInconsistencyStatistics[i].end())
                                 m_pairwiseInconsistencyStatistics[i][t]=GaussEstimatorType();
 
-                            m_pairwiseInconsistencyStatistics[s][t].addImage(directionalDifference);
-                            m_pairwiseInconsistencyStatistics[i][t].addImage(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(directionalDifference,-1));
+                            //m_pairwiseInconsistencyStatistics[s][t].addImage(directionalDifference);
+                            m_pairwiseInconsistencyStatistics[s][t].addImage(directionalDeform);
+                            //m_pairwiseInconsistencyStatistics[i][t].addImage(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(directionalDifference,-1));
 
 
                             //compute norm
@@ -744,26 +791,7 @@ protected:
                                     inside=getNearestNeighbors(dIntermediateTarget,ptIntermediate,ptIntermediateNeighbors);
                                 }
 
-                                //this can be used to index the circle constraint equation with the true deform if known. cheating!
-                                //or with an estimation from the previous iteration
-                                DeformationType trueDef;
-                                bool newEstimate=false;
-                                PointType truePtIntermediate;
-                                //#define ORACLE
-                                    
-#ifdef ORACLE
-                                trueDef =(*m_trueDeformations)[(*m_imageIDList)[intermediate]][(*m_imageIDList)[target]]->GetPixel(targetIndex);
-                                newEstimate=true;
-
-#endif
-                                if (newEstimate){
-                                    truePtIntermediate=ptTarget + trueDef;
-                                    inside= inside && getLinearNeighbors(dIntermediateTarget,truePtIntermediate,ptIntermediateNeighborsCircle);
-                                    ++trueIt;
-                                }else{
-                                    ptIntermediateNeighborsCircle=ptIntermediateNeighbors;
-                                }
-
+ 
                                         
                                 this->m_ROI->TransformPhysicalPointToIndex(ptTarget,roiTargetIndex);
                                 LOGV(9)<<VAR(targetIndex)<<" "<<VAR(roiTargetIndex)<<endl;
@@ -773,7 +801,7 @@ protected:
                                     double val=1.0;
                                     bool segVal=1.0;
 
-                                    val*=getIndexBasedWeight(roiTargetIndex,roiSize);
+                                    //val*=getIndexBasedWeight(roiTargetIndex,roiSize);
 
                                     //multiply val by segConsistencyWeight if deformation starts from atlas segmentation
                                     if (haveSeg){
@@ -851,12 +879,12 @@ protected:
 #endif
                                         
                                         double defSum=0.0;
-                                        for (int i=0;i<ptIntermediateNeighborsCircle.size();++i){
+                                        for (int i=0;i<ptIntermediateNeighbors.size();++i){
                                             x[c]=eq;
-                                            y[c]=edgeNumDeformation(source,intermediate,ptIntermediateNeighborsCircle[i].first,d); // this is an APPROXIMIATION!!! might be bad :o
-                                            v[c++]=ptIntermediateNeighborsCircle[i].second*val* m_wCircleNorm;
-                                            LOGV(8)<<VAR(roiTargetIndex)<<" "<<VAR(truePtIntermediate)<<" "<<VAR(i)<<" "<<VAR(ptIntermediateNeighborsCircle[i].first)<<" "<<VAR(ptIntermediateNeighborsCircle[i].second)<<endl;
-                                            defSum+=ptIntermediateNeighborsCircle[i].second*dSourceIntermediate->GetPixel(ptIntermediateNeighborsCircle[i].first)[d];
+                                            y[c]=edgeNumDeformation(source,intermediate,ptIntermediateNeighbors[i].first,d); // this is an APPROXIMIATION!!! might be bad :o
+                                            v[c++]=ptIntermediateNeighbors[i].second*val* m_wCircleNorm;
+                                            LOGV(8)<<VAR(roiTargetIndex)<<" "<<VAR(i)<<" "<<VAR(ptIntermediateNeighbors[i].first)<<" "<<VAR(ptIntermediateNeighbors[i].second)<<endl;
+                                            defSum+=ptIntermediateNeighbors[i].second*dSourceIntermediate->GetPixel(ptIntermediateNeighbors[i].first)[d];
                                         }
                                         //minus direct
                                         x[c]=eq;
@@ -879,16 +907,16 @@ protected:
                                         y[c]=edgeNumError(intermediate,target,roiTargetIndex,d);
                                         v[c++]=val* m_wErrorInconsistency;
                                         
-                                        for (int i=0;i<ptIntermediateNeighborsCircle.size();++i){
+                                        for (int i=0;i<ptIntermediateNeighbors.size();++i){
                                             x[c]=eq;
-                                            y[c]=edgeNumError(source,intermediate,ptIntermediateNeighborsCircle[i].first,d); // this is an APPROXIMIATION!!! might be bad :o
-                                            v[c++]=ptIntermediateNeighborsCircle[i].second*val* m_wErrorInconsistency;
+                                            y[c]=edgeNumError(source,intermediate,ptIntermediateNeighbors[i].first,d); // this is an APPROXIMIATION!!! might be bad :o
+                                            v[c++]=ptIntermediateNeighbors[i].second*val* m_wErrorInconsistency;
                                         }
                                         //minus direct
                                         x[c]=eq;
                                         y[c]=edgeNumError(source,target,roiTargetIndex,d);
                                         v[c++]= - val* m_wErrorInconsistency;
-                                        b[eq-1]=disp;
+                                        b[eq-1]=disp* m_wErrorInconsistency;
                                         ++eq;
                                         
                                     }
@@ -929,70 +957,26 @@ protected:
                       
 
                       
-                 
                     FloatImagePointerType lncc;
                     FloatImageIterator lnccIt;
-                    if (m_sigma>0.0 && m_wErrorNorm>0.0){
-                        ostringstream oss;
-                        oss<<"lncc-"<<sourceID<<"-TO-"<<targetID;
-                        if (D==2)
-                            oss<<".png";
-                        else
-                            oss<<".nii";
-
-#if 0 //#ifdef ORACLE
-                        DeformationFieldPointerType diff=TransfUtils<ImageType>::subtract(
-                                                                                          (*this->m_downSampledDeformationCache)[sourceID][targetID],
-                                                                                          (*m_trueDeformations)[sourceID][targetID]
-                                                                                          );
-                                                                                              
-                        lncc=TransfUtils<ImageType>::computeLocalDeformationNormWeights(diff,m_exponent);
-#else
-                        DeformationFieldPointerType def = (*this->m_deformationCache)[sourceID][targetID];
-                        ImagePointerType warpedImage= TransfUtils<ImageType>::warpImage((ConstImagePointerType)(*m_imageList)[sourceID],def);
-                        //compute lncc
-                        lncc= Metrics<ImageType,FloatImageType>::efficientLNCC(warpedImage,(*m_imageList)[targetID],m_sigma,m_exponent);
-                        //lncc= Metrics<ImageType,FloatImageType>::LSADNorm(warpedImage,(*m_imageList)[targetID],m_sigma,m_exponent);
-                        //lncc= Metrics<ImageType,FloatImageType>::LSSDNorm(warpedImage,(*m_imageList)[targetID],m_sigma,m_exponent);
-                        //lncc= Metrics<ImageType,FloatImageType>::LSSD(warpedImage,(*m_imageList)[targetID],m_sigma);
-                        //lncc= Metrics<ImageType,FloatImageType>::localMetricAutocorrelation(warpedImage,(*m_imageList)[targetID],m_sigma,2,"lssd");
-                        //FloatImagePointerType laplacian=FilterUtils<ImageType,FloatImageType>::laplacian((*m_imageList)[targetID],m_sigma);
-                        if (0){
-                            FloatImagePointerType laplacian=FilterUtils<ImageType,FloatImageType>::normalizedLaplacianWeighting((*m_imageList)[targetID],m_sigma,m_exponent);
-                            ostringstream oss2;
-                            oss2<<"laplacian-"<<sourceID<<"-TO-"<<targetID<<".nii";
-                                
-                            LOGI(6,ImageUtils<ImageType>::writeImage(oss2.str(),FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(laplacian,255))));
-                                
-                            lncc=ImageUtils<FloatImageType>::multiplyImageOutOfPlace(lncc,laplacian);
-                        }
-                        LOGI(6,ImageUtils<ImageType>::writeImage(oss.str(),FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(lncc,255))));
-                        //resample lncc result
-                        if (1){
-                            //lncc = FilterUtils<FloatImageType>::gaussian(lncc,8);
-                            //lncc = FilterUtils<FloatImageType>::LinearResample(lncc, FilterUtils<ImageType,FloatImageType>::cast(this->m_ROI),false);
-                            lncc = FilterUtils<FloatImageType>::LinearResample(lncc, FilterUtils<ImageType,FloatImageType>::cast(this->m_ROI),true);
-                        }else{
-                            lncc = FilterUtils<FloatImageType>::minimumResample(lncc,FilterUtils<ImageType,FloatImageType>::cast(this->m_ROI), m_sigmaD);
-                        }
-#endif
-
-                          
-                        oss<<"-resampled";
-                        if (D==2){
-                            oss<<".png";
-                        }else
-                            oss<<".nii";
-                        LOGI(6,ImageUtils<ImageType>::writeImage(oss.str(),FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(lncc,255))));
-
+                    if (m_sigma>0.0 && (m_wErrorNorm>0.0 || m_wTransformationSimilarity)){
+                        lncc=m_pairwiseLocalWeightMaps[s][t];
                         lnccIt=FloatImageIterator(lncc,lncc->GetLargestPossibleRegion());
                         lnccIt.GoToBegin();
                     }
 
                     DeformationFieldIterator previousIt;
-                    if (true && m_haveDeformationEstimate && (*m_updatedDeformationCache)[sourceID][targetID].IsNotNull()){
+                    double priorWeight=1.0;
+                    if (true && ! m_updateDeformations && m_haveDeformationEstimate && (*m_updatedDeformationCache)[sourceID][targetID].IsNotNull()){
                         DeformationFieldPointerType estDef=(*m_updatedDeformationCache)[sourceID][targetID];
-                        previousIt=DeformationFieldIterator(estDef,m_regionOfInterest);
+                        DeformationFieldPointerType diff=TransfUtils<ImageType>::subtract(estDef,(*m_downSampledDeformationCache)[sourceID][targetID]);
+                        double defNorm=TransfUtils<ImageType>::computeDeformationNorm(diff);
+                        if (defNorm!=0.0){
+                            priorWeight=1.0/defNorm;
+                        }else
+                            priorWeight= 10;
+                        
+                        previousIt=DeformationFieldIterator(diff,m_regionOfInterest);
                         previousIt.GoToBegin();
                     }
                         
@@ -1014,7 +998,9 @@ protected:
                         double trueError  = (*this->m_downSampledDeformationCache)[sourceID][targetID]->GetPixel(idx)[d]-(*m_trueDeformations)[sourceID][targetID]->GetPixel(idx)[d];
 
                         int edgeNumDef=edgeNumDeformation(source,target,idx,d);
-                        int edgeNumErr=edgeNumError(source,target,idx,d);
+                        int edgeNumErr;
+                        if (m_estError)
+                            edgeNumErr=edgeNumError(source,target,idx,d);
 
                         //intensity based weight
                         double weight=1.0;
@@ -1028,9 +1014,24 @@ protected:
                                 
                         //weight based on previous estimate
                         double weight2=1.0;
-                        if (false && m_haveDeformationEstimate && m_sigmaD>0.0){
-                            weight2 = exp ( - (localDef-previousIt.Get()).GetSquaredNorm() / m_sigmaD );
-                            ++previousIt;
+                        double expectedError=0.0;
+                        if (m_haveDeformationEstimate){
+                            priorWeight=1.0;
+                            if (false ){
+                                expectedError = previousIt.Get()[d];
+                                ++previousIt;
+
+                                //double diffNorm=fabs(previousIt.Get()[d]);
+                                double diffNorm=previousIt.Get().GetNorm();
+                                if (diffNorm!=0.0){
+                                    priorWeight=1.0/diffNorm;
+                                }else
+                                    priorWeight=100;
+                                
+                                //LOGV(4)<<VAR(diffNorm)<<" "<<VAR(priorWeight)<<endl;
+                                priorWeight=min(max(priorWeight,0.001),10.0);
+                                
+                            }
                         }
                                 
                         //set w_delta
@@ -1047,15 +1048,15 @@ protected:
 
                         //set w_delta
                         //set eqn for soft constraining the error to be small
-                        double meanInconsistency=-statisticsEstimatorSourceTarget.getMean()->GetPixel(idx);
-                        double varInconsistency=(fabs(statisticsEstimatorSourceTarget.getVariance()->GetPixel(idx)));
+                        double meanInconsistency=statisticsEstimatorSourceTarget.getMean()->GetPixel(idx);
+                        double varInconsistency=sqrt((fabs(statisticsEstimatorSourceTarget.getVariance()->GetPixel(idx))));
                         if (varInconsistency == 0.0){
                             varInconsistency = 1e-5;
                         }
                         if (m_wErrorStatistics>0.0){
-                            weight2=1.0;//1.0/(varInconsistency);
+                            weight2 = 1.0/(varInconsistency);
                             x[c]    = eq;
-                            y[c]    = edgeNumErr;
+                            y[c]    = edgeNumDef;
                             v[c++]  = 1.0*m_wErrorStatistics*weight2;
                             //b[eq-1] = m_wErrorStatistics*weight2*trueError;
                             b[eq-1] = m_wErrorStatistics*weight2*meanInconsistency;
@@ -1066,13 +1067,14 @@ protected:
                         //set w_T
                         //set eqn for soft constraining the estimated true deformation to be similar to the original deformation
                         if (m_wTransformationSimilarity>0.0){
+                            //weight=1.0/sqrt(fabs(meanInconsistency));
                             x[c]    = eq;
                             y[c]    = edgeNumDef;
-                            v[c++]  = 1.0*m_wTransformationSimilarity *weight;
+                            v[c++]  = 1.0*m_wTransformationSimilarity *weight*priorWeight;
                             //b[eq-1] = (localDef[d]-trueError)*m_wTransformationSimilarity;// * weight;
-                            b[eq-1] = localDef[d]*m_wTransformationSimilarity * weight;
+                            b[eq-1] = localDef[d]*m_wTransformationSimilarity * weight*priorWeight;
                             ++eq;
-                            LOGV(8)<<VAR(source)<<" "<<VAR(target)<<" "<<VAR(idx)<<" "<<VAR(d)<<" "<<VAR(edgeNumErr)<<endl;
+                            LOGV(8)<<VAR(source)<<" "<<VAR(target)<<" "<<VAR(idx)<<" "<<VAR(d)<<" "<<VAR(edgeNumErr)<<" "<<VAR(priorWeight)<<endl;
                         }
                             
                             
@@ -1105,7 +1107,7 @@ protected:
                                 IndexType neighborIndexLeft=idx+off2;
                                 if (defSourceInterm->GetLargestPossibleRegion().IsInside(neighborIndexRight) &&defSourceInterm->GetLargestPossibleRegion().IsInside(neighborIndexLeft) ){
                                     x[c]=eq;
-                                    y[c]=edgeNumErr;
+                                    y[c]=edgeNumDef;
                                     v[c++]=-2*smoothenessWeight;
                                     x[c]=eq;
                                     y[c]=edgeNumDeformation(source,target,neighborIndexRight,d);
@@ -1177,7 +1179,7 @@ protected:
                             ub[edgeNumDef-1]   =  defSourceInterm->GetOrigin()[d]+extent-pt[d] +0.0001;
                             LOGV(4)<<VAR(pt)<<" "<<VAR(defSourceInterm->GetOrigin()[d]-pt[d])<<" "<<VAR( defSourceInterm->GetOrigin()[d]+extent-pt[d]  )<<endl;
                             //init[edgeNumDef-1] =  0;
-                            init[edgeNumDef-1] =  localDef[d] ;
+                            init[edgeNumDef-1] =  localDef[d] -expectedError;
                             //init[edgeNumDef-1] =  (localDef[d]-trueError) ;
                             
                         }
@@ -1188,7 +1190,84 @@ protected:
             }//target
         }//source
     }//computePairwiseEnergiesAndBounds
-  
+public:
+    void computePairwiseSimilarityWeights()
+    {
+        LOG<<"Computing similarity based local weights" <<endl;
+        for (int s = 0;s<m_numImages;++s){                            
+            int source=s;
+            m_pairwiseLocalWeightMaps[s]=map<int,FloatImagePointerType>();
+            for (int t=0;t<m_numImages;++t){
+                if (t!=s){
+                    int target=t;
+                    string sourceID=(*this->m_imageIDList)[source];
+                    string targetID = (*this->m_imageIDList)[target];
+                                                       
+                 
+                    FloatImagePointerType lncc;
+                    if (m_sigma>0.0 && (m_wErrorNorm>0.0 || m_wTransformationSimilarity)){
+                        ostringstream oss;
+                        oss<<"lncc-"<<sourceID<<"-TO-"<<targetID;
+                        if (D==2)
+                            oss<<".png";
+                        else
+                            oss<<".nii";
+
+#if 0 //#ifdef ORACLE
+                        DeformationFieldPointerType diff=TransfUtils<ImageType>::subtract(
+                                                                                          (*this->m_downSampledDeformationCache)[sourceID][targetID],
+                                                                                          (*m_trueDeformations)[sourceID][targetID]
+                                                                                          );
+                                                                                              
+                        lncc=TransfUtils<ImageType>::computeLocalDeformationNormWeights(diff,m_exponent);
+#else
+                        DeformationFieldPointerType def = (*this->m_deformationCache)[sourceID][targetID];
+                        ImagePointerType warpedImage= TransfUtils<ImageType>::warpImage((ConstImagePointerType)(*m_imageList)[sourceID],def);
+                        //compute lncc
+                        lncc= Metrics<ImageType,FloatImageType>::efficientLNCC(warpedImage,(*m_imageList)[targetID],m_sigma,m_exponent);
+                        //lncc= Metrics<ImageType,FloatImageType>::LSADNorm(warpedImage,(*m_imageList)[targetID],m_sigma,m_exponent);
+                        //lncc= Metrics<ImageType,FloatImageType>::LSSDNorm(warpedImage,(*m_imageList)[targetID],m_sigma,m_exponent);
+                        //lncc= Metrics<ImageType,FloatImageType>::LSSD(warpedImage,(*m_imageList)[targetID],m_sigma);
+                        //lncc= Metrics<ImageType,FloatImageType>::localMetricAutocorrelation(warpedImage,(*m_imageList)[targetID],m_sigma,2,"lssd");
+                        //FloatImagePointerType laplacian=FilterUtils<ImageType,FloatImageType>::laplacian((*m_imageList)[targetID],m_sigma);
+                        if (0){
+                            FloatImagePointerType laplacian=FilterUtils<ImageType,FloatImageType>::normalizedLaplacianWeighting((*m_imageList)[targetID],m_sigma,m_exponent);
+                            ostringstream oss2;
+                            oss2<<"laplacian-"<<sourceID<<"-TO-"<<targetID<<".nii";
+                                
+                            LOGI(6,ImageUtils<ImageType>::writeImage(oss2.str(),FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(laplacian,255))));
+                                
+                            lncc=ImageUtils<FloatImageType>::multiplyImageOutOfPlace(lncc,laplacian);
+                        }
+                        LOGI(6,ImageUtils<ImageType>::writeImage(oss.str(),FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(lncc,255))));
+                        //resample lncc result
+                        if (1){
+                            //lncc = FilterUtils<FloatImageType>::gaussian(lncc,8);
+                            //lncc = FilterUtils<FloatImageType>::LinearResample(lncc, FilterUtils<ImageType,FloatImageType>::cast(this->m_ROI),false);
+                            lncc = FilterUtils<FloatImageType>::LinearResample(lncc, FilterUtils<ImageType,FloatImageType>::cast(this->m_ROI),true);
+                        }else{
+                            lncc = FilterUtils<FloatImageType>::minimumResample(lncc,FilterUtils<ImageType,FloatImageType>::cast(this->m_ROI), m_sigmaD);
+                        }
+#endif
+
+                          
+                        oss<<"-resampled";
+                        if (D==2){
+                            oss<<".png";
+                        }else
+                            oss<<".nii";
+                        LOGI(6,ImageUtils<ImageType>::writeImage(oss.str(),FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(lncc,255))));
+
+                      
+                    }
+                    m_pairwiseLocalWeightMaps[s][t]=lncc;
+                }
+            }
+        }
+        LOGV(1)<<"done"<<endl;
+    }//computePairwiseSimWeights
+
+
     double getIndexBasedWeight(IndexType idx,SizeType size){
         double weight=100.0;
         for (int d=0;d<D;++d){
