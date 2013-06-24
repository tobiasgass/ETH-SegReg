#pragma once
#include "matrix.h"
#include "BaseLinearSolver.h"
#include "TransformationUtils.h"
#include "Log.h"
#include "ImageUtils.h"
#include <vector>
#include <sstream>
#include "SolveAquircGlobalDeformationNormCVariables.h"
#include "Metrics.h"
#include "TemporalMedianImageFilter.h"
#include "itkGaussianImage.h"
#include <boost/lexical_cast.hpp>

template<class ImageType>
class AquircLocalDeformationAndErrorSolverIndependentDimensions: public AquircGlobalDeformationNormSolverCVariables< ImageType>{
public:
    typedef typename  TransfUtils<ImageType>::DeformationFieldType DeformationFieldType;
    typedef typename  DeformationFieldType::Pointer DeformationFieldPointerType;
    typedef typename  ImageType::OffsetType OffsetType;
    typedef typename DeformationFieldType::SpacingType SpacingType;

    typedef typename ImageUtils<ImageType>::FloatImagePointerType FloatImagePointerType;
    typedef typename ImageUtils<ImageType>::FloatImageType FloatImageType;
    typedef typename itk::ImageRegionIterator<FloatImageType> FloatImageIterator;
    typedef typename itk::ImageRegionIteratorWithIndex<DeformationFieldType> DeformationFieldIterator;
    typedef typename itk::ImageRegionIterator<ImageType> ImageIterator;
    typedef typename DeformationFieldType::PixelType DeformationType;
    typedef typename DeformationFieldType::IndexType IndexType;
    typedef typename DeformationFieldType::PointType PointType;
    typedef typename ImageType::Pointer ImagePointerType;
    typedef typename ImageType::ConstPointer ConstImagePointerType;
    typedef typename ImageType::RegionType RegionType;
    static const unsigned int D=ImageType::ImageDimension;
    static const unsigned int internalD=1;
protected:
    int m_nVars,m_nEqs,m_nNonZeroes;
    int m_numImages;
    map< string, map <string, DeformationFieldPointerType> > * m_deformationCache,* m_trueDeformations, *m_updatedDeformationCache,*m_downSampledDeformationCache;
    std::vector<string> * m_imageIDList;
    bool m_additive, m_updateDeformations;
    RegionType m_regionOfInterest;
    
    map<string,ImagePointerType> * m_imageList;
    map<string,ImagePointerType> * m_segmentationList;

private:
    int m_nPixels;// number of pixels/voxels

    int m_nEqWd; // number of equations for  observation energy E_d
    int m_nVarWd; // number of variables for each equation of E_d;
    double m_wWd;

    int m_nEqWdelta; // number of equations for error minimizing energy E_delta
    int m_nVarWdelta; // number of variables for each equation of E_delta;
    double m_wWdelta;

    int m_nEqWcirc; // number of equations for energy circular constraint E_circ
    int m_nVarWcirc; // number of variables for each equation of E_circ;
    double m_wWcirc;

    int m_nEqWincErr; // number of equations for energy incErrular constraint E_incErr
    int m_nVarWincErr; // number of variables for each equation of E_incErr;
    double m_wWincErr;

    int m_nEqWs; // number of equations for spatial smoothing energy E_s
    int m_nVarWs; // number of variables for each equation of E_d;
    double m_wWs;

    int m_nEqWsDelta; // number of equations for spatial smoothing energy E_s
    int m_nVarWsDelta; // number of variables for each equation of E_d;
    double m_wWsDelta;

    int m_nEqWT; // number of equations for Transformation similarity energy E_T
    int m_nVarWT; // number of variables for each equation of E_T
    double m_wWT;

    double m_wSum;

    double m_sigma, m_sigmaD;

    double m_exponent;
    
    double m_shearingReduction;
    bool m_linearInterpol;
    bool m_haveDeformationEstimate;

    double m_segConsisntencyWeight;

    std::vector<mxArray * > m_results;

    bool m_estDef,m_estError;
public:
    AquircLocalDeformationAndErrorSolverIndependentDimensions(){
        m_wWT=1.0;
        m_wWs=1.0;
        m_wWcirc=1.0;
        m_wWdelta=1.0;
        m_wWd=1.0;
        m_wSum=1.0;
        m_sigma = 10.0;
        m_linearInterpol=false;
        m_haveDeformationEstimate=false;
        m_updatedDeformationCache = new  map< string, map <string, DeformationFieldPointerType> > ; 
        m_results = std::vector<mxArray * >(D);
        //m_updateDeformations=true;
        m_updateDeformations=false;
        m_exponent=1.0;
        m_shearingReduction = 1.0;
        m_segConsisntencyWeight = 1.0;
        m_sigmaD = 0.0;
    }
    virtual void SetVariables(std::vector<string> * imageIDList, map< string, map <string, DeformationFieldPointerType> > * deformationCache, map< string, map <string, DeformationFieldPointerType> > * trueDeformations,ImagePointerType ROI, map<string,ImagePointerType> * imagelist, map< string, map <string, DeformationFieldPointerType> > * downSampledDeformationCache){
        m_imageList=imagelist;
        m_imageIDList=imageIDList;
        m_deformationCache=deformationCache;
        m_downSampledDeformationCache=downSampledDeformationCache;
        m_numImages=imageIDList->size();
        this->m_ROI=ROI;
        if (!ROI.IsNotNull()){
            this->m_ROI=FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::createEmpty(TransfUtils<ImageType>::computeLocalDeformationNorm((*m_downSampledDeformationCache)[(*m_imageIDList)[0]][(*m_imageIDList)[1]],1.0)));
        }
        m_nPixels=this->m_ROI->GetLargestPossibleRegion().GetNumberOfPixels( );

        int interpolationFactor;
        if (m_linearInterpol)
            interpolationFactor = pow(2,D); //linear interpolation
        else
            interpolationFactor = 1 ; //NNinterpolation;
        

        m_nEqWd  = (m_wWd>0.0)* internalD * m_nPixels *  m_numImages*(m_numImages-1)*(m_numImages-2); //there is one equation for each component of every pixel of every triple of images
        m_nVarWd = 2*(interpolationFactor+2); //two variables per uniqe pair in the triple (2*2), plus linear interpolation for the third pair (2*2^D)

        m_nEqWcirc =  (m_wWcirc>0.0)* m_nPixels * internalD * m_numImages*(m_numImages-1)*(m_numImages-2); //again all components of all triples
        m_nVarWcirc = interpolationFactor+2 ; // only one/2^D variables per pair
        LOG<<VAR(D)<<" "<<VAR(interpolationFactor)<<" "<<VAR(m_nVarWcirc)<<std::endl;
        
        m_nEqWincErr =  (m_wWincErr>0.0)* m_nPixels * internalD * m_numImages*(m_numImages-1)*(m_numImages-2); //again all components of all triples
        m_nVarWincErr = interpolationFactor+2; // only one/2^D variables per pair




        m_nEqWs =  (m_wWs>0.0)* D* m_nPixels * internalD * m_numImages*(m_numImages-1); //every pixel in each registration has D neighbors (in one direction), and each component separately
        m_nVarWs = 3; //for piecewise linear regularization, 2 for piecewise constant

        m_nEqWsDelta =  (m_wWsDelta>0.0)* D* m_nPixels * internalD * m_numImages*(m_numImages-1); //every pixel in each registration has D neighbors (in one direction), and each component separately
        m_nVarWsDelta = 3; //for piecewise linear regularization, 2 for piecewise constant
      
        m_nEqWdelta = (m_wWdelta>0.0)*  m_nPixels * internalD * m_numImages*(m_numImages-1); //every error at every location in each image pair
        m_nVarWdelta = 1;

        m_nEqWT =  (m_wWT>0.0)*m_nPixels * internalD * m_numImages*(m_numImages-1); //same as Wdelta
        m_nVarWT= 1;

        int m_nEqSUM=(m_wSum>0.0)*m_nPixels * internalD * m_numImages*(m_numImages-1);
        int m_nVarSUM=2;

        m_nEqs=  m_nEqWd + m_nEqWcirc+ m_nEqWs + m_nEqWdelta+ m_nEqWT + m_nEqSUM +  m_nEqWsDelta + m_nEqWincErr; // total number of equations
        
        m_estError= m_nEqWd ||  m_nEqWdelta || m_nEqSUM||m_nEqWincErr;
        m_estDef = m_nEqWd || m_nEqWT ||  m_nEqWs  ||  m_nEqWcirc || m_nEqSUM  ;
        m_nVars= m_numImages*(m_numImages-1)*m_nPixels*internalD *(m_estError + m_estDef); // total number of free variables (error and deformation)
        
        m_nNonZeroes=  m_nEqWsDelta*m_nVarWsDelta +m_nEqWd *m_nVarWd + m_nEqWcirc * m_nVarWcirc + m_nEqWs*m_nVarWs + m_nEqWdelta*m_nVarWdelta + m_nEqWT*m_nVarWT + m_nEqSUM*m_nVarSUM + m_nVarWincErr*m_nEqWincErr; //maximum number of non-zeros

        m_trueDeformations=trueDeformations;


        m_regionOfInterest.SetSize(this->m_ROI->GetLargestPossibleRegion().GetSize());
        IndexType startIndex,nullIdx;
        nullIdx.Fill(0);
        PointType startPoint;
        this->m_ROI->TransformIndexToPhysicalPoint(nullIdx,startPoint);
        (*m_downSampledDeformationCache)[(*m_imageIDList)[0]][(*m_imageIDList)[1]]->TransformPhysicalPointToIndex(startPoint,startIndex);
        m_regionOfInterest.SetIndex(startIndex);

    }

    void setWeightWd(double w){m_wWd=w;}
    void setWeightWT(double w){m_wWT=w;}
    void setWeightWs(double w){m_wWs=w;}
    void setWeightWsDelta(double w){m_wWsDelta=w;}
    void setWeightWcirc(double w){m_wWcirc=w;}
    void setWeightWdelta(double w){m_wWdelta=w;}
    void setWeightSum(double w){m_wSum=w;}
    void setWeightInconsistencyError(double w){m_wWincErr=w;}
    void setLinearInterpol(bool i){m_linearInterpol=i;}
    void setSigma(double s){m_sigma=s;}
    void setSigmaD(double s){m_sigmaD=s;}
    void setLocalWeightExp(double e){ m_exponent=e;}
    void setShearingReduction(double r){m_shearingReduction = r;}
    void setSegmentationList(  map<string,ImagePointerType> * list){ m_segmentationList = list; }
    void setScalingFactorForConsistentSegmentation(double scalingFactorForConsistentSegmentation){ m_segConsisntencyWeight = scalingFactorForConsistentSegmentation;}

    virtual void createSystem(){

        LOG<<"Creating equation system.."<<endl;
        LOG<<VAR(m_numImages)<<" "<<VAR(m_nPixels)<<" "<<VAR(m_nEqs)<<" "<<VAR(m_nVars)<<" "<<VAR(m_nNonZeroes)<<endl;
        LOG<<VAR(  m_wWT)<<" "<<VAR(        m_wWs)<<" "<<VAR(m_wWcirc)<<" "<<VAR(m_wWdelta)<<" "<<VAR(m_wWd)<<" "<<VAR(m_wSum)<<" "<<VAR(m_wWsDelta)<<" "<<VAR(m_sigma)<<" "<<VAR(m_sigmaD)<<" "<<VAR(m_exponent)<<endl;
        double totalInconsistency = 0.0;
        int totalCount = 0;
        for (unsigned int d = 0; d< D; ++d){

            mxArray *mxX=mxCreateDoubleMatrix((mwSize)m_nNonZeroes,1,mxREAL);
            mxArray *mxY=mxCreateDoubleMatrix((mwSize)m_nNonZeroes,1,mxREAL);
            mxArray *mxV=mxCreateDoubleMatrix((mwSize)m_nNonZeroes,1,mxREAL);
            mxArray *mxB=mxCreateDoubleMatrix((mwSize)m_nEqs,1,mxREAL);
            
            mxArray *mxInit=mxCreateDoubleMatrix((mwSize)m_nVars,1,mxREAL);
            mxArray *mxUpperBound=mxCreateDoubleMatrix((mwSize)m_nVars,1,mxREAL);
            mxArray *mxLowerBound=mxCreateDoubleMatrix((mwSize)m_nVars,1,mxREAL);
          

            if ( !mxX || !mxY || !mxV || !mxB || !mxInit){
                LOG<<"couldn't allocate memory!"<<endl;
                exit(0);
            }
            double * x=( double *)mxGetData(mxX);
            std::fill(x,x+m_nNonZeroes,m_nEqs);
            double * y=( double *)mxGetData(mxY);
            std::fill(y,y+m_nNonZeroes,m_nVars);
            double * v=( double *)mxGetData(mxV);
            double * b=mxGetPr(mxB);
        
            double * init=mxGetPr(mxInit);
            double * lb=mxGetPr(mxLowerBound);
            std::fill(lb,lb+m_nVars,-200);
            double * ub=mxGetPr(mxUpperBound);
            std::fill(ub,ub+m_nVars,200);
            LOG<<"creating"<<VAR(d)<<endl;
     


            char buffer[256+1];
            buffer[256] = '\0';
            engOutputBuffer(this->m_ep, buffer, 256);
      
            //attention matlab index convention?!?
            long int eq = 1;
            long int c=0;
            double maxAbsDisplacement=0.0;
            
            //0=min,1=mean,2=max,3=median,-1=off,4=gauss;
            int accumulate=4;
            
            for (int s = 0;s<m_numImages;++s){                            
                int source=s;
                for (int t=0;t<m_numImages;++t){
                    if (t!=s){
                        int target=t;
                        DeformationFieldPointerType dSourceTarget=(*m_downSampledDeformationCache)[(*m_imageIDList)[source]][(*m_imageIDList)[target]];
                        FloatImagePointerType diffSums=TransfUtils<FloatImageType>::createEmptyImage(dSourceTarget);
                        FloatImageIterator diffSumIt(diffSums,diffSums->GetLargestPossibleRegion());
                        typedef TemporalMedianImageFilter<FloatImageType> MedianFilterType;
                        MedianFilterType medianFilter;
                        typedef GaussianEstimatorScalarImage<FloatImageType> GaussEstimatorType;
                        GaussEstimatorType gaussEstimator;
                        switch(accumulate){
                        case 0:
                            diffSums->FillBuffer(100000.0);
                            break;
                        case 1:
                            diffSums->FillBuffer(0.0);
                            break;
                        case 2:
                            diffSums->FillBuffer(-100000.0);
                            break;
                        case 3:
                            medianFilter=MedianFilterType(diffSums,m_numImages-2);
                            break;
                        case -1:
                            break;
                        case 4:
                            break;
                        }
                            


                        //triplet energies
                        for (int i=0;i<m_numImages;++i){ 
                            if (t!=i && i!=s){
                                //define a set of 3 images
                                int intermediate=i;
                                DeformationFieldPointerType dSourceIntermediate=(*m_downSampledDeformationCache)[(*m_imageIDList)[intermediate]][(*m_imageIDList)[target]];
                                DeformationFieldPointerType dIntermediateTarget=(*m_downSampledDeformationCache)[(*m_imageIDList)[source]][(*m_imageIDList)[intermediate]];

                            
                                //compute indirect deform
                                DeformationFieldPointerType indirectDeform = TransfUtils<ImageType>::composeDeformations(dSourceIntermediate,dIntermediateTarget);
                                //compute difference of direct and indirect deform
                                DeformationFieldPointerType difference = TransfUtils<ImageType>::subtract(indirectDeform,dSourceTarget);
                            
                                //compute norm
                                DeformationFieldIterator it(difference,m_regionOfInterest);
                                it.GoToBegin();
                            
                                DeformationFieldIterator trueIt;
                                if ((*m_trueDeformations)[(*m_imageIDList)[intermediate]][(*m_imageIDList)[target]].IsNotNull()){
                                    trueIt=DeformationFieldIterator((*m_trueDeformations)[(*m_imageIDList)[intermediate]][(*m_imageIDList)[target]],(*m_trueDeformations)[(*m_imageIDList)[intermediate]][(*m_imageIDList)[target]]->GetLargestPossibleRegion());
                                    trueIt.GoToBegin();
                                }
                                
                                ImageIterator segmentationIt;
                                bool haveSeg=false;
                                
                                if (false && m_segmentationList->find((*m_imageIDList)[target]) != m_segmentationList->end()){
                                    segmentationIt=ImageIterator((*m_segmentationList)[(*m_imageIDList)[target]],(*m_segmentationList)[(*m_imageIDList)[target]]->GetLargestPossibleRegion());
                                    haveSeg = true;
                                }

                               
                                diffSumIt.GoToBegin();
                                // LOG<<VAR(dir)<<" "<<VAR(start)<<endl;
                                for (;!it.IsAtEnd();++it,++diffSumIt){

                                    bool valid=true;

                                    //get index in target domain
                                    IndexType targetIndex=it.GetIndex(),intermediateIndex,idx1;
                                    LOGV(9)<<VAR(targetIndex)<<endl;
                                    PointType ptIntermediate,ptTarget;
                                    IndexType roiIntermediateIndex,roiTargetIndex;
                                
                                    //get physical point in target domain
                                    dSourceTarget->TransformIndexToPhysicalPoint(targetIndex,ptTarget);
                                    //get corresponding point in intermediate deform
                                    DeformationType dIntermediate=dSourceIntermediate->GetPixel(targetIndex);
                                    ptIntermediate= ptTarget + dIntermediate;
                                
                                    //get neighbors of that point
                                    std::vector<std::pair<IndexType,double> >ptIntermediateNeighbors,ptIntermediateNeighborsCircle;
                                    bool inside;
                                    if (m_linearInterpol){
                                        inside=getLinearNeighbors(dIntermediateTarget,ptIntermediate,ptIntermediateNeighbors);
                                    }else{
                                        inside=getNearestNeighbors(dIntermediateTarget,ptIntermediate,ptIntermediateNeighbors);
                                    }

                                    //this can be used to index the circle constraint equation with the true deform if known. cheating!
                                    //or with an estimation from the previous iteration
                                    //#define CHEATING                          
                                    //if (true  && (*m_trueDeformations)[(*m_imageIDList)[target]][(*m_imageIDList)[intermediate]].IsNotNull()){
                                    //
  
#if 1
                                    //#define ORACLE
                                    
#ifdef ORACLE
                                        if (true){
                                            DeformationType trueDef=(*m_trueDeformations)[(*m_imageIDList)[intermediate]][(*m_imageIDList)[target]]->GetPixel(targetIndex);
#else
                                            if (true && m_haveDeformationEstimate && (*m_updatedDeformationCache)[(*m_imageIDList)[intermediate]][(*m_imageIDList)[target]].IsNotNull()){
                                        
                                                DeformationType trueDef=(*m_updatedDeformationCache)[(*m_imageIDList)[intermediate]][(*m_imageIDList)[target]]->GetPixel(targetIndex);
#endif
                                                PointType truePtIntermediate=ptTarget + trueDef;
                                                inside= inside && getLinearNeighbors(dIntermediateTarget,truePtIntermediate,ptIntermediateNeighborsCircle);
                                                ++trueIt;
                                            }else{
                                                ptIntermediateNeighborsCircle=ptIntermediateNeighbors;
#ifdef ORACLE
                                            }
#else
                                        }
#endif
#endif
                                        
                                        this->m_ROI->TransformPhysicalPointToIndex(ptTarget,roiTargetIndex);
                                        LOGV(9)<<VAR(targetIndex)<<" "<<VAR(roiTargetIndex)<<endl;
                                
                                        if (inside){
                                            double val=1.0;
                                            bool segVal=1.0;

                                            //multiply val by segConsistencyWeight if deformation starts from atlas segmentation
                                            if (haveSeg){
                                                segVal=segmentationIt.Get()>0;
                                                if (segVal){
                                                    val=val*m_segConsisntencyWeight;
                                                }
                                                ++segmentationIt;
                                            }

                                            DeformationType localDiscrepance=it.Get();
                                            float disp=localDiscrepance[d];
                                            float discNorm=localDiscrepance.GetNorm();
                                            switch(accumulate){
                                            case 0:
                                                diffSumIt.Set(min(diffSumIt.Get(),discNorm));
                                                break;
                                            case 1:
                                                diffSumIt.Set(diffSumIt.Get()+discNorm);
                                                break;
                                            case 2:
                                                diffSumIt.Set(max(diffSumIt.Get(),discNorm));
                                                break;
                                            case 3:
                                                diffSumIt.Set(discNorm);
                                                break;
                                            case -1:
                                                break;
                                            case 4:
                                                diffSumIt.Set(disp);
                                                break;
                                            }

                                            //diffSumIt.Set(diffSumIt.Get()+(disp));

                                            //only count inconsistencies of atlas segmentation, if given
                                            if (segVal){
                                                totalInconsistency += discNorm;//fabs(disp);
                                                totalCount++;
                                            }

                                            LOGV(9)<<VAR(source)<<" "<<VAR(intermediate)<<" "<<VAR(target)<<" "<<VAR(roiTargetIndex)<<" "<<VAR(d)<<endl;
                                            LOGV(9)<<VAR(edgeNumError(intermediate,target,roiTargetIndex,d))<<" "<<VAR(edgeNumDeformation(intermediate,target,roiTargetIndex,d))<<endl;
                                            //set w_d ~ 
                                            if (m_wWd>0){
                                                //def and error intermediate->target
                                                x[c]=eq;
                                                y[c]=edgeNumError(intermediate,target,roiTargetIndex,d);
                                                v[c++]=val*m_wWd;
                                                x[c]=eq;
                                                y[c]=edgeNumDeformation(intermediate,target,roiTargetIndex,d);
                                                v[c++]=val*m_wWd;
                                                //interpolated def and error source->intermediate
                                                for (int i=0;i<ptIntermediateNeighbors.size();++i){
                                                    x[c]=eq;
                                                    y[c]=edgeNumError(source,intermediate,ptIntermediateNeighbors[i].first,d);
                                                    v[c++]=ptIntermediateNeighbors[i].second*val*m_wWd;
                                                    x[c]=eq;
                                                    y[c]=edgeNumDeformation(source,intermediate,ptIntermediateNeighbors[i].first,d);
                                                    v[c++]=ptIntermediateNeighbors[i].second*val*m_wWd;
                                                }
                                                //minus def and error source->target
                                                x[c]=eq;
                                                y[c]=edgeNumError(source,target,roiTargetIndex,d);
                                                v[c++]= - val*m_wWd;
                                                x[c]=eq;
                                                y[c]=edgeNumDeformation(source,target,roiTargetIndex,d);
                                                v[c++]= - val*m_wWd;
                                                b[eq-1]= disp*m_wWd;
                                                ++eq;
                                            
                                            }
                                    
                                            //set w_circ
                                            if (m_wWcirc>0.0){
                                                if (false && m_sigmaD>0.0){
                                                    val *= ( 1.0 - exp(- disp* disp / m_sigmaD ) );
                                                }else if(false &&  m_sigmaD){
                                                    //val *= 1.0/( fabs(disp)/sigmaD+1);
                                                    val *= 1.0/( fabs(disp)+m_sigmaD);
                                                }else if (false && m_sigmaD>0.){
                                                    val*=exp(-localDiscrepance.GetNorm()/m_sigmaD);
                                                    //val*=1.0/(localDiscrepance.GetNorm()+m_sigmaD);
                                                }
                                            
                                                //indirect
                                                LOGV(8)<<VAR(localDiscrepance.GetNorm())<<" "<<VAR(val)<<endl;
                                                x[c]=eq;
                                                y[c]=edgeNumDeformation(intermediate,target,roiTargetIndex,d);
                                                v[c++]=val* m_wWcirc;
                                                
#if 0
                                                //EXPERIMENTAL*****************************************
                                                x[c]=eq;
                                                y[c]=edgeNumError(intermediate,target,roiTargetIndex,d);
                                                v[c++]=-val*m_wWcirc;
                                                // *****************************************************
#endif

                                                for (int i=0;i<ptIntermediateNeighborsCircle.size();++i){
                                                    x[c]=eq;
                                                    y[c]=edgeNumDeformation(source,intermediate,ptIntermediateNeighborsCircle[i].first,d); // this is a APPROXIMIATION!!! might be bad :o
                                                    v[c++]=ptIntermediateNeighborsCircle[i].second*val* m_wWcirc;
                                                    LOGV(8)<<VAR(i)<<" "<<VAR(ptIntermediateNeighborsCircle[i].first)<<" "<<VAR(ptIntermediateNeighborsCircle[i].second)<<endl;
                                                }
                                                //minus direct
                                                x[c]=eq;
                                                y[c]=edgeNumDeformation(source,target,roiTargetIndex,d);
                                                v[c++]= - val* m_wWcirc;

                                                b[eq-1]=0;
                                                ++eq;
                                            }

                                            if (m_wWincErr>0.0){
                                                val=1.0;
                                                //indirect
                                                x[c]=eq;
                                                y[c]=edgeNumError(intermediate,target,roiTargetIndex,d);
                                                v[c++]=2*val* m_wWincErr;
                                               
                                                for (int i=0;i<ptIntermediateNeighborsCircle.size();++i){
                                                    x[c]=eq;
                                                    y[c]=edgeNumError(source,intermediate,ptIntermediateNeighborsCircle[i].first,d); // this is a APPROXIMIATION!!! might be bad :o
                                                    v[c++]=ptIntermediateNeighborsCircle[i].second*val* m_wWincErr;
                                                }
                                                //minus direct
                                                x[c]=eq;
                                                y[c]=edgeNumError(source,target,roiTargetIndex,d);
                                                v[c++]= - val* m_wWincErr;
                                                b[eq-1]=disp;
                                                ++eq;

                                            }
                                        }//inside
                                    
                                }//image iterator
                                if (accumulate == 3){
                                    medianFilter.insertImage(diffSums);
                                }else if (accumulate==4){
                                    gaussEstimator.addImage(diffSums);
                                    LOGV(4)<<"adding image to gauss estimator"<<endl;
                                }

                            }//if

                          
                        }//target
                        if (accumulate == 3){
                            diffSums=medianFilter.getMedian();
                        }else if (accumulate ==4 ){
                            gaussEstimator.finalize();
                        }

                        //pairwise energies!
                        string sourceID=(*this->m_imageIDList)[source];
                        string targetID = (*this->m_imageIDList)[target];
                        {
                            if (accumulate != -1 && m_exponent !=0){
                                if (accumulate == 2){
                                    ImageUtils<FloatImageType>::multiplyImage(diffSums,1.0/(m_numImages-2));
                                }
                                ImageUtils<FloatImageType>::expNormImage(diffSums,m_exponent);
                                ostringstream oss1;
                                oss1<<"diffSum-"<<sourceID<<"-TO-"<<targetID;
                                if (D==2)
                                    oss1<<".png";
                                else
                                    oss1<<".nii";
                                LOGI(6,ImageUtils<ImageType>::writeImage(oss1.str(),FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(diffSums,255))));
                            }
                        }
                                
                      

                      
                 
                        FloatImagePointerType lncc;
                        FloatImageIterator lnccIt;
                        if (m_sigma>0.0 && m_wWdelta>0.0){
                            ostringstream oss;
                            oss<<"lncc-"<<sourceID<<"-TO-"<<targetID;
                            if (D==2)
                                oss<<".png";
                            else
                                oss<<".nii";

#ifdef ORACLE
                            DeformationFieldPointerType diff=TransfUtils<ImageType>::subtract(
                                                                                              (*this->m_downSampledDeformationCache)[sourceID][targetID],
                                                                                              (*m_trueDeformations)[sourceID][targetID]
                                                                                              );
                                                                                              
                            lncc=TransfUtils<ImageType>::computeLocalDeformationNormWeights(diff,m_exponent);
#else
                            DeformationFieldPointerType def = (*this->m_deformationCache)[sourceID][targetID];
                            ImagePointerType warpedImage= TransfUtils<ImageType>::warpImage((ConstImagePointerType)(*m_imageList)[sourceID],def);
                            //compute lncc
                            lncc= Metrics<ImageType,FloatImageType>::efficientLNCC(warpedImage,(*m_imageList)[targetID],m_sigma,m_exponent);
                            //lncc= Metrics<ImageType,FloatImageType>::LSADNorm(warpedImage,(*m_imageList)[targetID],m_sigma,m_exponent);
                            //lncc= FilterUtils<ImageType,FloatImageType>::LSSDNorm(warpedImage,(*m_imageList)[targetID],m_sigma,m_exponent);
                            //lncc= Metrics<ImageType,FloatImageType>::localMetricAutocorrelation(warpedImage,(*m_imageList)[targetID],m_sigma,2,"lssd");
                            //FloatImagePointerType laplacian=FilterUtils<ImageType,FloatImageType>::laplacian((*m_imageList)[targetID],m_sigma);
                            if (0){
                                FloatImagePointerType laplacian=FilterUtils<ImageType,FloatImageType>::normalizedLaplacianWeighting((*m_imageList)[targetID],m_sigma,m_exponent);
                                ostringstream oss2;
                                oss2<<"laplacian-"<<sourceID<<"-TO-"<<targetID<<".nii";
                                
                                LOGI(6,ImageUtils<ImageType>::writeImage(oss2.str(),FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(laplacian,255))));
                                
                                lncc=ImageUtils<FloatImageType>::multiplyImageOutOfPlace(lncc,laplacian);
                            }
                            LOGI(6,ImageUtils<ImageType>::writeImage(oss.str(),FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(lncc,255))));
                            //resample lncc result
                            if (1){
                                //lncc = FilterUtils<FloatImageType>::gaussian(lncc,8);
                                //lncc = FilterUtils<FloatImageType>::LinearResample(lncc, FilterUtils<ImageType,FloatImageType>::cast(this->m_ROI),false);
                                lncc = FilterUtils<FloatImageType>::LinearResample(lncc, FilterUtils<ImageType,FloatImageType>::cast(this->m_ROI),true);
                            }else{
                                lncc = FilterUtils<FloatImageType>::minimumResample(lncc,FilterUtils<ImageType,FloatImageType>::cast(this->m_ROI), m_sigmaD);
                            }
#endif

                          
                            oss<<"-resampled";
                            if (D==2){
                                oss<<".png";
                            }else
                                oss<<".nii";
                            LOGI(6,ImageUtils<ImageType>::writeImage(oss.str(),FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(lncc,255))));

                            lnccIt=FloatImageIterator(lncc,lncc->GetLargestPossibleRegion());
                            lnccIt.GoToBegin();
                        }

                        DeformationFieldIterator previousIt;
                        if (true && m_haveDeformationEstimate && (*m_updatedDeformationCache)[sourceID][targetID].IsNotNull()){
                            DeformationFieldPointerType estDef=(*m_updatedDeformationCache)[sourceID][targetID];
                            previousIt=DeformationFieldIterator(estDef,m_regionOfInterest);
                            previousIt.GoToBegin();
                        }
                        
                        DeformationFieldPointerType defSourceInterm=(*this->m_downSampledDeformationCache)[sourceID][targetID];
                        DeformationFieldIterator it(defSourceInterm,m_regionOfInterest);
                        it.GoToBegin();
                        diffSumIt.GoToBegin();

           

                        for (;!it.IsAtEnd();++it){
                            DeformationType localDef=it.Get();
                            if (abs(localDef[d])>maxAbsDisplacement){
                                maxAbsDisplacement=abs(localDef[d]);
                            }
                            IndexType idx=it.GetIndex();
                            LOGV(8)<<VAR(eq)<<" "<<VAR(localDef)<<endl;
                        
                            int edgeNumDef=edgeNumDeformation(source,target,idx,d);
                            int edgeNumErr=edgeNumError(source,target,idx,d);

                            //intensity based weight
                            double weight=1.0;
                            if (lncc.IsNotNull()){
                                weight = lnccIt.Get();
                                LOGV(6)<<VAR(weight)<<endl;
                                ++lnccIt;
                            }
                            double expectedError=0.0;
                            if (false){
                                weight=diffSumIt.Get();
                                //                                weight = exp(-fabs(diffSumIt.Get())/m_sigmaD);
                                //weigth = 1.0/(diffSumIt.Get()/( m_numImages-2)+m_sigmaD);
                                //expectedError = diffSumIt.Get()/(m_numImages-2);
                            }
                                
                            //weight based on previous estimate
                            double weight2=1.0;
                            if (false && m_haveDeformationEstimate && m_sigmaD>0.0){
                                weight2 = exp ( - (localDef-previousIt.Get()).GetSquaredNorm() / m_sigmaD );
                                ++previousIt;
                            }
                                
                            //set w_delta
                            //set eqn for soft constraining the error to be small
                            if (m_wWdelta>0.0){
                                x[c]=eq;
                                y[c]= edgeNumErr;
                                LOGV(8)<<VAR(source)<<" "<<VAR(target)<<" "<<VAR(idx)<<" "<<VAR(d)<<" "<<VAR(edgeNumErr)<<endl;
                                v[c++]=1.0*m_wWdelta*weight*weight2;
                                b[eq-1]=expectedError*weight*m_wWdelta;
                                ++eq;
                            }

                            //set w_T
                            //set eqn for soft constraining the estimated true deformation to be similar to the original deformation
                            if (m_wWT>0.0){
                                x[c]=eq;
                                y[c]=edgeNumDef;
                                LOGV(8)<<VAR(source)<<" "<<VAR(target)<<" "<<VAR(idx)<<" "<<VAR(d)<<" "<<VAR(edgeNumErr)<<endl;
                                v[c++]=1.0*m_wWT ;//* weight;
                                b[eq-1]=localDef[d]*m_wWT;// * weight;
                                ++eq;
                            }
                            
                            
                            //constraint that estimated def + estimated error = original def
                            if (m_wSum>0.0){
                                x[c]=eq;
                                y[c]=edgeNumDef;
                                v[c++]=m_wSum;
                                x[c]=eq;
                                y[c]=edgeNumErr;
                                v[c++]=m_wSum;
                                b[eq-1]=m_wSum*localDef[d];
                                ++eq;
                            }
                          
                            //spatial smootheness of estimated deformations
                            if (m_wWs>0.0){
                                for (unsigned int n=0;n<D;++n){
                                    OffsetType off,off2;
                                    off.Fill(0);
                                    off2=off;
                                    off[n]=1;
                                    off2[n]=-1;
                                    double smoothenessWeight =this->m_wWs;
                                    if (n!=d){
                                        //smoothenss for shearing is different
                                        smoothenessWeight*=m_shearingReduction;
                                    }
                                    IndexType neighborIndexRight=idx+off;
                                    IndexType neighborIndexLeft=idx+off2;
                                    if (defSourceInterm->GetLargestPossibleRegion().IsInside(neighborIndexRight) &&defSourceInterm->GetLargestPossibleRegion().IsInside(neighborIndexLeft) ){
                                        x[c]=eq;
                                        y[c]=edgeNumErr;
                                        v[c++]=-2*smoothenessWeight;
                                        x[c]=eq;
                                        y[c]=edgeNumDeformation(source,target,neighborIndexRight,d);
                                        v[c++]=smoothenessWeight;
                                        x[c]=eq;
                                        y[c]=edgeNumDeformation(source,target,neighborIndexLeft,d);
                                        v[c++]=smoothenessWeight;
                                        b[eq-1]=0.0;
                                        ++eq;
                                    }
                                }//inside
                            }
                            //spatial un-smootheness of estimated errors
                            if (m_wWsDelta>0.0){
                                for (unsigned int n=0;n<D;++n){
                                    OffsetType off,off2;
                                    off.Fill(0);
                                    off2=off;
                                    off[n]=1;
                                    off2[n]=-1;
                                    IndexType neighborIndexRight=idx+off;
                                    IndexType neighborIndexLeft=idx+off2;
                                    if (defSourceInterm->GetLargestPossibleRegion().IsInside(neighborIndexRight) &&defSourceInterm->GetLargestPossibleRegion().IsInside(neighborIndexLeft) ){
                                        x[c]=eq;
                                        y[c]=edgeNumErr;
                                        v[c++]=-2.0;
                                        x[c]=eq;
                                        y[c]=edgeNumError(source,target,neighborIndexRight,d);
                                        v[c++]=1.0;
                                        x[c]=eq;
                                        y[c]=edgeNumError(source,target,neighborIndexLeft,d);
                                        v[c++]=1.0;
                                        b[eq-1]=this->m_wWsDelta;
                                        ++eq;
                                    }
                                }//inside
                            }//wwsDelta
                            //set initialisation values

                            
                            //set bounds on variables
                            //error
                            double extent=(defSourceInterm->GetLargestPossibleRegion().GetSize()[d]-1)*defSourceInterm->GetSpacing()[d];
                            double upperBound=extent;
                            double lowerBound=-upperBound;
                            LOGV(4)<<VAR(upperBound)<<endl;

                            if (accumulate == 4){
                                double mean=-gaussEstimator.getMean()->GetPixel(idx);
                                double variance=fabs(gaussEstimator.getVariance()->GetPixel(idx));
                                if (variance==0.0)
                                    variance = 1e-10;
                                lowerBound=mean-3*sqrt(variance);
                                upperBound=mean+3*sqrt(variance);
                                LOGV(6)<<VAR(lowerBound)<<VAR(upperBound)<<" "<<VAR(mean)<<" "<<VAR(variance)<<endl;
                            }
                            if (m_estError){
                                LOGV(6)<<VAR(edgeNumErr)<<" "<<VAR(m_nVars)<<endl;
                                init[edgeNumErr-1] = 0.0;
                                lb[edgeNumErr-1] = lowerBound;
                                ub[edgeNumErr-1] = upperBound;
                            }
                            //deformation
                            //deformation should not fall outside image bounds
                            PointType pt;
                            defSourceInterm->TransformIndexToPhysicalPoint(idx,pt);
                            //warning: assumes 1 1 1 direction!
                            //deformation can maximally go back to origin
                            if (m_estDef){
                                LOGV(6)<<VAR(edgeNumDef)<<" "<<VAR(m_nVars)<<endl;
                                lb[edgeNumDef-1] =  defSourceInterm->GetOrigin()[d]-pt[d] -0.0001;
                                //deformation can maximally transform pt to extent of image
                                ub[edgeNumDef-1] =  defSourceInterm->GetOrigin()[d]+extent-pt[d] +0.0001;
                                LOGV(4)<<VAR(pt)<<" "<<VAR(defSourceInterm->GetOrigin()[d]-pt[d])<<" "<<VAR( defSourceInterm->GetOrigin()[d]+extent-pt[d]  )<<endl;
                                init[edgeNumDef-1] = localDef[d] ;
                            
                            }
                           
                        }//for

                    }//if
                }//intermediate
            }//source
            LOG<<VAR(eq)<<" "<<VAR(c)<<endl;
            

            //put variables into workspace and immediately destroy them
            engPutVariable(this->m_ep,"xCord",mxX);
            mxDestroyArray(mxX);
            engPutVariable(this->m_ep,"yCord",mxY);
            mxDestroyArray(mxY);
            engPutVariable(this->m_ep,"val",mxV);
            mxDestroyArray(mxV);
            engPutVariable(this->m_ep,"b",mxB);
            mxDestroyArray(mxB);

            if (0){
                ostringstream nEqs;
                nEqs<<"nEq="<<eq<<";";
                engEvalString(this->m_ep,nEqs.str().c_str());
                ostringstream nNz;
                nNz<<"nNz="<<c<<";";
                engEvalString(this->m_ep,nNz.str().c_str());
                engEvalString(this->m_ep,"xCord=xCord(1:nNz+1)");
                engEvalString(this->m_ep,"yCord=yCord(1:nNz+1)");
                engEvalString(this->m_ep,"val=val(1:nNz+1)");
                engEvalString(this->m_ep,"b=b(1:nEq+1)");
            }

            LOG<<"Creating sparse matrix"<<endl;
            engEvalString(this->m_ep,"A=sparse(xCord,yCord,val);" );
            LOGI(6,engEvalString(this->m_ep,"save('sparse.mat');" ));
            LOG<<"done, cleaning up"<<endl;
            //clear unnneeded variables from matlab workspace
            engEvalString(this->m_ep,"clear xCord yCord val;" );

            engPutVariable(this->m_ep,"init",mxInit);
            mxDestroyArray(mxInit);
            this->haveInit=true;
            LOG<<"Solving "<<VAR(d)<<endl;
            if (0){
                engEvalString(this->m_ep, "lb=[-200000*ones(size(A,2),1)];");
                engEvalString(this->m_ep, "ub=[200000*ones(size(A,2),1);]");
                engEvalString(this->m_ep, "init=init(1:size(A,2));");
            }else{
                engPutVariable(this->m_ep,"lb",mxLowerBound);
                engPutVariable(this->m_ep,"ub",mxUpperBound);
            }
            mxDestroyArray(mxLowerBound);
            mxDestroyArray(mxUpperBound);

            
            if (1){
                
                engEvalString(this->m_ep, "options=optimset(optimset('lsqlin'),'Display','iter','TolFun',1e-54,'PrecondBandWidth',Inf);");//,'Algorithm','active-set' );");
                //solve using trust region method
                TIME(engEvalString(this->m_ep, "tic;[x resnorm residual flag  output lambda] =lsqlin(A,b,[],[],[],[],lb,ub,init);toc"));
                //solve using active set method (backslash)
                //TIME(engEvalString(this->m_ep, "tic;[x resnorm residual flag output lambda] =lsqlin(A,b,[],[],[],[],[],[],init);toc"));
                printf("%s", buffer+2);
                engEvalString(this->m_ep, " resnorm");
                printf("%s", buffer+2);
                engEvalString(this->m_ep, "output");
                printf("%s", buffer+2);
            }else{
                //solve using pseudo inverse
                //TIME(engEvalString(this->m_ep, "tic;x = pinv(full(A))*b;toc"));
            }
            LOGI(6,engEvalString(this->m_ep,"save('test.mat');" ));
            if ((m_results[d] = engGetVariable(this->m_ep,"x")) == NULL)
                printf("something went wrong when getting the variable.\n Result is probably wrong. \n");
            engEvalString(this->m_ep,"clear A b init lb ub x;" );
        }//dimensions

        double averageInconsistency = totalInconsistency/totalCount;
        LOG<<VAR(totalInconsistency)<<" "<<VAR(averageInconsistency) << endl;
    }
    virtual void solve(){}

    virtual void storeResult(string directory){
        std::vector<double> result(m_nVars);
        std::vector<double*> rData(D);
        for (int d= 0; d<D ; ++d){
            rData[d]=mxGetPr(this->m_results[d]);
        }

        ImagePointerType mask;
        
        double trueResidual=0.0;
        double estimationResidual=0.0;
        double circleResidual=0.0;
        double averageError=0.0;
        double averageOldError=0.0;
        double averageInconsistency = 0.0;
        int c3 = 0;
        int c=0;
        int c2=0;
        for (int s = 0;s<m_numImages;++s){
            for (int t=0;t<m_numImages;++t){
                if (s!=t){
                    //slightly(!!!) stupid creation of empty image
                    DeformationFieldPointerType estimatedError=TransfUtils<ImageType>::createEmpty(this->m_ROI);//ImageUtils<DeformationFieldType>::createEmpty((*m_downSampledDeformationCache)[(*m_imageIDList)[s]][(*m_imageIDList)[t]]);
                    DeformationFieldIterator itErr(estimatedError,estimatedError->GetLargestPossibleRegion());
                    DeformationFieldPointerType estimatedDeform=TransfUtils<ImageType>::createEmpty(this->m_ROI);//ImageUtils<DeformationFieldType>::createEmpty((*m_downSampledDeformationCache)[(*m_imageIDList)[s]][(*m_imageIDList)[t]]);
                    DeformationFieldIterator itDef(estimatedDeform,estimatedDeform->GetLargestPossibleRegion());
                    itErr.GoToBegin();
                    itDef.GoToBegin();

                    DeformationFieldIterator origIt((*m_downSampledDeformationCache)[(*m_imageIDList)[s]][(*m_imageIDList)[t]],(*m_downSampledDeformationCache)[(*m_imageIDList)[s]][(*m_imageIDList)[t]]->GetLargestPossibleRegion());
                    origIt.GoToBegin();

                    DeformationFieldIterator trueIt;
                    if ((*m_trueDeformations)[(*m_imageIDList)[s]][(*m_imageIDList)[t]].IsNotNull()){
                        trueIt=DeformationFieldIterator((*m_trueDeformations)[(*m_imageIDList)[s]][(*m_imageIDList)[t]],(*m_trueDeformations)[(*m_imageIDList)[s]][(*m_imageIDList)[t]]->GetLargestPossibleRegion());
                        trueIt.GoToBegin();
                    }

                  
                    for (int p=0;!itErr.IsAtEnd();++itErr,++itDef,++origIt){
                        //get solution of eqn system
                        DeformationType estimatedError,estimatedDeformation;
                        IndexType idx = itErr.GetIndex();
                        for (unsigned int d=0;d<D;++d,++p){
                            // minus 1 to correct for matlab indexing
                            estimatedError[d]=rData[d][edgeNumError(s,t,idx,d)-1];
                            estimatedDeformation[d]=rData[d][edgeNumDeformation(s,t,idx,d)-1];
                        }
                        itErr.Set(estimatedError);
                        itDef.Set(estimatedDeformation);
                        LOGV(8)<<VAR(c)<<" "<<VAR(estimatedDeformation)<<endl;
                        if ((*m_trueDeformations)[(*m_imageIDList)[s]][(*m_imageIDList)[t]].IsNotNull()){
                            DeformationType trueErr = origIt.Get()-trueIt.Get();
                            DeformationType estimatedDiffErr = estimatedDeformation - trueIt.Get();
                            //LOGV(5)<<VAR(trueErr.GetNorm())<<" "<<VAR(estimatedError.GetNorm())<<endl;
                            //LOGV(5)<<VAR(trueErr.GetNorm())<<" "<<VAR(estimatedDiffErr.GetNorm())<<endl;
                            LOGV(5)<<VAR(trueErr)<<" "<<VAR(estimatedError)<<endl;
                            LOGV(5)<<VAR(trueErr)<<" "<<VAR(estimatedDiffErr)<<endl;
                           
                            ++trueIt;
                        }
                        ++c;
                    }
                    
                    if ((*m_trueDeformations)[(*m_imageIDList)[s]][(*m_imageIDList)[t]].IsNotNull()){
                        mask=TransfUtils<ImageType>::createEmptyImage(estimatedDeform);
                        mask->FillBuffer(0);
                        typename ImageType::SizeType size=mask->GetLargestPossibleRegion().GetSize();
                        IndexType offset;
                        double fraction=0.7;
                        for (int d=0;d<D;++d){
                            offset[d]=(1.0-fraction)/2*size[d];
                            size[d]=fraction*size[d];
                        }
                        
                        typename ImageType::RegionType region;
                        region.SetSize(size);
                        region.SetIndex(offset);
                        LOGV(6)<<VAR(region)<<endl;
                        ImageUtils<ImageType>::setRegion(mask,region,1);
                        //mask = TransfUtils<ImageType>::warpImage(mask,estimatedDeform);

                        double newError=TransfUtils<ImageType>::computeDeformationNormMask(TransfUtils<ImageType>::subtract(estimatedDeform,(*m_trueDeformations)[(*m_imageIDList)[s]][(*m_imageIDList)[t]]),mask);
                        //double newError=TransfUtils<ImageType>::computeDeformationNorm(TransfUtils<ImageType>::subtract(estimatedDeform,(*m_trueDeformations)[(*m_imageIDList)[s]][(*m_imageIDList)[t]]),1);
                        //mask->FillBuffer(1);
                        //mask = TransfUtils<ImageType>::warpImage(mask,(*m_downSampledDeformationCache)[(*m_imageIDList)[s]][(*m_imageIDList)[t]]);
                        double oldError=TransfUtils<ImageType>::computeDeformationNormMask(TransfUtils<ImageType>::subtract((*m_downSampledDeformationCache)[(*m_imageIDList)[s]][(*m_imageIDList)[t]],(*m_trueDeformations)[(*m_imageIDList)[s]][(*m_imageIDList)[t]]),mask);
                        //double oldError=TransfUtils<ImageType>::computeDeformationNorm(TransfUtils<ImageType>::subtract((*m_downSampledDeformationCache)[(*m_imageIDList)[s]][(*m_imageIDList)[t]],(*m_trueDeformations)[(*m_imageIDList)[s]][(*m_imageIDList)[t]]),1);
                        LOGV(1)<<VAR(s)<<" "<<VAR(t)<<" "<<VAR(oldError)<<" "<<VAR(newError)<<endl;
                        averageError+=newError;
                        averageOldError+=oldError;
                    }
                    c2++;
                    ostringstream outfile;
                    if (directory != ""){
                        outfile<<directory<<"/estimatedLocalComposedDeformationError-FROM-"<<(*m_imageIDList)[s]<<"-TO-"<<(*m_imageIDList)[t]<<".mha";
                        ImageUtils<DeformationFieldType>::writeImage(outfile.str().c_str(),estimatedError);
                        ostringstream outfile2;
                        outfile2<<directory<<"/estimatedLocalComposedDeformation-FROM-"<<(*m_imageIDList)[s]<<"-TO-"<<(*m_imageIDList)[t]<<".mha";
                        ImageUtils<DeformationFieldType>::writeImage(outfile2.str().c_str(),estimatedDeform);
                    }
                    if (m_updateDeformations){
                        (*m_downSampledDeformationCache)[(*m_imageIDList)[s]][(*m_imageIDList)[t]]= estimatedDeform;
                    }else{
                        (*m_updatedDeformationCache)[(*m_imageIDList)[s]][(*m_imageIDList)[t]] = estimatedDeform;
                        m_haveDeformationEstimate = true;
                    }
                   
                }
            }
        }
        estimationResidual=(estimationResidual)/c;
        trueResidual=(trueResidual)/c;
        averageError/=c2;
        averageOldError/=c2;
        LOG<<VAR(averageError)<<" "<<VAR(averageOldError)<<" "<<VAR(trueResidual)<<" "<<VAR(c)<<endl;

        for (int d= 0; d<D ; ++d){
            mxDestroyArray(this->m_results[d]);
        }
        double inc;
        LOG<<"inconsistency after CLERC: "<<endl;
        if (m_updateDeformations){
            inc=computeInconsistency(m_downSampledDeformationCache,mask);
        }else{
            inc=computeInconsistency(m_updatedDeformationCache,mask);
            LOG<<"inconsistency before CLERC: "<<endl;
            inc=computeInconsistency(m_downSampledDeformationCache,mask);

        }
        LOG<<"inconsistency of true deformations: "<<endl;
        inc=computeInconsistency(m_trueDeformations,mask);
        //inc=computeInconsistency(m_deformationCache,mask);

      
        
    }

    std::vector<double> getResult(){
        std::vector<double> result(m_nVars);
        return result;
        

    }

public:
    double computeInconsistency(map< string, map <string, DeformationFieldPointerType> > * cache, ImagePointerType mask=NULL){
        //compute inconsistency over triplets
        double averageInconsistency=0;
        int c3=0;
        for (int s = 0;s<m_numImages;++s){
            for (int t=0;t<m_numImages;++t){
                DeformationFieldPointerType directDeform;
               
                    directDeform= (*cache)[(*m_imageIDList)[s]][(*m_imageIDList)[t]];
               
                if (s!=t){
                    for (int i=0;i<m_numImages;++i){
                        if (i!=t && i !=s){
                            DeformationFieldPointerType d0,d1;
                          
                            d0 = (*cache)[(*m_imageIDList)[s]][(*m_imageIDList)[i]];
                            d1 = (*cache)[(*m_imageIDList)[i]][(*m_imageIDList)[t]];
                            
                          
                            
                            DeformationFieldPointerType indirectDef = TransfUtils<ImageType>::composeDeformations(d1,d0);
                            indirectDef=TransfUtils<ImageType>::linearInterpolateDeformationField(indirectDef,TransfUtils<ImageType>::createEmptyImage(directDeform));
                            DeformationFieldPointerType diff  = TransfUtils<ImageType>::subtract(directDeform,indirectDef);
                            double residual = TransfUtils<ImageType>::computeDeformationNormMask(diff,mask);
                            averageInconsistency += residual;
                            c3++;
                        }
                    }
                }
            }
        }
        LOG<<VAR(averageInconsistency/c3)<<endl;
        return averageInconsistency/c3;

    }

protected:
    //return fortlaufende number of pairs n1,n2, 0..(n*(n-1)-1)
    inline long int edgeNum(int n1,int n2){ return ((n1)*(m_numImages-1) + n2 - (n2>n1));}
 

    inline long int edgeNumDeformation(int n1,int n2,IndexType idx, int d){ 
        long int offset = this->m_ROI->ComputeOffset(idx);
        //return offset*internalD+edgeNum(n1,n2)*m_nPixels*internalD + 1 ;
        return internalD*(edgeNum(n1,n2)*m_nPixels+offset) + 1 ;
    }

    inline long int edgeNumError(int n1,int n2,IndexType idx, int d){ 
        return  m_nPixels*internalD*(m_numImages-1)*(m_numImages) + edgeNumDeformation(n1,n2,idx,d);
    }


    //compose 3 deformations. order is left-to-right
    DeformationFieldPointerType composeDeformations(DeformationFieldPointerType d1,DeformationFieldPointerType d2,DeformationFieldPointerType d3){
        return TransfUtils<ImageType>::composeDeformations(d3,TransfUtils<ImageType>::composeDeformations(d2,d1));

    }


    inline bool getLinearNeighbors(const DeformationFieldPointerType def, const PointType & point, std::vector<std::pair<IndexType,double> > & neighbors){
        bool inside=false;
        neighbors= std::vector<std::pair<IndexType,double> >(pow(2,D));
        int nNeighbors=0;
        IndexType idx1;
        def->TransformPhysicalPointToIndex(point,idx1);
        inside=inside || def->GetLargestPossibleRegion().IsInside(idx1);
        if (!inside) return false;
        PointType pt1;
        def->TransformIndexToPhysicalPoint(idx1,pt1);
        DeformationType dist=point-pt1;
        if (inside){
            neighbors[nNeighbors++]=std::make_pair(idx1,getWeight(dist,def->GetSpacing()));
        }
        OffsetType off;
        off.Fill(0);
        for (int i=1;i<pow(2,D);++i){
            int spill=1;
            for (int d=0;d<D;++d){
                off[d]+=spill*sign(dist[d]);
                if (fabs(off[d])>1){
                    spill=1;off[d]=0;
                }else{
                    break;
                }

            }
            IndexType idx=idx1+off;
            PointType pt;
            def->TransformIndexToPhysicalPoint(idx,pt);
            DeformationType delta=point-pt;
            if (def->GetLargestPossibleRegion().IsInside(idx)){
                neighbors[nNeighbors++]=std::make_pair(idx,getWeight(delta,def->GetSpacing()));
                inside=true;
            }
        }
        neighbors.resize(nNeighbors);
        return inside;
    }
    inline bool getNearestNeighbors(const DeformationFieldPointerType def, const PointType & point, std::vector<std::pair<IndexType,double> > & neighbors){
        bool inside=false;
        neighbors= std::vector<std::pair<IndexType,double> >(1);
        int nNeighbors=0;
        IndexType idx1;
        def->TransformPhysicalPointToIndex(point,idx1);
        inside=inside || def->GetLargestPossibleRegion().IsInside(idx1);
        PointType pt1;
        def->TransformIndexToPhysicalPoint(idx1,pt1);
        DeformationType dist=point-pt1;
        if (inside){
            neighbors[nNeighbors++]=std::make_pair(idx1,1.0);
        }
       
        return inside;
    }

    inline double getWeight(const DeformationType & dist, const SpacingType & space){
        double w=1.0;
        for (int d=0;d<D;++d){
            w*=(1-fabs(dist[d])/space[d]);
        }
        LOGV(9)<<VAR(dist)<<" "<<VAR(space)<<" "<<VAR(w)<<endl;
        return w;
    }
    inline int sign(const double s){

        if (s>=0) return 1;
        if (s<0) return -1;
        return 0;
    }
};
