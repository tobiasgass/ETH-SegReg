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
#include "itkNormalizedCorrelationImageToImageMetric.h"
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkMeanSquaresImageToImageMetricv4.h"
#include "itkCorrelationImageToImageMetricv4.h"
#include "itkRegistrationParameterScalesFromJacobian.h"
#include "itkRegistrationParameterScalesFromPhysicalShift.h"
#include "itkLabelOverlapMeasuresImageFilter.h"
#include "SegmentationFusion.h"
#include <itkGradientMagnitudeImageFilter.h>
#include <itkGradientMagnitudeRecursiveGaussianImageFilter.h>
#include <itkDisplacementFieldJacobianDeterminantFilter.h>
#include "SegmentationTools.hxx"
template<class ImageType >
//template<class ImageType, class MetricType=itk::NormalizedCorrelationImageToImageMetric<ImageType,ImageType> >
//template<class ImageType, class MetricType=itk::MeanSquaresImageToImageMetric<ImageType,ImageType> >
class CLERCIndependentDimensions: public AquircGlobalDeformationNormSolverCVariables< ImageType>{
public:
    typedef typename  TransfUtils<ImageType>::DeformationFieldType DeformationFieldType;
    typedef typename  DeformationFieldType::Pointer DeformationFieldPointerType;
    typedef typename  ImageType::OffsetType OffsetType;
    typedef typename  DeformationFieldType::SpacingType SpacingType;

    typedef typename ImageType::SizeType SizeType;
    typedef typename ImageUtils<ImageType,double>::FloatImagePointerType FloatImagePointerType;
    typedef typename ImageUtils<ImageType,double>::FloatImageType FloatImageType;
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
    typedef GaussianEstimatorScalarImage<FloatImageType> GaussEstimatorType;
    //typedef MinEstimatorScalarImage<FloatImageType> GaussEstimatorType;

    typedef typename TransfUtils<ImageType>::DeformationCacheType DeformationCacheType;

    typedef typename TransfUtils<ImageType,double>::DisplacementFieldTransformType DisplacementFieldTransformType;
    typedef typename DisplacementFieldTransformType::Pointer DisplacementFieldTransformPointer;
    
    typedef map<string,ImagePointerType> ImageCacheType;
    typedef map<string, map< string, string> > FileListCacheType;
    

protected:
    int m_nVars,m_nEqs,m_nNonZeroes;
    int m_numImages;
    DeformationCacheType  m_deformationCache, m_trueDeformations, m_updatedDeformationCache, m_downSampledDeformationCache,m_pairwiseGradients,m_estimatedErrors;
    FileListCacheType m_deformationFileList, m_trueDeformationFileList;
    map<string,string> m_landmarkFileList;
    std::vector<string>  m_imageIDList;
    bool m_additive, m_updateDeformations, m_updateDeformationsGlobalSim;
    RegionType m_regionOfInterest;
    
    ImageCacheType  m_imageList;
    ImageCacheType  *m_maskList;
    ImageCacheType  m_atlasSegmentations,m_groundTruthSegmentations;

    map< int, map <int, GaussEstimatorType > > m_pairwiseInconsistencyStatistics;
    map< string , map <string, FloatImagePointerType > >  m_pairwiseLocalWeightMaps,  m_updatedPairwiseLocalWeightMaps;
    map< string, map <string, double> > m_pairwiseGlobalSimilarity;
    string m_metric;
private:
    bool m_smoothDeformationDownsampling;
    int m_nPixels;// number of pixels/voxels

    int m_nEqFullCircleEnergy; // number of equations for  observation energy E_d
    int m_nVarFullCircleEnergy; // number of variables for each equation of E_d;
    double m_wFullCircleEnergy;

    int m_nEqCircleNorm; // number of equations for energy circular constraint E_circ
    int m_nVarCircleNorm; // number of variables for each equation of E_circ;
    double m_wCircleNorm;

    int m_nEqTransformationSimilarity; // number of equations for Transformation similarity energy E_T
    int m_nVarTransformationSimilarity; // number of variables for each equation of E_T
    double m_wTransformationSimilarity;

    int m_nEqTransformationSymmetry; // number of equations for Transformation similarity energy E_T
    int m_nVarTransformationSymmetry; // number of variables for each equation of E_T
    double m_wTransformationSymmetry;
    
    int m_nEqErrorInconsistency; // number of equations for energy incErrular constraint E_incErr
    int m_nVarErrorInconsistency; // number of variables for each equation of E_incErr;
    double m_wErrorInconsistency;

    int m_nEqErrorNorm; // number of equations for error minimizing energy E_delta
    int m_nVarErrorNorm; // number of variables for each equation of E_delta;
    double m_wErrorNorm;

    int m_nEqDeformationSmootheness; // number of equations for spatial smoothing energy E_s
    int m_nVarDeformationSmootheness; // number of variables for each equation of E_d;
    double m_wDeformationSmootheness;

    int m_nEqErrorSmootheness; // number of equations for spatial smoothing energy E_s
    int m_nVarErrorSmootheness; // number of variables for each equation of E_d;
    double m_wErrorSmootheness;

    int m_nEqErrorStatistics; // number of equations for statistics over error inconsistencies
    int m_nVarErrorStatistics; // number of variables for statistics over error inconsistencies
    double m_wErrorStatistics;
    enum StatisticType {MIN,MAX,MEAN,MEDIAN};
    StatisticType m_TypeErrorStatistics;
        


    
    double m_wSum;

    double m_sigma;
    
    bool m_locallyUpdateDeformationEstimate;

    double m_exponent;
    
    double m_shearingReduction;
    bool m_linearInterpol;
    bool m_ORACLE;
    bool m_haveDeformationEstimate;

    double m_segConsisntencyWeight;

    std::vector<mxArray * > m_results;

    bool m_estDef,m_estError;

    int m_numDeformationsToEstimate,m_nCircles;
    
    double m_resolutionFactor;
    double m_ADE, m_TRE, m_dice,m_Inconsistency;
    bool m_lowResolutionEval,m_bSplineInterpol;
    bool m_filterMetricWithGradient;
    bool m_lineSearch;
    double m_spacingBasedSmoothnessReduction;
    bool m_useConstraints;
    double m_minMinJacobian,m_averageNCC;
    
public:
    CLERCIndependentDimensions(){
        m_wTransformationSimilarity=1.0;
        m_wDeformationSmootheness=1.0;
        m_wCircleNorm=1.0;
        m_wErrorNorm=1.0;
        m_wFullCircleEnergy=1.0;
        m_wSum=1.0;
        m_sigma = 10.0;
        m_linearInterpol=false;
        m_haveDeformationEstimate=false;
        //m_updatedDeformationCache = new  map< string, map <string, DeformationFieldPointerType> > ; 
        m_results = std::vector<mxArray * >(D,NULL);
        //m_updateDeformations=true;
        m_updateDeformations=false;
        m_exponent=1.0;
        m_shearingReduction = 1.0;
        m_segConsisntencyWeight = 1.0;
        m_locallyUpdateDeformationEstimate = false;
        m_ORACLE=false;
        m_nEqTransformationSimilarity=0;
        m_metric="lncc";
        m_wTransformationSymmetry=0.0;
        m_resolutionFactor=0.125;
        m_ADE=-1;
        m_TRE=-1;
        m_dice=-1;
        m_Inconsistency=-1;
        m_lowResolutionEval=false;
        m_bSplineInterpol=false;
        m_lineSearch=false;
        m_updateDeformationsGlobalSim=false;
    }

    double getADE(){return m_ADE;}
    double getTRE(){return m_TRE;}
    double getDice(){return m_dice;}
    double getInconsistency(){return m_Inconsistency;}
    double getMinJac(){return m_minMinJacobian;}
    double getAverageNCC(){ return m_averageNCC;}

    void setDeformationFilenames(FileListCacheType deformationFilenames) {m_deformationFileList = deformationFilenames;}
    void setTrueDeformationFilenames(FileListCacheType trueDeformationFilenames){m_trueDeformationFileList=trueDeformationFilenames;}
    void setLandmarkFilenames(map<string,string> landmarkFilenames){m_landmarkFileList=landmarkFilenames;}
    void setAtlasSegmentations(ImageCacheType  atlasSegmentations){m_atlasSegmentations=atlasSegmentations;}
    void setGroundTruthSegmentations(ImageCacheType  groundTruthSegmentations){m_groundTruthSegmentations = groundTruthSegmentations;}
    void setImageIDs( std::vector<string> imageIDs){m_imageIDList=imageIDs;}
    void setImages(ImageCacheType  inputImages){        m_imageList=inputImages;    };
    void setMasks(ImageCacheType *  masks){        m_maskList=masks;    };
    void setLowResEval(bool b){ m_lowResolutionEval=b;}
    void setBSplineInterpol(bool b){m_bSplineInterpol=b;}
    void setLineSearch(bool b){m_lineSearch=b;}
    void setUseConstraints(bool b){m_useConstraints=b;}
    void setUpdateDeformationsGlobalSim(bool b){ m_updateDeformationsGlobalSim=b;}
    void setROI(ImagePointerType ROI){ 
        this->m_ROI=ROI;
        m_nPixels=this->m_ROI->GetLargestPossibleRegion().GetNumberOfPixels( );
        IndexType startIndex,nullIdx;
        nullIdx.Fill(0);
        PointType startPoint;
        this->m_ROI->TransformIndexToPhysicalPoint(nullIdx,startPoint);
        (m_imageList)[(m_imageIDList)[0]]->TransformPhysicalPointToIndex(startPoint,startIndex);
        m_regionOfInterest.SetIndex(startIndex);
        m_regionOfInterest.SetSize(this->m_ROI->GetLargestPossibleRegion().GetSize());
    }

    virtual void Initialize(){
        m_numImages=m_imageIDList.size();
        m_numDeformationsToEstimate=0;
        m_nCircles=0;
        //calculate number of deformations and number of deformation circles
        for (int s = 0;s<m_numImages;++s){                            
            int source=s;
            string sourceID=(m_imageIDList)[source];
            for (int t=0;t<m_numImages;++t){
                if (t!=s){
                    int target=t;
                    string targetID=(m_imageIDList)[target];


                    bool estSourceTarget=false;
                    bool skip=false;
                    if (findDeformation(m_deformationFileList,sourceID,targetID)){
                        estSourceTarget=true;
                    }else{
                        if (findDeformation(m_trueDeformationFileList,sourceID,targetID)){
                            (m_trueDeformations)[sourceID][targetID] = ImageUtils<DeformationFieldType>::readImage(m_trueDeformationFileList[sourceID][targetID]);
                            (m_trueDeformations)[sourceID][targetID]=TransfUtils<ImageType>::linearInterpolateDeformationField( (m_trueDeformations)[sourceID][targetID],this->m_ROI,m_smoothDeformationDownsampling);
                        }
                        else
                            skip=true;
                    }
                    m_numDeformationsToEstimate+=estSourceTarget;

                    for (int i=0;i<m_numImages;++i){ 
                        if (t!=i && i!=s){
                            //define a set of 3 images
                            int intermediate=i;
                            string intermediateID=(m_imageIDList)[i];
                            
                            bool estSourceIntermediate=false;
                            if (findDeformation(m_deformationFileList,sourceID,intermediateID)){
                                estSourceIntermediate=true;
                            }else{
                                if (findDeformation(m_trueDeformationFileList,sourceID,intermediateID))
                                    {
                                        (m_trueDeformations)[sourceID][intermediateID] = ImageUtils<DeformationFieldType>::readImage(m_trueDeformationFileList[sourceID][intermediateID]);
                                        (m_trueDeformations)[sourceID][intermediateID]=TransfUtils<ImageType>::linearInterpolateDeformationField( (m_trueDeformations)[sourceID][intermediateID],this->m_ROI,m_smoothDeformationDownsampling);
                 
                                    }
                                else
                                    skip=true;
                            }

                            bool estIntermediateTarget=false;
                            if (findDeformation(m_deformationFileList,intermediateID,targetID)){
                                estIntermediateTarget=true;
                            }else{
                                if (findDeformation(m_trueDeformationFileList,intermediateID,targetID))
                                    {
                                        (m_trueDeformations)[intermediateID][targetID] = ImageUtils<DeformationFieldType>::readImage(m_trueDeformationFileList[intermediateID][targetID]);
                                        (m_trueDeformations)[intermediateID][targetID]=TransfUtils<ImageType>::linearInterpolateDeformationField( (m_trueDeformations)[intermediateID][targetID],this->m_ROI,m_smoothDeformationDownsampling);
                                    }
                                else
                                    skip = true;
                            }
                            
                            //check if any of the deformations of the loop should be estimated
                            if (! skip && (estIntermediateTarget || estSourceTarget || estSourceIntermediate)){
                                ++m_nCircles;
                            }
                        }
                    }
                }
            }
        }
        LOGV(1)<<VAR(m_nCircles)<<" "<<VAR(m_numDeformationsToEstimate)<<endl;
       
     

      
        
        //normalize weights according to number of deformation pairs/circles
        m_wFullCircleEnergy /=m_nCircles;
        m_wCircleNorm/=m_nCircles;
        m_wErrorInconsistency/=m_nCircles;
        m_wDeformationSmootheness/=m_numDeformationsToEstimate;
        m_wErrorSmootheness/=m_numDeformationsToEstimate;
        m_wErrorNorm/=m_numDeformationsToEstimate;
        m_wErrorStatistics/=m_numDeformationsToEstimate;
        m_wTransformationSimilarity/=m_numDeformationsToEstimate;
        m_wTransformationSymmetry/=m_numDeformationsToEstimate;
        
      

        //compute errors, weights, and so on and so forth?
        DoALot();

    }
    
    void setOracle(bool o){m_ORACLE=o;}
    void setWeightFullCircleEnergy(double w){m_wFullCircleEnergy=w;}
    void setWeightTransformationSimilarity(double w,bool update=false){
        m_wTransformationSimilarity=w;  
        if (update && m_nEqTransformationSimilarity){
            m_wTransformationSimilarity/=m_numDeformationsToEstimate;
        }
    }
    void setWeightTransformationSymmetry(double w, bool update=false){
        m_wTransformationSymmetry=w;  
        if (update && m_nEqTransformationSymmetry){
            m_wTransformationSymmetry/=m_numDeformationsToEstimate;
        }
    }
  
    void setWeightDeformationSmootheness(double w){m_wDeformationSmootheness=w;}
    void setWeightErrorSmootheness(double w){m_wErrorSmootheness=w;}
    void setWeightErrorNorm(double w){m_wErrorNorm=w;}
    void setWeightErrorStatistics(double w){m_wErrorStatistics=w;}
    void setWeightCircleNorm(double w){m_wCircleNorm=w;}
    void setWeightSum(double w){m_wSum=w;}
    void setWeightInconsistencyError(double w){m_wErrorInconsistency=w;}
    void setLinearInterpol(bool i){m_linearInterpol=i;}
    void setSigma(double s){m_sigma=s;}

    void setLocalWeightExp(double e){ m_exponent=e;}
    void setShearingReduction(double r){m_shearingReduction = r;}
    void setScalingFactorForConsistentSegmentation(double scalingFactorForConsistentSegmentation){ m_segConsisntencyWeight = scalingFactorForConsistentSegmentation;}
    void setUpdateDeformations(bool b){m_updateDeformations=b;}
    void setLocallyUpdateDeformations(bool s){m_locallyUpdateDeformationEstimate=s;}
    void setSmoothDeformationDownsampling(bool s){m_smoothDeformationDownsampling=s;}
    void setMetric(string m){m_metric=m;}
    void setFilterMetricWithGradient(bool b){m_filterMetricWithGradient=b;}
    virtual void createSystem(){
        //set up ROI

      
        int interpolationFactor;
        if (m_linearInterpol){
            interpolationFactor = pow(2,D); //linear interpolation
        }
        else
            interpolationFactor = 1 ; //NNinterpolation;
        

        //compute number of variables
        m_nEqFullCircleEnergy  = (m_wFullCircleEnergy>0.0)* internalD * m_nPixels *  m_numImages*(m_numImages-1)*(m_numImages-2); //there is one equation for each component of every pixel of every triple of images
        m_nVarFullCircleEnergy = 2*(interpolationFactor+2); //two variables per uniqe pair in the triple (2*2), plus linear interpolation for the third pair (2*2^D)
        m_nEqCircleNorm =  (m_wCircleNorm>0.0)* m_nPixels * internalD * m_nCircles;//m_numImages*(m_numImages-1)*(m_numImages-2); //again all components of all triples
        m_nVarCircleNorm = interpolationFactor+2 ; // only one/2^D variables per pair
        m_nEqErrorInconsistency =  (m_wErrorInconsistency>0.0)* m_nPixels * internalD * m_numImages*(m_numImages-1)*(m_numImages-2); //again all components of all triples
        m_nVarErrorInconsistency = interpolationFactor+2; // only one/2^D variables per pair
        m_nEqDeformationSmootheness =  (m_wDeformationSmootheness>0.0)* D* m_nPixels * internalD * m_numImages*(m_numImages-1); //every pixel in each registration has D neighbors (in one direction), and each component separately
        m_nVarDeformationSmootheness = 3; //for piecewise linear regularization, 2 for piecewise constant
        m_nEqErrorSmootheness =  (m_wErrorSmootheness>0.0)* D* m_nPixels * internalD * m_numImages*(m_numImages-1); //every pixel in each registration has D neighbors (in one direction), and each component separately
        m_nVarErrorSmootheness = 3; //for piecewise linear regularization, 2 for piecewise constant
        m_nEqErrorNorm = (m_wErrorNorm>0.0)*  m_nPixels * internalD * m_numImages*(m_numImages-1); //every error at every location in each image pair
        m_nVarErrorNorm = 1;
        m_nEqErrorStatistics = (m_wErrorStatistics>0.0)*  m_nPixels * internalD * m_numImages*(m_numImages-1); //every error at every location in each image pair
        m_nVarErrorStatistics = 1;
        m_nEqTransformationSimilarity =  (m_wTransformationSimilarity>0.0)*m_nPixels * internalD * m_numDeformationsToEstimate;//m_numImages*(m_numImages-1); //same as ErrorNorm
        m_nVarTransformationSimilarity= 1;
        m_nEqTransformationSymmetry =  (m_wTransformationSymmetry>0.0)*m_nPixels * internalD * m_numImages*(m_numImages-1); //same as ErrorNorm
        m_nVarTransformationSymmetry= interpolationFactor;
        int m_nEqSUM=(m_wSum>0.0)*m_nPixels * internalD * m_numImages*(m_numImages-1);
        if (m_nEqSUM)
            m_wSum/=m_nEqSUM;
        int m_nVarSUM=2;

        m_nEqs= m_nEqTransformationSymmetry+  m_nEqErrorStatistics+  m_nEqFullCircleEnergy + m_nEqCircleNorm+ m_nEqDeformationSmootheness + m_nEqErrorNorm+ m_nEqTransformationSimilarity + m_nEqSUM +  m_nEqErrorSmootheness + m_nEqErrorInconsistency; // total number of equations
        
        m_estError= m_nEqFullCircleEnergy ||  m_nEqErrorNorm || m_nEqSUM||m_nEqErrorInconsistency ;
        m_estDef = m_nEqFullCircleEnergy || m_nEqTransformationSimilarity ||  m_nEqDeformationSmootheness  ||  m_nEqCircleNorm || m_nEqSUM  ||  m_nEqErrorStatistics;
        m_nVars= m_numImages*(m_numImages-1)*m_nPixels*internalD *(m_estError + m_estDef); // total number of free variables (error and deformation)
        
        m_nNonZeroes=m_nEqTransformationSymmetry*m_nVarTransformationSymmetry+ m_nEqErrorStatistics+ m_nEqErrorSmootheness*m_nVarErrorSmootheness +m_nEqFullCircleEnergy *m_nVarFullCircleEnergy + m_nEqCircleNorm * m_nVarCircleNorm + m_nEqDeformationSmootheness*m_nVarDeformationSmootheness + m_nEqErrorNorm*m_nVarErrorNorm + m_nEqTransformationSimilarity*m_nVarTransformationSimilarity + m_nEqSUM*m_nVarSUM + m_nVarErrorInconsistency*m_nEqErrorInconsistency; //maximum number of non-zeros


        LOGV(1)<<"Creating equation system.."<<endl;
        LOGV(1)<<VAR(m_numImages)<<" "<<VAR(m_nPixels)<<" "<<VAR(m_nEqs)<<" "<<VAR(m_nVars)<<" "<<VAR(m_nNonZeroes)<<endl;
        LOGV(1)<<VAR(  m_wTransformationSimilarity)<<" "<<VAR(        m_wDeformationSmootheness)<<" "<<VAR(m_wCircleNorm)<<" "<<VAR(m_wErrorNorm)<<" "<<VAR(m_wErrorStatistics)<<" "<<VAR(m_wFullCircleEnergy)<<" "<<VAR(m_wSum)<<" "<<VAR(m_wErrorSmootheness)<<" "<<VAR(m_sigma)<<" "<<VAR(m_locallyUpdateDeformationEstimate)<<" "<<VAR(m_exponent)<<endl;
        double totalInconsistency = 0.0;
        int totalCount = 0;

        bool haveLocalWeights=false;


        for (unsigned int d = 0; d< D; ++d){
#ifdef SEPENGINE       
            if (!(this->m_ep = engOpen("matlab -nodesktop -nodisplay -nosplash -nojvm"))) {
                fprintf(stderr, "\nCan't start MATLAB engine\n");
                exit(EXIT_FAILURE);
            }
#endif
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
            std::fill(x,x+m_nNonZeroes,-1);
            double * y=( double *)mxGetData(mxY);
            std::fill(y,y+m_nNonZeroes,m_nVars);
            double * v=( double *)mxGetData(mxV);
            double * b=mxGetPr(mxB);
            std::fill(b,b+m_nEqs,-999999);

            double * init=mxGetPr(mxInit);
            double * lb=mxGetPr(mxLowerBound);
            std::fill(lb,lb+m_nVars,-200);
            double * ub=mxGetPr(mxUpperBound);
            std::fill(ub,ub+m_nVars,200);
            LOGV(1)<<"creating"<<VAR(d)<<endl;
     


            char buffer[256+1];
            buffer[256] = '\0';
            engOutputBuffer(this->m_ep, buffer, 256);
      
            //attention matlab index convention?!?
            long int eq = 1;
            long int c=0;
        
            
            computeTripletEnergies( x,  y, v,  b, c,  eq,d);
            computePairwiseEnergiesAndBounds( x,  y, v,  b, init, lb, ub, c,  eq,d);
            LOGV(1)<<VAR(eq)<<" "<<VAR(c)<<endl;

            if (m_useConstraints){
                int nEqConstraints=3*D*m_numDeformationsToEstimate*m_nPixels;
                int nNonZeroConstraints=2*nEqConstraints;
                
                mxArray *mxX2=mxCreateDoubleMatrix((mwSize)nNonZeroConstraints,1,mxREAL);
                mxArray *mxY2=mxCreateDoubleMatrix((mwSize)nNonZeroConstraints,1,mxREAL);
                mxArray *mxV2=mxCreateDoubleMatrix((mwSize)nNonZeroConstraints,1,mxREAL);
                mxArray *mxB2=mxCreateDoubleMatrix((mwSize)nEqConstraints,1,mxREAL);
                double * x2=( double *)mxGetData(mxX2);
                std::fill(x2,x2+nNonZeroConstraints,0);
                double * y2=( double *)mxGetData(mxY2);
                std::fill(y2,y2+nNonZeroConstraints,m_nVars);
                double * v2=( double *)mxGetData(mxV2);
                double * b2=mxGetPr(mxB2);
                std::fill(b2,b2+nEqConstraints,-999999);


                computeConstraints( x2,y2,v2,b2,d);
                //put variables into workspace and immediately destroy them
                engPutVariable(this->m_ep,"xCord2",mxX2);
                mxDestroyArray(mxX2);
                engPutVariable(this->m_ep,"yCord2",mxY2);
                mxDestroyArray(mxY2);
                engPutVariable(this->m_ep,"val2",mxV2);
                mxDestroyArray(mxV2);
                engPutVariable(this->m_ep,"b2",mxB2);
                mxDestroyArray(mxB2);

                engEvalString(this->m_ep,"nEq=sum(b2>0)");
                engEvalString(this->m_ep,"nNz=sum(xCord2>0)");
                engEvalString(this->m_ep,"xCord2=xCord2(1:nNz)");
                engEvalString(this->m_ep,"yCord2=yCord2(1:nNz)");
                engEvalString(this->m_ep,"val2=val2(1:nNz)");
                engEvalString(this->m_ep,"b2=b2(1:nEq)");
            }

            

            //put variables into workspace and immediately destroy them
            engPutVariable(this->m_ep,"xCord",mxX);
            mxDestroyArray(mxX);
            engPutVariable(this->m_ep,"yCord",mxY);
            mxDestroyArray(mxY);
            engPutVariable(this->m_ep,"val",mxV);
            mxDestroyArray(mxV);
            engPutVariable(this->m_ep,"b",mxB);
            mxDestroyArray(mxB);

            if (1){
                ostringstream nEqs;
                nEqs<<"nEq="<<eq<<";";
                engEvalString(this->m_ep,nEqs.str().c_str());
                ostringstream nNz;
                nNz<<"nNz="<<c<<";";
                engEvalString(this->m_ep,nNz.str().c_str());
                engEvalString(this->m_ep,"xCord=xCord(1:nNz)");
                engEvalString(this->m_ep,"yCord=yCord(1:nNz)");
                engEvalString(this->m_ep,"val=val(1:nNz)");
                engEvalString(this->m_ep,"b=b(1:nEq-1)");
            }

            LOGV(1)<<"Creating sparse matrix"<<endl;

            //transform indexing of variables to be 1..nVariables
            engEvalString(this->m_ep,"oldCode=unique(sort(yCord));newCode=1:size(oldCode,1);newCode=newCode'; [a1 b1]=ismember(yCord,oldCode);yCord=newCode(b1(a1));");
            engEvalString(this->m_ep,"A=sparse(xCord,yCord,val);" );
            LOGV(1)<<"done, cleaning up"<<endl;
            //clear unnneeded variables from matlab workspace
            engEvalString(this->m_ep,"clear xCord yCord val b1 a1;" );
            
            if (m_useConstraints){
                engEvalString(this->m_ep,"[a1 b1]=ismember(yCord2,oldCode);yCord2=newCode(b1(a1));");
                engEvalString(this->m_ep,"C=sparse(xCord2,yCord2,val2);" );
                engEvalString(this->m_ep,"clear xCord2 yCord2 val2 b1 a1;" );
                engEvalString(this->m_ep, "size(C)");
                LOGI(2,printf("size(C) %s", buffer+2));
                engEvalString(this->m_ep, "size(b2)");
                LOGI(2,printf("size(b2) %s", buffer+2));
            }

            engPutVariable(this->m_ep,"init",mxInit);
            mxDestroyArray(mxInit);
            this->haveInit=true;
            LOGV(1)<<"Solving "<<VAR(d)<<endl;
            if (1){
                engEvalString(this->m_ep, "lb=[-200000*ones(size(A,2),1)];");
                engEvalString(this->m_ep, "ub=[200000*ones(size(A,2),1);]");
                engEvalString(this->m_ep, "init=init(1:size(A,2));");
            }else{
                engPutVariable(this->m_ep,"lb",mxLowerBound);
                engPutVariable(this->m_ep,"ub",mxUpperBound);
            }
            mxDestroyArray(mxLowerBound);
            mxDestroyArray(mxUpperBound);
            LOGI(6,engEvalString(this->m_ep,"save('sparse.mat');" ));
            engEvalString(this->m_ep, "norm(A*init-b)^2");
            LOGI(2,printf("initialisation residual %s", buffer+2));

            
            if (1){
                engEvalString(this->m_ep, "options=optimset(optimset('lsqlin'),'Display','iter','TolFun',1e-54,'LargeScale','on');");//,'Algorithm','active-set' );");
                //solve using trust region method
                if (m_useConstraints){
#if 1

#if 1
                    engEvalString(this->m_ep, "tic;addpath('/scratch_net/ouroboros/gasst/progs/cvx/');");
                    engEvalString(this->m_ep, "cvx_startup");
                    engEvalString(this->m_ep, "cvx_solver sedumi");
                    engEvalString(this->m_ep, "n = size(A,2); ");
                    engEvalString(this->m_ep, "cvx_begin ");
                    engEvalString(this->m_ep, "   variable x(n) ");
                    engEvalString(this->m_ep, "   minimize( norm(A*x-b) ) ");
                    engEvalString(this->m_ep, "   subject to ");
                    //engEvalString(this->m_ep, "      lb <= x <= ub ");
                    engEvalString(this->m_ep, "       C*x <= b2 ");
                    engEvalString(this->m_ep, "cvx_end ");
                    engEvalString(this->m_ep, "t=toc; ");
#else
                    engEvalString(this->m_ep, "options=   optimset('Algorithm','interior-point-convex','Display','iter');" );
                    TIME(engEvalString(this->m_ep, "tic;   x = quadprog(2*A'*A,-2*A'*b,C,b2,[],[],lb,ub,[],options);t=toc;"));
                    LOGI(2,printf("%s", buffer+2));
#endif
#else
                    TIME(engEvalString(this->m_ep, "tic;[x resnorm residual flag  output lambda] =lsqlin(A,b,C,b2,[],[],lb,ub,init,options);t=toc;"));
#endif
                }else{
                    TIME(engEvalString(this->m_ep, "tic;[x resnorm residual flag  output lambda] =lsqlin(A,b,[],[],[],[],lb,ub,init,options);t=toc;"));
                    mxArray * flag=engGetVariable(this->m_ep,"flag");
                    if (flag !=NULL){
                        mxDestroyArray(flag);
                    }else{
                        LOG<<"LSQLIN large scale failed, trying medium scale algorithm"<<endl;
                        TIME(engEvalString(this->m_ep, "tic;[x resnorm residual flag output lambda] =lsqlin(A,b,[],[],[],[],[],[],[]);t=toc"));
                    }
                       
                    LOGI(1,printf("%s", buffer+2));
                    engEvalString(this->m_ep, " resnorm");
                    LOGI(1,printf("%s", buffer+2));
                    engEvalString(this->m_ep, "output");
                    LOGI(2,printf("%s", buffer+2));
                }



                mxArray * time=engGetVariable(this->m_ep,"t");
                double * t = ( double *) mxGetData(time);
                LOGADDTIME((int)(t[0]));
                mxDestroyArray(time);
                //solve using active set method (backslash)
             
            }else{
                //solve using pseudo inverse
                //TIME(engEvalString(this->m_ep, "tic;x = pinv(full(A))*b;toc"));
            }

            //backtransform indexing
            engEvalString(this->m_ep, "newX=zeros(max(oldCode),1);newX(oldCode)=x;x=newX;");


            LOGI(6,engEvalString(this->m_ep,"save('test.mat');" ));
            if ((m_results[d] = engGetVariable(this->m_ep,"x")) == NULL)
                printf("something went wrong when getting the variable.\n Result is probably wrong. \n");
            engEvalString(this->m_ep,"clearvars" );

            
#ifdef SEPENGINE
            engClose(this->m_ep);
#endif
        }//dimensions

    }
    virtual void solve(){}

    virtual void storeResult(string directory){
        //std::vector<double> result(m_nVars);
        std::vector<double*> rData(D);
        for (int d= 0; d<D ; ++d){
            rData[d]=mxGetPr(this->m_results[d]);
        }

        ImagePointerType mask;
        
        for (int s = 0;s<m_numImages;++s){
            for (int t=0;t<m_numImages;++t){
                if (s!=t){
                    string sourceID=(this->m_imageIDList)[s];
                    string targetID = (this->m_imageIDList)[t];
                    //slightly(!!!) stupid creation of empty image
                    DeformationFieldPointerType estimatedError=TransfUtils<ImageType>::createEmpty(this->m_ROI);
                    DeformationFieldIterator itErr(estimatedError,estimatedError->GetLargestPossibleRegion());
                    DeformationFieldPointerType estimatedDeform=TransfUtils<ImageType>::createEmpty(this->m_ROI);
                    DeformationFieldIterator itDef(estimatedDeform,estimatedDeform->GetLargestPossibleRegion());
                    itErr.GoToBegin();
                    itDef.GoToBegin();

                    mask=TransfUtils<ImageType>::createEmptyImage(estimatedDeform);
                    mask->FillBuffer(0);
                    typename ImageType::SizeType size=mask->GetLargestPossibleRegion().GetSize();
                    IndexType offset;
                    double fraction=0.9;
                    for (int d=0;d<D;++d){
                        offset[d]=(1.0-fraction)/2*size[d];
                        size[d]=fraction*size[d];
                    }
                    
                    typename ImageType::RegionType region;
                    region.SetSize(size);
                    region.SetIndex(offset);
                    LOGV(6)<<VAR(region)<<endl;
                    ImageUtils<ImageType>::setRegion(mask,region,1);
                    
                    
                    DeformationFieldPointerType oldDeformation;
                    
                    if ((m_downSampledDeformationCache)[sourceID][targetID].IsNotNull()){
                        oldDeformation=(m_downSampledDeformationCache)[sourceID][targetID];

                        DeformationFieldIterator origIt(oldDeformation,oldDeformation->GetLargestPossibleRegion());
                        origIt.GoToBegin();

                        DeformationFieldIterator trueIt;
                        if ((m_trueDeformations)[sourceID][targetID].IsNotNull()){
                            trueIt=DeformationFieldIterator((m_trueDeformations)[sourceID][targetID],(m_trueDeformations)[sourceID][targetID]->GetLargestPossibleRegion());
                            trueIt.GoToBegin();
                        }

                   
                        for (int p=0;!itErr.IsAtEnd();++itErr,++itDef,++origIt){
                            //get solution of eqn system
                            DeformationType estimatedError,estimatedDeformation,originalDeformation;
                            IndexType idx = itErr.GetIndex();
                            originalDeformation=origIt.Get();
                            for (unsigned int d=0;d<D;++d,++p){
                                // minus 1 to correct for matlab indexing
                                if (m_estError){
                                    estimatedError[d]=rData[d][edgeNumError(s,t,idx,d)-1];
                                }
                                if (m_estDef)
                                    estimatedDeformation[d]=rData[d][edgeNumDeformation(s,t,idx,d)-1];
                            
                                if (!m_estError && m_estDef){
                                    estimatedError[d]= originalDeformation[d]-estimatedDeformation[d];
                                } else if (!m_estDef && m_estError)
                                    estimatedDeformation[d]= originalDeformation[d]-estimatedError[d];
                            }
                            itErr.Set(estimatedError);
                            itDef.Set(estimatedDeformation);


                            if (mask->GetPixel(itErr.GetIndex()) && (m_trueDeformations)[sourceID][targetID].IsNotNull()){
                                DeformationType trueErr = origIt.Get()-trueIt.Get();
                                DeformationType estimatedDiffErr = estimatedDeformation - trueIt.Get();
                                //LOGV(5)<<VAR(trueErr.GetNorm())<<" "<<VAR(estimatedError.GetNorm())<<endl;
                                //LOGV(5)<<VAR(trueErr.GetNorm())<<" "<<VAR(estimatedDiffErr.GetNorm())<<endl;
                                LOGV(5)<<VAR(trueErr)<<" "<<VAR(estimatedError)<<endl;
                                LOGV(5)<<VAR(trueErr)<<" "<<VAR(estimatedDiffErr)<<endl;
                           
                                ++trueIt;
                            }
                        }

#if 0
                        FloatImagePointerType localSim=getLocalWeightMap(estimatedDeform,(m_imageList)[targetID],(m_imageList)[sourceID]);
                        origIt.GoToBegin();
                        DeformationFieldIterator newIt(estimatedDeform,estimatedDeform->GetLargestPossibleRegion());
                        newIt.GoToBegin();
                        FloatImageIterator oldLocalSim((m_pairwiseLocalWeightMaps)[s][t],(m_pairwiseLocalWeightMaps)[s][t]->GetLargestPossibleRegion());
                        oldLocalSim.GoToBegin();
                        FloatImageIterator newLocalSim(localSim,localSim->GetLargestPossibleRegion());
                        newLocalSim.GoToBegin();
                        for (;!newLocalSim.IsAtEnd();++origIt,++newIt,++oldLocalSim,++newLocalSim){
                            DeformationType newDef;
                            double newSim=newLocalSim.Get();
                            double oldSim=oldLocalSim.Get();

#if 0                       
                            newDef=newIt.Get()*newSim+origIt.Get()*oldSim;
                            newDef=newDef*0.5*(newSim+oldSim);
                            newIt.Set(newDef);
                            oldLocalSim.Set(0.5*(newSim+oldSim));
#else
                        
                            //if (newSim<oldLocalSim.Get()){
                            if (! (oldSim==0.0) && newSim/oldSim < 0.7){
                                //origIt.Set(newIt.Get());
                                newIt.Set(origIt.Get());

                            }else{
                                oldLocalSim.Set(newSim);
                            }

#endif
                        }
                    

#endif                    



                        if ((m_trueDeformations)[sourceID][targetID].IsNotNull()){
                            mask=TransfUtils<ImageType>::createEmptyImage(estimatedDeform);
                            mask->FillBuffer(0);
                            typename ImageType::SizeType size=mask->GetLargestPossibleRegion().GetSize();
                            IndexType offset;
                            double fraction=0.9;
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

                            if (false){
                                double newError=TransfUtils<ImageType>::computeDeformationNormMask(TransfUtils<ImageType>::subtract(estimatedDeform,(m_trueDeformations)[sourceID][targetID]),mask);
                                //double newError=TransfUtils<ImageType>::computeDeformationNorm(TransfUtils<ImageType>::subtract(estimatedDeform,(m_trueDeformations)[sourceID][targetID]),1);
                                //mask->FillBuffer(1);
                                //mask = TransfUtils<ImageType>::warpImage(mask,(m_downSampledDeformationCache)[sourceID][targetID]);
                                double oldError=TransfUtils<ImageType>::computeDeformationNormMask(TransfUtils<ImageType>::subtract((m_downSampledDeformationCache)[sourceID][targetID],(m_trueDeformations)[sourceID][targetID]),mask);
                                //double oldError=TransfUtils<ImageType>::computeDeformationNorm(TransfUtils<ImageType>::subtract((m_downSampledDeformationCache)[sourceID][targetID],(m_trueDeformations)[sourceID][targetID]),1);
                                LOGV(1)<<VAR(s)<<" "<<VAR(t)<<" "<<VAR(oldError)<<" "<<VAR(newError)<<endl;
                            }
                        }
                      
                        m_estimatedErrors[sourceID][targetID]=estimatedDeform;//Error;
                      
                   
                    }else{
                        (m_downSampledDeformationCache)[sourceID][targetID]=NULL;
                        (m_updatedDeformationCache)[sourceID][targetID]=NULL;
                    }
                }
            }
        }
      
        for (int d= 0; d<D ; ++d){
            mxDestroyArray(this->m_results[d]);
        }
       
       
    }

    std::vector<double> getResult(){
        std::vector<double> result(m_nVars);
        return result;
    }

public:
  
   

protected:
    //return fortlaufende number of pairs n1,n2, 0..(n*(n-1)-1)
    inline long int edgeNum(int n1,int n2){ return ((n1)*(m_numImages-1) + n2 - (n2>n1));}
 

    inline long int edgeNumDeformation(int n1,int n2,IndexType idx, int d){ 
        long int offset = this->m_ROI->ComputeOffset(idx);
        //return offset*internalD+edgeNum(n1,n2)*m_nPixels*internalD + 1 ;
        double result= internalD*(edgeNum(n1,n2)*m_nPixels+offset) + 1 ;
        if (result > m_nVars){
            LOG<<VAR(result)<<" "<<VAR(n1)<<" "<<VAR(n2)<<" "<<VAR(idx)<<endl;
        }
        return result;
    }

    inline long int edgeNumError(int n1,int n2,IndexType idx, int d){ 
        double result = m_estDef*m_nPixels*internalD*(m_numImages-1)*(m_numImages) + edgeNumDeformation(n1,n2,idx,d);
        if (result>m_nVars){
            LOG<<m_nVars<<" "<<VAR(result)<<" "<<VAR(n1)<<" "<<VAR(n2)<<" "<<VAR(idx)<<endl;
            sqrt(-1);
        }
        return result;
    }


    //compose 3 deformations. order is left-to-right
    DeformationFieldPointerType composeDeformations(DeformationFieldPointerType d1,DeformationFieldPointerType d2,DeformationFieldPointerType d3){
        return TransfUtils<ImageType>::composeDeformations(d3,TransfUtils<ImageType>::composeDeformations(d2,d1));

    }


    inline bool getLinearNeighbors(const DeformationFieldPointerType def, const PointType & point, std::vector<std::pair<IndexType,double> > & neighbors){
        bool inside=true;
        neighbors= std::vector<std::pair<IndexType,double> >(pow(2,D));
        int nNeighbors=0;
        IndexType idx1;
        inside = def->TransformPhysicalPointToIndex(point,idx1);
#if 0
        if ( !def->GetLargestPossibleRegion().IsInside(idx1))
            return false;
        for (int d=0;d<D;++d){
            double orig= def->GetOrigin()[d] ;
            double size= def->GetLargestPossibleRegion().GetSize()[d]*def->GetSpacing()[d];
            LOGV(8)<<d<<" "<<point[d]<<" "<<orig<<" "<<size<<endl;
            if (point[d] < def->GetOrigin()[d] || point[d] >= orig+size-0.0001 ){
                inside = false;
                break;
            }
        }
#endif   
        LOGV(8)<<VAR(point)<<" "<<VAR(idx1)<<" "<<VAR(inside)<<" "<<VAR(def->GetOrigin())<<endl;
        
        if (!inside) return false;
        PointType pt1;
        def->TransformIndexToPhysicalPoint(idx1,pt1);
        DeformationType dist=point-pt1;
        if (inside){
            neighbors[nNeighbors++]=std::make_pair(idx1,getWeight(dist,def->GetSpacing()));
        }
        OffsetType off;
        off.Fill(0);
        double sum=0.0;
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
                //LOGV(1)<<VAR(def->GetLargestPossibleRegion().GetSize())<<" "<<VAR(idx)<<endl;
                double w=getWeight(delta,def->GetSpacing());
                sum+=w;
                neighbors[nNeighbors++]=std::make_pair(idx,w);
                inside=true;
            }
        }
        neighbors.resize(nNeighbors);
        return inside;
    }

    inline bool getGaussianNeighbors(const DeformationFieldPointerType def, const PointType & point, std::vector<std::pair<IndexType,double> > & neighbors, double sigma){
        bool inside=true;
        int nNeighbors=0;
        IndexType idx1;
        int pixelSigma=max(1.0,sigma/def->GetSpacing()[0]+0.5);
        sigma=max(0.5,sigma);
        neighbors= std::vector<std::pair<IndexType,double> >(pow(pixelSigma*2,D));

        inside = def->TransformPhysicalPointToIndex(point,idx1);

        LOGV(8)<<VAR(point)<<" "<<VAR(idx1)<<" "<<VAR(inside)<<" "<<VAR(def->GetOrigin())<<" "<<VAR(sigma)<<" "<<VAR(pixelSigma)<<endl;
        
        if (!inside) return false;
        PointType pt1;
        def->TransformIndexToPhysicalPoint(idx1,pt1);
        //get bottom left index
        typename ImageUtils<ImageType>::ContinuousIndexType cIdx;
        def->TransformPhysicalPointToContinuousIndex(pt1,cIdx);
        for (int d=0;d<D;++d)
            idx1[d]=int(cIdx[d]);

        OffsetType off;
        off.Fill(-pixelSigma+1);
        double sum=0.0;
        for (int i=0;i<pow(2*pixelSigma,D);++i){
            if (i>0){
                for (int d=0;d<D;++d){
                    off[d]++;
                    if (off[d]>pixelSigma){
                        off[d]=-pixelSigma+1;
                    }else{
                        break;
                    }
                }
            }
            IndexType idx=idx1+off;
            if (def->GetLargestPossibleRegion().IsInside(idx)){
                PointType pt;
                def->TransformIndexToPhysicalPoint(idx,pt);
                DeformationType delta=point-pt;
                double w=exp(- delta.GetSquaredNorm()/(sigma*sigma));
                LOGV(8)<<VAR(nNeighbors)<<" "<<VAR(idx)<<" "<<VAR(off)<<" "<<VAR(w)<<" "<<VAR(point)<<" "<<VAR(pt)<<" "<<VAR(delta.GetSquaredNorm())<<endl;
                sum+=w;
                neighbors[nNeighbors++]=std::make_pair(idx,w);
                inside=true;
            }
        }
        neighbors.resize(nNeighbors);
        for (int i=0;i<neighbors.size();++i){
            neighbors[i].second/=sum;
        }
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

    void computeTripletEnergies(double * x, double * y, double * v, double * b, long int &c,long int & eq, unsigned int d){
        double maxAbsDisplacement=0.0;
        m_pairwiseInconsistencyStatistics=map<int, map <int,  GaussEstimatorType > >();
        //0=min,1=mean,2=max,3=median,-1=off,4=gauss;
        int accumulate=4;
        double manualResidual=0.0;
        for (int s = 0;s<m_numImages;++s){                            
            int source=s;
            string sourceID=(m_imageIDList)[source];
            if (m_wErrorStatistics>0 && m_pairwiseInconsistencyStatistics.find(s)==m_pairwiseInconsistencyStatistics.end())
                m_pairwiseInconsistencyStatistics[s]=map <int,  GaussEstimatorType >();

            for (int t=0;t<m_numImages;++t){
                if (t!=s){
                    int target=t;
                    string targetID=(m_imageIDList)[target];
                    DeformationFieldPointerType dSourceTarget;
                    bool estSourceTarget=false;
                    bool skip=false;
                    if ((m_downSampledDeformationCache)[sourceID][targetID].IsNotNull()){
                        dSourceTarget=(m_downSampledDeformationCache)[sourceID][targetID];
                        estSourceTarget=true;
                    }else{
                        if ((m_trueDeformations)[sourceID][targetID].IsNotNull())
                            dSourceTarget=(m_trueDeformations)[sourceID][targetID];
                        else
                            skip=true;
                    }
                    
                    if (m_wErrorStatistics>0.0){
                        if (m_pairwiseInconsistencyStatistics[s].find(t)==m_pairwiseInconsistencyStatistics[s].end())
                            m_pairwiseInconsistencyStatistics[s][t]=GaussEstimatorType();
                        
                        //m_pairwiseInconsistencyStatistics[s][t].addImage(TransfUtils<ImageType>::getComponent(dSourceTarget,d));
                    }

                    
                    //triplet energies
                    for (int i=0;i<m_numImages;++i){ 
                        if (t!=i && i!=s){
                            //define a set of 3 images
                            int intermediate=i;
                            string intermediateID=(m_imageIDList)[i];
                            DeformationFieldPointerType dSourceIntermediate;
                            
                            bool estSourceIntermediate=false;
                            if ((m_downSampledDeformationCache)[sourceID][intermediateID].IsNotNull()){
                                dSourceIntermediate=(m_downSampledDeformationCache)[sourceID][intermediateID];
                                estSourceIntermediate=true;
                            }else{
                                if ((m_trueDeformations)[sourceID][intermediateID].IsNotNull())
                                    dSourceIntermediate=(m_trueDeformations)[sourceID][intermediateID];
                                else
                                    skip=true;
                            }

                            DeformationFieldPointerType dIntermediateTarget;
                            
                            bool estIntermediateTarget=false;
                            if ((m_downSampledDeformationCache)[intermediateID][targetID].IsNotNull()){
                                dIntermediateTarget=(m_downSampledDeformationCache)[intermediateID][targetID];
                                estIntermediateTarget=true;
                            }else{
                                if ((m_trueDeformations)[intermediateID][targetID].IsNotNull())
                                    dIntermediateTarget=(m_trueDeformations)[intermediateID][targetID];
                                else
                                    skip = true;
                            }
                            

                            //skip also if none of the registrations in the circle are to be re-estimated
                            skip= skip || (!(estIntermediateTarget || estSourceTarget || estSourceIntermediate));
                            LOGV(3)<<VAR(skip)<<" "<<VAR(estIntermediateTarget)<<" "<<VAR(estSourceTarget)<<" "<<VAR(estSourceIntermediate)<<" "<<VAR(sourceID)<<" "<<VAR(targetID)<<" "<<VAR(intermediateID)<<endl;
                            if ( ! skip ){

                                //use updated deform for constructing circle
                                if (m_ORACLE){
                                    dIntermediateTarget=(m_trueDeformations)[intermediateID][targetID];
                                }else if (estIntermediateTarget && m_haveDeformationEstimate && (m_updatedDeformationCache)[intermediateID][targetID].IsNotNull()){
                                    dIntermediateTarget=(m_updatedDeformationCache)[intermediateID][targetID];
                                }
                                

                                DeformationFieldPointerType indirectDeform = TransfUtils<ImageType>::composeDeformations(dIntermediateTarget,dSourceIntermediate);

                              
                                DeformationFieldPointerType difference = TransfUtils<ImageType>::subtract(indirectDeform,dSourceTarget);

                                //compute indirect deform
                                //DeformationFieldPointerType indirectDeform = TransfUtils<ImageType>::composeDeformations(dSourceIntermediate,dIntermediateTarget);
                                //compute difference of direct and indirect deform
                            
                                FloatImagePointerType directionalDifference = TransfUtils<ImageType,float,double,double>::getComponent(difference,d);
                                FloatImagePointerType directionalDeform = TransfUtils<ImageType,float,double,double>::getComponent(indirectDeform,d);
                                FloatImagePointerType diffNorm = TransfUtils<ImageType,float,double,double>::computeLocalDeformationNorm(difference);
                                if (m_wErrorStatistics>0.0){
                                    //check if all accumulators exist
                                    if (m_pairwiseInconsistencyStatistics.find(i)==m_pairwiseInconsistencyStatistics.end())
                                        m_pairwiseInconsistencyStatistics[i]=map <int,  GaussEstimatorType >();
                                    if (m_pairwiseInconsistencyStatistics[i].find(t)==m_pairwiseInconsistencyStatistics[i].end())
                                        m_pairwiseInconsistencyStatistics[i][t]=GaussEstimatorType();
                                
                                    //m_pairwiseInconsistencyStatistics[s][t].addImage(directionalDifference);
                                    //m_pairwiseInconsistencyStatistics[s][t].addImage(directionalDeform);
                                    m_pairwiseInconsistencyStatistics[s][t].addImage(diffNorm);
                                    //m_pairwiseInconsistencyStatistics[i][t].addImage(diffNorm);
                                    //m_pairwiseInconsistencyStatistics[i][t].addImage(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(directionalDifference,-1));
                                }

                              
                            

                            

                            
                         

                                //compute norm
                                DeformationFieldIterator it(difference,difference->GetLargestPossibleRegion());
                                it.GoToBegin();
                            
                                DeformationFieldIterator trueIt;
                                if ((m_trueDeformations)[intermediateID][targetID].IsNotNull()){
                                    trueIt=DeformationFieldIterator((m_trueDeformations)[intermediateID][targetID],(m_trueDeformations)[intermediateID][targetID]->GetLargestPossibleRegion());
                                    trueIt.GoToBegin();
                                }
                                
                                ImageIterator * segmentationIt;
                                ImagePointerType atlasSegmentation;
                                bool haveSeg=m_atlasSegmentations.find(targetID) != m_atlasSegmentations.end();
                                if (haveSeg){
                                    atlasSegmentation=FilterUtils<ImageType>::NNResample(m_atlasSegmentations[targetID],this->m_ROI,false);
                                    LOGI(3,ImageUtils<ImageType>::writeImage("downsampledAtlasSegmentation.png",atlasSegmentation));
                                    segmentationIt = new ImageIterator(atlasSegmentation,this->m_ROI->GetLargestPossibleRegion());
                                    segmentationIt->GoToBegin();
                                    LOGV(3)<<"Executing loop with segmentation: "<<VAR(sourceID)<<" " <<VAR(targetID)<<" "<<VAR(intermediateID)<<endl;
                                    LOGV(3)<<VAR(atlasSegmentation->GetLargestPossibleRegion().GetSize())<<endl;
                                    LOGV(3)<<"not chrashed yet: "<<VAR(segmentationIt->GetIndex())<<" " <<endl;


                                }
                                
                                if ( ! haveSeg &&  m_atlasSegmentations.size()>0 ){
                                    LOGV(2)<<VAR(haveSeg)<<" "<<VAR(m_atlasSegmentations.size())<<endl;
                                    LOGV(2)<<"skipping loop "<<VAR(sourceID)<<" " <<VAR(targetID)<<" "<<VAR(intermediateID)<<endl;
                                    continue; //if we're using atlas segmentation as start point for closed loops, skip all loops which do not start at an atlas :o
                                }
                                SizeType roiSize=m_regionOfInterest.GetSize();
                            
                                // LOGV(1)<<VAR(dir)<<" "<<VAR(start)<<endl;
                                for (;!it.IsAtEnd();++it){

                                    bool valid=true;

                                    //get index in target domain
                                    IndexType targetIndex=it.GetIndex(),intermediateIndex,idx1;
                                    LOGV(9)<<VAR(targetIndex)<<endl;
                                    PointType ptIntermediate,ptTarget,ptSourceDirect;
                                    IndexType roiIntermediateIndex,roiTargetIndex;
                                
                                    //get physical point in target domain
                                    dSourceTarget->TransformIndexToPhysicalPoint(targetIndex,ptTarget);
                                    //get corresponding point in intermediate deform
                                    DeformationType dIntermediate=dIntermediateTarget->GetPixel(targetIndex);
                                    ptIntermediate= ptTarget + dIntermediate;
                                
                                    //get point in source image where direct deformation points to
                                    ptSourceDirect = ptTarget + dSourceTarget->GetPixel(targetIndex);
                                
                                    //get neighbors of that point
                                    std::vector<std::pair<IndexType,double> >ptIntermediateNeighbors,ptIntermediateNeighborsCircle;
                                    bool inside;
                                    if (m_linearInterpol){
                                        inside=getLinearNeighbors(dIntermediateTarget,ptIntermediate,ptIntermediateNeighbors);
                                        //inside=getGaussianNeighbors(dIntermediateTarget,ptIntermediate,ptIntermediateNeighbors,(ptIntermediate-ptSourceDirect).GetNorm());
                                    }else{
                                        inside=getNearestNeighbors(dIntermediateTarget,ptIntermediate,ptIntermediateNeighbors);
                                    }

 
                                        
                                    this->m_ROI->TransformPhysicalPointToIndex(ptTarget,roiTargetIndex);
                                    LOGV(9)<<VAR(targetIndex)<<" "<<VAR(roiTargetIndex)<<endl;
                                
                                    if (inside){

                                        //if there are atlas segmentations, we're only forming circles at atlas segmentation points
                                        //and at all points otherwise
                                        double val= m_atlasSegmentations.size()==0;
                                        
                                        bool segVal=1.0;

                                        //val*=getIndexBasedWeight(roiTargetIndex,roiSize);

                                        //multiply val by segConsistencyWeight if deformation starts from atlas segmentation
                                        if (haveSeg){
                                            segVal=segmentationIt->Get()>0;
                                            LOGV(3)<<VAR(segVal)<<" "<<VAR(segmentationIt->GetIndex())<<" " <<endl;
                                            LOGV(3)<<VAR(this->m_ROI->GetLargestPossibleRegion().GetSize())<<" "<<VAR(segmentationIt->GetImage()->GetLargestPossibleRegion().GetSize())<<endl;
                                            if (segVal){
                                                val=m_segConsisntencyWeight;
                                            }
                                        }

                              
                                 
                                        if (val>0){
                                            //statisticIt.Set(statisticIt.Get()+(disp));


                                            LOGV(9)<<VAR(source)<<" "<<VAR(intermediate)<<" "<<VAR(target)<<" "<<VAR(roiTargetIndex)<<" "<<VAR(d)<<endl;
                                            LOGV(9)<<VAR(edgeNumError(intermediate,target,roiTargetIndex,d))<<" "<<VAR(edgeNumDeformation(intermediate,target,roiTargetIndex,d))<<endl;
                                            //set w_d ~ 
                                            if (m_wFullCircleEnergy>0){
                                                //def and error intermediate->target
                                                x[c]=eq;
                                                y[c]=edgeNumError(intermediate,target,roiTargetIndex,d);
                                                v[c++]=val*m_wFullCircleEnergy;
                                                x[c]=eq;
                                                y[c]=edgeNumDeformation(intermediate,target,roiTargetIndex,d);
                                                v[c++]=val*m_wFullCircleEnergy;
                                                //interpolated def and error source->intermediate
                                                double localInconsistency=0.0;

                                                for (int i=0;i<ptIntermediateNeighbors.size();++i){
                                                    x[c]=eq;
                                                    y[c]=edgeNumError(source,intermediate,ptIntermediateNeighbors[i].first,d);
                                                    v[c++]=ptIntermediateNeighbors[i].second*val*m_wFullCircleEnergy;
                                                    x[c]=eq;
                                                    y[c]=edgeNumDeformation(source,intermediate,ptIntermediateNeighbors[i].first,d);
                                                    v[c++]=ptIntermediateNeighbors[i].second*val*m_wFullCircleEnergy;
                                                    PointType ptSourceIndirect= ptTarget + dSourceIntermediate->GetPixel(ptIntermediateNeighbors[i].first);
                                                    localInconsistency+=ptIntermediateNeighbors[i].second*(ptSourceIndirect[d]-ptSourceDirect[d]);
                                            
                                                }
                                                //minus def and error source->target
                                                x[c]=eq;
                                                y[c]=edgeNumError(source,target,roiTargetIndex,d);
                                                v[c++]= - val*m_wFullCircleEnergy;
                                                x[c]=eq;
                                                y[c]=edgeNumDeformation(source,target,roiTargetIndex,d);
                                                v[c++]= - val*m_wFullCircleEnergy;
                                                b[eq-1]= localInconsistency*m_wFullCircleEnergy;
                                                ++eq;

                                            
                                            }
                                    
                                            //set w_circ
                                            if (m_wCircleNorm>0.0){
                                                double RHS=0.0;
                                                if (estIntermediateTarget){
                                                    //indirect
                                                    x[c]=eq;
                                                    y[c]=edgeNumDeformation(intermediate,target,roiTargetIndex,d);
                                                    v[c++]=val* m_wCircleNorm;
                                                }else{
                                                    RHS-=dIntermediateTarget->GetPixel(roiTargetIndex)[d];
                                                }
                                                
#if 0
                                                //EXPERIMENTAL*****************************************
                                                x[c]=eq;
                                                y[c]=edgeNumError(intermediate,target,roiTargetIndex,d);
                                                v[c++]=-val*m_wCircleNorm;
                                                // *****************************************************
#endif
                                        
                                                double defSum=0.0;
                                                for (int i=0;i<ptIntermediateNeighbors.size();++i){
                                                    if (estSourceIntermediate){
                                                        x[c]=eq;
                                                        y[c]=edgeNumDeformation(source,intermediate,ptIntermediateNeighbors[i].first,d); // this is an APPROXIMIATION!!! might be bad :o
                                                        v[c++]=ptIntermediateNeighbors[i].second*val* m_wCircleNorm;
                                                        LOGV(8)<<VAR(roiTargetIndex)<<" "<<VAR(i)<<" "<<VAR(ptIntermediateNeighbors[i].first)<<" "<<VAR(ptIntermediateNeighbors[i].second)<<endl;
                                                    }else{
                                                        RHS-=ptIntermediateNeighbors[i].second*dSourceIntermediate->GetPixel(ptIntermediateNeighbors[i].first)[d];
                                                    }
                                                    defSum+=ptIntermediateNeighbors[i].second*dSourceIntermediate->GetPixel(ptIntermediateNeighbors[i].first)[d];
                                                }

                                                if (estSourceTarget){
                                                    //minus direct
                                                    x[c]=eq;
                                                    y[c]=edgeNumDeformation(source,target,roiTargetIndex,d);
                                                    v[c++]= - val* m_wCircleNorm;
                                                }else{
                                                    RHS+=dSourceTarget->GetPixel(roiTargetIndex)[d];
                                                }
                                        
                                                double residual= val*m_wCircleNorm*(dIntermediateTarget->GetPixel(roiTargetIndex)[d]
                                                                                    + defSum
                                                                                    - dSourceTarget->GetPixel(roiTargetIndex)[d]
                                                                                    );
                                                manualResidual+=residual*residual;
                                                b[eq-1]=m_wCircleNorm*RHS;
                                                ++eq;
                                            }
                                    
                                            if (m_wErrorInconsistency>0.0){
                                                val=1.0;
                                                //indirect
                                                x[c]=eq;
                                                y[c]=edgeNumError(intermediate,target,roiTargetIndex,d);
                                                v[c++]=val* m_wErrorInconsistency;
                                                double localInconsistency=0.0;
                                                for (int i=0;i<ptIntermediateNeighbors.size();++i){
                                                    x[c]=eq;
                                                    y[c]=edgeNumError(source,intermediate,ptIntermediateNeighbors[i].first,d); // this is an APPROXIMIATION!!! might be bad :o
                                                    v[c++]=ptIntermediateNeighbors[i].second*val* m_wErrorInconsistency;
                                                    PointType ptIntermediate;
                                                    dSourceIntermediate->TransformIndexToPhysicalPoint(ptIntermediateNeighbors[i].first,ptIntermediate);
                                                    PointType ptSourceIndirect= ptIntermediate + dSourceIntermediate->GetPixel(ptIntermediateNeighbors[i].first);
                                                    localInconsistency+=ptIntermediateNeighbors[i].second*(ptSourceIndirect[d]-ptSourceDirect[d]);
                                            
                                                }
                                                //minus direct
                                                x[c]=eq;
                                                y[c]=edgeNumError(source,target,roiTargetIndex,d);
                                                v[c++]= - val* m_wErrorInconsistency;
                                                b[eq-1]=localInconsistency* m_wErrorInconsistency;
                                                ++eq;
                                        
                                            }
                                        }//val >0
                                    }//inside
                                    if (haveSeg)                                             ++(*segmentationIt);

                                }//image iterator
                                if (haveSeg)         delete segmentationIt;

                            }//if anything is to estimate :)
                        }//if
                        
                    }//intermediate
                }//if
            }//target
        }//source
        LOGV(1)<<VAR(manualResidual)<<endl;
    }//compute triplets

    void computePairwiseEnergiesAndBounds(double * x, 
                                          double * y,
                                          double * v, 
                                          double * b, 
                                          double * init, 
                                          double * lb, 
                                          double * ub, 
                                          long int & c , 
                                          long int & eq , 
                                          unsigned int  d)
    {
    
        for (int s = 0;s<m_numImages;++s){                            
            int source=s;
            for (int t=0;t<m_numImages;++t){
                if (t!=s){
                    int target=t;
                    //pairwise energies!
                    string sourceID=(this->m_imageIDList)[source];
                    string targetID = (this->m_imageIDList)[target];
                  
                    //only compute pairwise energies if deformation is to be estimated
                    if ((m_downSampledDeformationCache)[sourceID][targetID].IsNotNull()){
                      
                        FloatImagePointerType lncc;
                        FloatImageIterator lnccIt;
                        if (  m_sigma>0.0 && (m_wErrorNorm>0.0 || m_wTransformationSimilarity)){
                            lncc=(m_pairwiseLocalWeightMaps)[sourceID][targetID];
                            lnccIt=FloatImageIterator(lncc,lncc->GetLargestPossibleRegion());
                            lnccIt.GoToBegin();
                        }

                        DeformationFieldIterator previousIt;
                        double priorWeight=1.0;
                        if (true && ! m_updateDeformations && m_haveDeformationEstimate && (m_updatedDeformationCache)[sourceID][targetID].IsNotNull()){
                            DeformationFieldPointerType estDef=(m_updatedDeformationCache)[sourceID][targetID];
                            DeformationFieldPointerType diff=TransfUtils<ImageType>::subtract(estDef,(m_downSampledDeformationCache)[sourceID][targetID]);
                            double defNorm=TransfUtils<ImageType>::computeDeformationNorm(diff);
                            if (defNorm!=0.0){
                                priorWeight=1.0/defNorm;
                            }else
                                priorWeight= 10;
                        
                            previousIt=DeformationFieldIterator(diff,m_regionOfInterest);
                            previousIt.GoToBegin();
                        }
                        
                        DeformationFieldPointerType dSourceTarget=(this->m_downSampledDeformationCache)[sourceID][targetID];
                        DeformationFieldIterator it(dSourceTarget,m_regionOfInterest);
                        it.GoToBegin();
                    
                    
                        GaussEstimatorType * statisticsEstimatorSourceTarget;
                        double globalMeanInconsistency=1.0;
                        double trueError  ;
                        if (m_wErrorStatistics>0.0){
                            statisticsEstimatorSourceTarget = &m_pairwiseInconsistencyStatistics[s][t];
                            statisticsEstimatorSourceTarget->finalize();
                            globalMeanInconsistency=FilterUtils<FloatImageType>::getMean(statisticsEstimatorSourceTarget->getMean());
                            if (globalMeanInconsistency==0.0){
                                globalMeanInconsistency=0.01;
                            }
                            globalMeanInconsistency*=globalMeanInconsistency;
                            LOGV(1)<<sourceID<<" "<<targetID<<" "<<VAR(globalMeanInconsistency)<<endl;
                            }

                        FloatImagePointerType newLocalWeights;
                        if (m_locallyUpdateDeformationEstimate && ! m_updateDeformations && m_haveDeformationEstimate && (m_updatedDeformationCache)[sourceID][targetID].IsNotNull()){
                            newLocalWeights=(m_updatedPairwiseLocalWeightMaps)[sourceID][targetID];
                        }


                        for (;!it.IsAtEnd();++it){
                            DeformationType localDef=it.Get();
                            IndexType idx=it.GetIndex();
                            LOGV(8)<<VAR(eq)<<" "<<VAR(localDef)<<endl;
                        

                            int edgeNumDef=edgeNumDeformation(source,target,idx,d);
                            int edgeNumErr;
                            if (m_estError)
                                edgeNumErr=edgeNumError(source,target,idx,d);

                            //intensity based weight
                            double weight=1.0;
                            if (lncc.IsNotNull()){
                                weight = lnccIt.Get();
                                LOGV(6)<<VAR(weight)<<endl;
                                ++lnccIt;
                            }
                          
                        
                                
                            //weight based on previous estimate
                            double weight2=1.0;
                            double expectedError=0.0;
                            if (m_haveDeformationEstimate){
                                priorWeight=1.0;
                                if (false ){
                                    expectedError = previousIt.Get()[d];
                                    ++previousIt;

                                    //double diffNorm=fabs(previousIt.Get()[d]);
                                    double diffNorm=previousIt.Get().GetNorm();
                                    if (diffNorm!=0.0){
                                        priorWeight=1.0/diffNorm;
                                    }else
                                        priorWeight=100;
                                
                                    //LOGV(4)<<VAR(diffNorm)<<" "<<VAR(priorWeight)<<endl;
                                    priorWeight=min(max(priorWeight,0.001),10.0);
                                
                                }
                            }
                                
                            //set w_delta
                            //set eqn for soft constraining the error to be small
                            if (m_wErrorNorm>0.0){
                                x[c]    = eq;
                                y[c]    = edgeNumErr;
                                v[c++]  = 1.0*m_wErrorNorm*weight;
                                b[eq-1] = 0.0;//weight*m_wErrorNorm;
                                ++eq;
                                LOGV(8)<<VAR(source)<<" "<<VAR(target)<<" "<<VAR(idx)<<" "<<VAR(d)<<" "<<VAR(edgeNumErr)<<" "<<VAR(weight)<<endl;
                            }


                            //set w_delta
                            //set eqn for soft constraining the error to be small
                            double meanInconsistency=1.0;
                            double varInconsistency;
                            if ( m_wErrorStatistics>0.0){
                                LOGV(7)<<VAR(statisticsEstimatorSourceTarget->getMean()->GetLargestPossibleRegion().GetSize())<<" "<<VAR(idx)<<endl;
                                meanInconsistency=statisticsEstimatorSourceTarget->getMean()->GetPixel(idx);
                                meanInconsistency=meanInconsistency!=0.0?meanInconsistency:0.0001;
#if 0
                                //varInconsistency=sqrt((fabs(statisticsEstimatorSourceTarget->getVariance()->GetPixel(idx))));
                                if (varInconsistency == 0.0){
                                    varInconsistency = 1e-5;
                                }
                                weight2 = 1.0/(varInconsistency);
                                x[c]    = eq;
                                y[c]    = edgeNumDef;
                                v[c++]  = 1.0*m_wErrorStatistics*weight2;
                                //b[eq-1] = m_wErrorStatistics*weight2*trueError;
                                b[eq-1] = m_wErrorStatistics*weight2*meanInconsistency;
                                ++eq;
#endif     
                              
                                LOGV(7)<<VAR(source)<<" "<<VAR(target)<<" "<<VAR(idx)<<" "<<VAR(d)<<" "<<VAR(edgeNumErr)<<" "<<VAR(meanInconsistency)<<" "<<VAR(weight)<<" "<<VAR(trueError)<<endl;
     
                            }
                        
                            double localWeight=-1;
                            double localUpdatedDef;
                            //get local similarity weight for updated deformation if available
                            if (m_locallyUpdateDeformationEstimate  &&  m_haveDeformationEstimate && ! m_updateDeformations){
                                localWeight=newLocalWeights->GetPixel(idx);
                                localUpdatedDef=(m_updatedDeformationCache)[sourceID][targetID]->GetPixel(idx)[d];
                            }
                        
                            //set w_T
                            //set eqn for soft constraining the estimated true deformation to be similar to the original deformation
                            if (m_wTransformationSimilarity>0.0){
                                //weight=1.0/sqrt(fabs(meanInconsistency));
                                double localDisp  =localDef[d];
                                if (m_metric !="gradient"){
                                    if (localWeight>=weight){
                                        localDisp=localUpdatedDef;
                                        weight=localWeight;
                                    }
                                    //double localIncStatistics=min(float(100.0),max(statisticsEstimatorSourceTarget->getMean()->GetPixel(idx),(float)0.01));
                                    //double trueError=fabs(localDef[d]-                                                                          (m_trueDeformations)[sourceID][targetID]->GetPixel(idx)[d]);
                                    //LOGV(3)<< VAR(sourceID)<< " "<<VAR(targetID)<<" "<<VAR(localIncStatistics) << "  " <<VAR(weight)<<" "<<VAR(trueError)<<endl;
                                    //weight/= min(float(100.0),max(statisticsEstimatorSourceTarget->getMean()->GetPixel(idx),(float)0.01));
                                    
                                    weight/=globalMeanInconsistency;
                                }
                                else{
                                    //weight=1.0;
                                    priorWeight=1.0;
                                    localDisp-=(m_pairwiseGradients)[sourceID][targetID]->GetPixel(idx)[d];
                                }
                                //trueError= (this->m_downSampledDeformationCache)[sourceID][targetID]->GetPixel(idx)[d]-(m_trueDeformations)[sourceID][targetID]->GetPixel(idx)[d];
                                //weight=trueError!=0.0?1.0/abs(trueError):100;//pow(weight,meanInconsistency);
                                //weight=pow((1.0-weight),4.0)*25;
                                //weight=weight!=0.0?1.0/weight:100;
                                weight=weight/meanInconsistency;
                                if (weight>0.0){
                                    x[c]    = eq;
                                    y[c]    = edgeNumDef;
                                    v[c++]  = 1.0*m_wTransformationSimilarity *weight*priorWeight;
                                    //b[eq-1] = (localDef[d]-trueError)*m_wTransformationSimilarity;// * weight;
                                    b[eq-1] = localDisp*m_wTransformationSimilarity * weight*priorWeight;
                                    ++eq;
                                    LOGV(8)<<VAR(source)<<" "<<VAR(target)<<" "<<VAR(idx)<<" "<<VAR(d)<<" "<<VAR(edgeNumErr)<<" "<<VAR(priorWeight)<<endl;
                                }
                            }
                            
                            if (m_wTransformationSymmetry>0.0){
                                std::vector<std::pair<IndexType,double> > ptNeighbors;
                                DeformationType def=localDef;
                                if ((m_updatedDeformationCache)[sourceID][targetID].IsNotNull()){
                                    def=(m_updatedDeformationCache)[sourceID][targetID]->GetPixel(idx);
                                }
                                
                                PointType deformedPoint;
                                dSourceTarget->TransformIndexToPhysicalPoint(idx,deformedPoint);
                                deformedPoint = deformedPoint+def;
                                bool inside=getLinearNeighbors(dSourceTarget,deformedPoint,ptNeighbors);
                                if (inside){
                                    x[c]    = eq;
                                    y[c]    = edgeNumDef;
                                    v[c++]  = 1.0*m_wTransformationSymmetry;
                                    for (int i=0;i<ptNeighbors.size();++i){
                                        x[c]=eq;
                                        y[c]=edgeNumDeformation(source,target,ptNeighbors[i].first,d); // this is an APPROXIMIATION!!! might be bad :o
                                        v[c++]=-1.0*ptNeighbors[i].second* m_wTransformationSymmetry;
                                    }
                                    b[eq-1] = 0.0;
                                    ++eq;
                                }

                            }
                            
                            
                            //constraint that estimated def + estimated error = original def
                            if (m_wSum>0.0){
                                x[c]   = eq;
                                y[c]   = edgeNumDef;
                                v[c++] = m_wSum;
                                x[c]   = eq;
                                y[c]   = edgeNumErr;
                                v[c++] = m_wSum;
                                b[eq-1]=m_wSum*localDef[d];
                                ++eq;
                            }

#define CURVATURE
#ifdef CURVATURE
                            //spatial smootheness of estimated deformations
                            if (m_wDeformationSmootheness>0.0){
                                for (unsigned int n=0;n<D;++n){
                                    OffsetType off,off2;
                                    off.Fill(0);
                                    off2=off;
                                    off[n]=1;
                                    off2[n]=-1;
                                    double smoothenessWeight =this->m_wDeformationSmootheness*m_spacingBasedSmoothnessReduction;
                                    if (n!=d){
                                        //smoothenss for shearing is different
                                        smoothenessWeight*=m_shearingReduction;
                                    }
                                    IndexType neighborIndexRight=idx+off;
                                    IndexType neighborIndexLeft=idx+off2;
                                    if (dSourceTarget->GetLargestPossibleRegion().IsInside(neighborIndexRight) &&dSourceTarget->GetLargestPossibleRegion().IsInside(neighborIndexLeft) ){
                                        x[c]=eq;
                                        y[c]=edgeNumDef;
                                        v[c++]=-2.0*smoothenessWeight;
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

#else

                            //spatial smootheness of estimated deformations
                            if (m_wDeformationSmootheness>0.0){
                                for (unsigned int n=0;n<D;++n){
                                    OffsetType off,off2;
                                    off.Fill(0);
                                    off2=off;
                                    off[n]=1;
                                    off2[n]=-1;
                                    double smoothenessWeight =this->m_wDeformationSmootheness*m_spacingBasedSmoothnessReduction;
                                    if (n!=d){
                                        //smoothenss for shearing is different
                                        smoothenessWeight*=m_shearingReduction;
                                    }
                                    IndexType neighborIndexRight=idx+off;
                                    if (dSourceTarget->GetLargestPossibleRegion().IsInside(neighborIndexRight)){
                                        x[c]=eq;
                                        y[c]=edgeNumDef;
                                        v[c++]=-1.0*smoothenessWeight;
                                        x[c]=eq;
                                        y[c]=edgeNumDeformation(source,target,neighborIndexRight,d);
                                        v[c++]=smoothenessWeight;
                                        b[eq-1]=0.0;
                                        ++eq;
                                    }
                                }//inside
                            }

#endif
                            //spatial un-smootheness of estimated errors
                            if (m_wErrorSmootheness>0.0){
                                for (unsigned int n=0;n<D;++n){
                                    OffsetType off,off2;
                                    off.Fill(0);
                                    off2=off;
                                    off[n]=1;
                                    off2[n]=-1;
                                    IndexType neighborIndexRight=idx+off;
                                    IndexType neighborIndexLeft=idx+off2;
                                    if (dSourceTarget->GetLargestPossibleRegion().IsInside(neighborIndexRight) &&dSourceTarget->GetLargestPossibleRegion().IsInside(neighborIndexLeft) ){
                                        x[c]   = eq;
                                        y[c]   = edgeNumErr;
                                        v[c++] = -2.0;
                                        x[c]   = eq;
                                        y[c]   = edgeNumError(source,target,neighborIndexRight,d);
                                        v[c++] = 1.0;
                                        x[c]   = eq;
                                        y[c]   = edgeNumError(source,target,neighborIndexLeft,d);
                                        v[c++] = 1.0;
                                        b[eq-1]= this->m_wErrorSmootheness;
                                        ++eq;
                                    }
                                }//inside
                            }//wwsDelta
                            //set initialisation values

                            
                            //set bounds on variables
                            //error
                            double extent     = (dSourceTarget->GetLargestPossibleRegion().GetSize()[d]-1)*dSourceTarget->GetSpacing()[d];
                            double upperBound = extent;
                            double lowerBound = -upperBound;
                            //LOGV(4)<<VAR(upperBound)<<endl;
                            if (m_wErrorStatistics>0.){
                                lowerBound        = meanInconsistency-3*sqrt(varInconsistency);
                                upperBound        = meanInconsistency+3*sqrt(varInconsistency);
                                //LOGV(6)<<VAR(weight)<<" "<<VAR(lowerBound)<<VAR(upperBound)<<" "<<VAR(meanInconsistency)<<" "<<VAR(varInconsistency)<<" "<<VAR(trueError)<<endl;
                            }

                        
                            if (m_estError){
                                //LOGV(6)<<VAR(edgeNumErr)<<" "<<VAR(m_nVars)<<endl;
                                init[edgeNumErr-1] = 0;//meanInconsistency;
                                lb[edgeNumErr-1]   = lowerBound;
                                ub[edgeNumErr-1]   = upperBound;
                            }
                            //deformation
                            //deformation should not fall outside image bounds
                            PointType pt;
                            dSourceTarget->TransformIndexToPhysicalPoint(idx,pt);
                            //warning: assumes 1 1 1 direction!
                            //deformation can maximally go back to origin
                            if (m_estDef){
                                //LOGV(6)<<VAR(edgeNumDef)<<" "<<VAR(m_nVars)<<endl;
                                lb[edgeNumDef-1]   =  dSourceTarget->GetOrigin()[d]-pt[d] -0.0001;
                                //deformation can maximally transform pt to extent of image
                                ub[edgeNumDef-1]   =  dSourceTarget->GetOrigin()[d]+extent-pt[d] +0.0001;
                                //LOGV(4)<<VAR(pt)<<" "<<VAR(dSourceTarget->GetOrigin()[d]-pt[d])<<" "<<VAR( dSourceTarget->GetOrigin()[d]+extent-pt[d]  )<<endl;
                                //init[edgeNumDef-1] =  0;
                                //init[edgeNumDef-1] =  localDef[d] -expectedError;
                                init[edgeNumDef-1] =  localDef[d];
                                //init[edgeNumDef-1] =  (localDef[d]-trueError) ;
                            
                            }
                           
                        }//for

                    }//if
                }//check if estimation is necessary
            }//target
        }//source
    }//computePairwiseEnergiesAndBounds



    void computeConstraints(double * x, 
                            double * y,
                            double * v, 
                            double * b, 
                            int d
                            )
    {

        int c=0;
        int eq=1;
        double dblSpacing=this->m_ROI->GetSpacing()[d];
        for (int s = 0;s<m_numImages;++s){                            
            int source=s;
            for (int t=0;t<m_numImages;++t){
                if (t!=s){
                    int target=t;
                    //pairwise energies!
                    string sourceID=(this->m_imageIDList)[source];
                    string targetID = (this->m_imageIDList)[target];
                  
                    //only compute pairwise energies if deformation is to be estimated
                    if ((m_downSampledDeformationCache)[sourceID][targetID].IsNotNull()){
                        DeformationFieldPointerType dSourceTarget=(this->m_downSampledDeformationCache)[sourceID][targetID];
                        DeformationFieldIterator it(dSourceTarget,m_regionOfInterest);
                        it.GoToBegin();
                        for (;!it.IsAtEnd();++it){
                            DeformationType localDef=it.Get();
                            IndexType idx=it.GetIndex();
                            LOGV(8)<<VAR(eq)<<" "<<VAR(localDef)<<endl;
                            int edgeNumDef=edgeNumDeformation(source,target,idx,d);

                            //constraints
                            for (unsigned int n=0;n<D;++n){
                                OffsetType off,off2;
                                off.Fill(0);
                                off2=off;
                                off[n]=2;
                                off2[n]=-1;

                                //only constraint in-line neighbors to prevent folding.
                                // d1-d2<spacing
                                IndexType neighborIndexRight=idx+off;
                                if (dSourceTarget->GetLargestPossibleRegion().IsInside(neighborIndexRight)){
                                    if (n == d ){
                                        x[c]=eq;
                                        y[c]=edgeNumDef;
                                        v[c++]=1.0;
                                        x[c]=eq;
                                        y[c]=edgeNumDeformation(source,target,neighborIndexRight,d);
                                        v[c++]=-1.0;
                                        b[eq-1]=1.8*dblSpacing;
                                        ++eq;
                                    }//n==d
#if 1
                                    else{
                                        x[c]=eq;
                                        y[c]=edgeNumDef;
                                        v[c++]=1.0;
                                        x[c]=eq;
                                        y[c]=edgeNumDeformation(source,target,neighborIndexRight,d);
                                        v[c++]=-1.0;
                                        b[eq-1]=dblSpacing*1.1;
                                        ++eq;
                                        
                                        x[c]=eq;
                                        y[c]=edgeNumDef;
                                        v[c++]=-1.0;
                                        x[c]=eq;
                                        y[c]=edgeNumDeformation(source,target,neighborIndexRight,d);
                                        v[c++]=1.0;
                                        b[eq-1]=dblSpacing*1.1;
                                        ++eq;
                                    }
#endif
                                }//inside
                            }//neighbors
                        }//for
                    }//if
                }//check if estimation is necessary
            }//target
        }//source
    }//computeConstraints

public:
  
    double getIndexBasedWeight(IndexType idx,SizeType size){
        double weight=100.0;
        for (int d=0;d<D;++d){
            //first center
            double halfSize=0.5*size[d];
            double localWeight=fabs(1.0*idx[d]-halfSize);
            double distance=localWeight-halfSize;
            if (distance==0.0){
                localWeight=0.0;
            }else{
                //compute falloff
                localWeight=max(0.0,1.0+1.0/(distance));
            }
            //take max
            weight=min(weight,localWeight);
            LOGV(8)<<VAR(idx[d])<<" "<<VAR(size[d])<<" "<<VAR(localWeight)<<endl;

        }
        return weight;
    }

 

    bool findDeformation( map<string, map<string, string> > cache, string id1, string id2){
        
        typename map<string, map<string, string> >::iterator it;
        
        
        it=cache.find(id1);
        
        if (it !=cache.end()){
            typename map<string, string>::iterator it2;
            it2=it->second.find(id2);
            if (it2!=it->second.end()){
                return true;
            }
        }
        return false;
    }

    void doubleImageResolution(){
        m_resolutionFactor*=2;
        m_resolutionFactor=max(1.0,m_resolutionFactor);
    }

   
   


    void DoALot(string directory=""){
        typedef typename itk::LabelOverlapMeasuresImageFilter<ImageType> OverlapMeasureFilterType;
        m_spacingBasedSmoothnessReduction=1.0/this->m_ROI->GetSpacing()[0];
        //initialize ADE with zero if it is going to be computed.
        m_ADE=(m_trueDeformationFileList.size()>0)?0:-1;
        m_dice=0;
        m_TRE=0;
        double m_averageMinJac=0.0;
        m_averageNCC=0.0;
        m_minMinJacobian=std::numeric_limits<double>::max();
        int count=0;
        bool firstRun=false;
        for (int target=0;target<m_numImages;++target){
            for (int source=0;source<m_numImages;++source){
                if (source!=target){
                    string targetID=(m_imageIDList)[target];
                    string sourceID=(m_imageIDList)[source];
                    DeformationFieldPointerType deformation,updatedDeform, knownDeformation;
                    bool estDef=false;
                    bool updateThisDeformation=false;
                    ImagePointerType targetImage=(m_imageList)[targetID];
                    double nCC;
                    if (findDeformation(m_deformationFileList,sourceID,targetID)){
                      

                        LOGV(5)<<"preparing deformation to be estimated for "<<sourceID<<" to " <<targetID <<endl;
                        estDef=true;
                     
                        if (m_estimatedErrors[sourceID][targetID].IsNotNull()){
                            if (m_lowResolutionEval){
                                DeformationFieldPointerType estimatedDeformation=m_estimatedErrors[sourceID][targetID];
                                if (m_bSplineInterpol){
                                    updatedDeform=TransfUtils<ImageType>::bSplineInterpolateDeformationField(estimatedDeformation,targetImage,false);
                                    //updatedDeform=TransfUtils<ImageType>::computeDeformationFieldFromBSplineTransform(estimatedDeformation,targetImage);
                                }else{
                                    updatedDeform=TransfUtils<ImageType>::linearInterpolateDeformationField(estimatedDeformation,targetImage,false);
                                }
                                
                            }else{
                                //read full resolution deformations
                                deformation=ImageUtils<DeformationFieldType>::readImage(m_deformationFileList[sourceID][targetID]);
                                //resample deformation to target image space
                                deformation=TransfUtils<ImageType>::linearInterpolateDeformationField(deformation,targetImage,false);
                                //update deformation
                                DeformationFieldPointerType estimatedDeformation=m_estimatedErrors[sourceID][targetID];
                                //downsample original deformation with smoothing
                                DeformationFieldPointerType downSampledDeformation;
                                if (m_bSplineInterpol){
                                    //downSampledDeformation=TransfUtils<ImageType>::computeDeformationFieldFromBSplineTransform(deformation,this->m_ROI);
                                    downSampledDeformation=TransfUtils<ImageType>::bSplineInterpolateDeformationField(deformation,m_estimatedErrors[sourceID][targetID],m_smoothDeformationDownsampling);
                                }else{
                                    downSampledDeformation=TransfUtils<ImageType>::linearInterpolateDeformationField(deformation,estimatedDeformation,m_smoothDeformationDownsampling);
                                }
                                //subtract low resolution error estimate (sampled to the correct resolution in case it was estimated for a lower resolution)
                                DeformationFieldPointerType estimatedError=TransfUtils<ImageType>::subtract(downSampledDeformation,estimatedDeformation);
                                
                                
                                //upsample estimated error
                                DeformationFieldPointerType fullResolutionErrorEstimate;
                                if (m_bSplineInterpol){
                                    fullResolutionErrorEstimate=TransfUtils<ImageType>::bSplineInterpolateDeformationField(estimatedError,targetImage,m_smoothDeformationDownsampling);
                                    //fullResolutionErrorEstimate=TransfUtils<ImageType>::computeBSplineTransformFromDeformationField(estimatedError,targetImage);
                                }else{
                                    fullResolutionErrorEstimate=TransfUtils<ImageType>::linearInterpolateDeformationField(estimatedError,targetImage,false);
                                }
                                if (m_lineSearch){
                                    //search for updating weight which improves the metric most
                                    //doesnt work well:D
                                    double alpha=2.0;
                                    double bestAlpha=1.0;
                                    double bestNCC=2;

                                    for (int i=0;i<10;++i){
                                        DeformationFieldPointerType def=TransfUtils<ImageType>::subtract(deformation,TransfUtils<ImageType>::multiplyOutOfPlace(fullResolutionErrorEstimate,alpha));
                                        double nCC=Metrics<ImageType>::nCC( m_imageList[targetID], m_imageList[sourceID],def);
                                        LOGV(2)<<VAR(sourceID)<<" "<<VAR(targetID)<<" "<<VAR(alpha)<<" "<<VAR(nCC)<<endl;
                                        if (nCC<bestNCC){
                                            updatedDeform=def;
                                            bestNCC=nCC;
                                        }
                                        alpha/=2.0;
                                    }
                                    
                                }else{
                                    updatedDeform=TransfUtils<ImageType>::subtract(deformation,fullResolutionErrorEstimate);
                                }
                                LOGV(3)<<"updated deformation"<<endl;
                            }
                        }else{
                            //read full resolution deformations
                            deformation=ImageUtils<DeformationFieldType>::readImage(m_deformationFileList[sourceID][targetID]);
                            //resample deformation to target image space
                            deformation=TransfUtils<ImageType>::linearInterpolateDeformationField(deformation,targetImage,false);
                            updatedDeform=deformation;
                        }
                        //compare segmentations
                        if (m_groundTruthSegmentations[targetID].IsNotNull() && m_groundTruthSegmentations[sourceID].IsNotNull()){
                            ImagePointerType deformedSeg=TransfUtils<ImageType>::warpImage(  m_groundTruthSegmentations[sourceID] , updatedDeform,true);
                            typename OverlapMeasureFilterType::Pointer filter = OverlapMeasureFilterType::New();
                            filter->SetSourceImage((m_groundTruthSegmentations)[targetID]);
                            filter->SetTargetImage(deformedSeg);
                            filter->SetCoordinateTolerance(1e-4);
                            filter->Update();
                            double dice=filter->GetDiceCoefficient();
                            LOGV(1)<<VAR(sourceID)<<" "<<VAR(targetID)<<" "<<VAR(dice)<<endl;
                            typedef typename SegmentationTools<ImageType>::OverlapScores OverlapScores;
                            std::vector<OverlapScores> scores=SegmentationTools<ImageType>::computeOverlapMultilabel((m_groundTruthSegmentations)[targetID],deformedSeg);
                            LOGV(1)<<VAR(sourceID)<<" "<<VAR(targetID)<<" ";
                             dice=0.0;
                            for (int s=0;s<scores.size();++s){
                                if (mylog.getVerbosity()>1){std::cout<<" "<<VAR(scores[s].labelID)<<" "<<scores[s].dice;}
                                dice+=scores[s].dice;
                            }
                            if (mylog.getVerbosity()>1){                                std::cout<<endl;}
                            dice=dice/(scores.size());
                            LOGV(1)<<"AverageDice : "<<dice<<endl;
                            
                            m_dice+=dice;
                        }
                        

                        // compare landmarks
                        if (m_landmarkFileList.size()){
                            //hope that all landmark files are available :D
                            m_TRE+=TransfUtils<ImageType>::computeTRE(m_landmarkFileList[targetID], m_landmarkFileList[sourceID],updatedDeform,m_imageList[targetID]);
                        }
                        
                        
                        
                        //compute LNCC
                        ImagePointerType warpedImage = TransfUtils<ImageType>::warpImage(  m_imageList[sourceID] , updatedDeform,m_metric == "categorical");
                        double samplingFactor=1.0*warpedImage->GetLargestPossibleRegion().GetSize()[0]/this->m_ROI->GetLargestPossibleRegion().GetSize()[0];
                        double imageResamplingFactor=min(1.0,8.0/(samplingFactor));
                        nCC= Metrics<ImageType>::nCC(targetImage,warpedImage);
                        m_averageNCC+=nCC;
                        if (m_metric != "categorical"){
                            warpedImage=FilterUtils<ImageType>::LinearResample(warpedImage,imageResamplingFactor,true);
                            targetImage=FilterUtils<ImageType>::LinearResample(targetImage,imageResamplingFactor,true);
                        }else{
                            warpedImage=FilterUtils<ImageType>::NNResample(warpedImage,imageResamplingFactor,false);
                            targetImage=FilterUtils<ImageType>::NNResample(targetImage,imageResamplingFactor,false);

                        }
                        LOGV(3)<<"Computing metric "<<m_metric<<" on images downsampled by "<<samplingFactor<<" to size "<<warpedImage->GetLargestPossibleRegion().GetSize()<<endl;
                        FloatImagePointerType lncc;
                        if (m_metric == "lncc"){
                            //lncc= Metrics<ImageType,FloatImageType>::efficientLNCC(warpedImage,targetImage,m_sigma,m_exponent);
                            lncc= Metrics<ImageType,FloatImageType,long double>::efficientLNCC(warpedImage,targetImage,m_sigma,m_exponent);
                        }else if (m_metric == "itklncc"){
                            lncc= Metrics<ImageType,FloatImageType>::ITKLNCC(warpedImage,targetImage,m_sigma,m_exponent, this->m_ROI);
                        }else if (m_metric == "lsad"){
                            lncc= Metrics<ImageType,FloatImageType>::LSADNorm(warpedImage,targetImage,m_sigma,m_exponent);
                          
                        }else if (m_metric == "lssd"){
                            lncc= Metrics<ImageType,FloatImageType>::LSSDNorm(warpedImage,targetImage,m_sigma,m_exponent);
                        }else if (m_metric == "localautocorrelation"){
                            //lncc= Metrics<ImageType,FloatImageType>::LSSD(warpedImage,(m_imageList)[targetID],m_sigma);
                            lncc= Metrics<ImageType,FloatImageType>::localMetricAutocorrelation(warpedImage,targetImage,m_sigma,2,"lssd",m_exponent);
                        }else if (m_metric == "gradient"){
                            //lncc= Metrics<ImageType,FloatImageType>::LSSD(warpedImage,(m_imageList)[targetID],m_sigma);
                            lncc=  Metrics<ImageType,FloatImageType>::efficientLNCC(warpedImage,targetImage,m_sigma,m_exponent);
                        }else if (m_metric == "categorical"){
                            lncc=  Metrics<ImageType,FloatImageType>::CategoricalDiffNorm(warpedImage,targetImage,m_sigma,m_exponent);
                        }else{
                            LOG<<"do not understand "<<VAR(m_metric)<<",aborting."<<endl;
                            exit(-1);
                        } 

                        //multiply image with normed gradients of target and warped source image
                        if (m_filterMetricWithGradient){
                            //typedef typename itk::GradientMagnitudeImageFilter<ImageType,FloatImageType> GradientFilter;
                            typedef typename itk::GradientMagnitudeRecursiveGaussianImageFilter<ImageType,FloatImageType> GradientFilter;

                            //get gradient of target image
                            typename GradientFilter::Pointer filter=GradientFilter::New();
                            filter->SetInput((ConstImagePointerType)targetImage);
                            filter->SetSigma(m_sigma);
                            filter->Update();
                            FloatImagePointerType gradient=filter->GetOutput();
                            ImageUtils<FloatImageType>::expNormImage(gradient,0.0);
                            ImageUtils<FloatImageType>::multiplyImage(gradient,-1.0);
                            ImageUtils<FloatImageType>::add(gradient,1.0);

                            //multiply lncc with target gradient
                            lncc=ImageUtils<FloatImageType>::multiplyImageOutOfPlace(lncc,gradient);
#if 0
                            //get warped moving gradient
                            typename GradientFilter::Pointer filter2=GradientFilter::New();
                            filter2->SetInput((ConstImagePointerType)warpedImage);
                            // filter2->SetSigma(m_sigma);
                            filter2->Update();
                            gradient=filter2->GetOutput();
                            ImageUtils<FloatImageType>::expNormImage(gradient,0.0);
                            ImageUtils<FloatImageType>::multiplyImage(gradient,-1.0);
                            ImageUtils<FloatImageType>::add(gradient,1.0);

                            //multiply
                            lncc=ImageUtils<FloatImageType>::multiplyImageOutOfPlace(lncc,gradient);
#endif
                        }

                        LOGV(3)<<"done."<<endl;
                        ostringstream oss;
                        oss<<m_metric<<"-"<<sourceID<<"-TO-"<<targetID<<".mha";
#if 0
                        if (D==2){
                            oss<<".png";
                        }
                        else{
                            oss<<".nii";
                        }
                        LOGI(6,ImageUtils<ImageType>::writeImage(oss.str(),FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(lncc,255))));
#else  
                        LOGI(6,ImageUtils<FloatImageType>::writeImage(oss.str(),lncc));
#endif
                        //resample lncc result
                         
                        //resample with smoothing
                        lncc=FilterUtils<FloatImageType>::LinearResample(lncc,FilterUtils<ImageType,FloatImageType>::cast(this->m_ROI),false);
                         
                        oss<<"-resampled";
                        if (D==2){
                            oss<<".png";
                        }else
                            oss<<".nii";
                        LOGI(6,ImageUtils<ImageType>::writeImage(oss.str(),FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(lncc,255))));
                         
                        //store...
                        if (m_updateDeformations || !m_pairwiseLocalWeightMaps[sourceID][targetID].IsNotNull()){

                            if (m_pairwiseLocalWeightMaps[sourceID][targetID].IsNull()){
                                //certainly update local weights maps when it was not set before ;)
                                m_pairwiseLocalWeightMaps[sourceID][targetID]=lncc;
                                m_pairwiseGlobalSimilarity[sourceID][targetID]=nCC;
                                updateThisDeformation=true;
                            }else{
                                if (! m_updateDeformationsGlobalSim || 
                                    (nCC <   m_pairwiseGlobalSimilarity[sourceID][targetID] ))
                                    {
                                        //update if global Sim improved or if it shouldn't be considered anyway
                                        m_pairwiseLocalWeightMaps[sourceID][targetID]=lncc;
                                        m_pairwiseGlobalSimilarity[sourceID][targetID]=nCC;
                                        updateThisDeformation=true;

                                    }
                            }
                        }else{
                            m_updatedPairwiseLocalWeightMaps[sourceID][targetID]=lncc;
                            //upsample if necessary.. note that here, quite some accuracy of the lncc is lost :( in fact, lncc should be recomputed
                            if (m_pairwiseLocalWeightMaps[sourceID][targetID]->GetLargestPossibleRegion().GetSize()!=this->m_ROI->GetLargestPossibleRegion().GetSize()){
                                //update the pairwise similarity of the original deformation to the correct resolution
                                warpedImage = TransfUtils<ImageType>::warpImage(  m_imageList[sourceID] , deformation,m_metric == "categorical");
                                 if (m_metric != "categorical"){
                                     warpedImage=FilterUtils<ImageType>::LinearResample(warpedImage,imageResamplingFactor,true);
                                 }else{
                                     warpedImage=FilterUtils<ImageType>::NNResample(warpedImage,imageResamplingFactor,false);
                                 }
                                if (m_metric == "lncc"){
                                    //lncc= Metrics<ImageType,FloatImageType>::efficientLNCC(warpedImage,targetImage,m_sigma,m_exponent);
                                    lncc= Metrics<ImageType,FloatImageType, double>::efficientLNCC(warpedImage,targetImage,m_sigma,m_exponent);
                                }else if (m_metric == "itklncc"){
                                    lncc= Metrics<ImageType,FloatImageType>::ITKLNCC(warpedImage,targetImage,m_sigma,m_exponent, this->m_ROI);
                                }else if (m_metric == "lsad"){
                                    lncc= Metrics<ImageType,FloatImageType>::LSADNorm(warpedImage,targetImage,m_sigma,m_exponent);
                                }else if (m_metric == "lssd"){
                                    lncc= Metrics<ImageType,FloatImageType>::LSSDNorm(warpedImage,targetImage,m_sigma,m_exponent);
                                }else if (m_metric == "localautocorrelation"){
                                    //lncc= Metrics<ImageType,FloatImageType>::LSSD(warpedImage,(m_imageList)[targetID],m_sigma);
                                    lncc= Metrics<ImageType,FloatImageType>::localMetricAutocorrelation(warpedImage,targetImage,m_sigma,2,"lssd",m_exponent);
                                }else if (m_metric == "gradient"){
                                    //lncc= Metrics<ImageType,FloatImageType>::LSSD(warpedImage,(m_imageList)[targetID],m_sigma);
                                    lncc=  Metrics<ImageType,FloatImageType>::efficientLNCC(warpedImage,targetImage,m_sigma,m_exponent);
                                }else if (m_metric == "categorical"){
                                    lncc=  Metrics<ImageType,FloatImageType>::CategoricalDiffNorm(warpedImage,targetImage,m_sigma,m_exponent);
                                }else{
                                    LOG<<"do not understand "<<VAR(m_metric)<<",aborting."<<endl;
                                    exit(-1);
                                } 
                                lncc=FilterUtils<FloatImageType>::LinearResample(lncc,FilterUtils<ImageType,FloatImageType>::cast(this->m_ROI),false);
                                m_pairwiseLocalWeightMaps[sourceID][targetID]=lncc;
                            }
                        }
                    }
                    //read true deformation if it is a) in the file list and b) we want to either compare it to the estimate or it needs to be cached.
                    if (findDeformation(m_trueDeformationFileList,sourceID,targetID) && (estDef || !m_trueDeformations[sourceID][targetID].IsNotNull())){
                        knownDeformation=ImageUtils<DeformationFieldType>::readImage(m_trueDeformationFileList[sourceID][targetID]);
                        if (estDef){
                            ImagePointerType mask=TransfUtils<ImageType>::createEmptyImage(knownDeformation);
                            mask->FillBuffer(0);
                            typename ImageType::SizeType size=mask->GetLargestPossibleRegion().GetSize();
                            IndexType offset;
                            double fraction=0.9;
                            for (int d=0;d<D;++d){
                                offset[d]=(1.0-fraction)/2*size[d];
                                size[d]=fraction*size[d];
                            }
                        
                            typename ImageType::RegionType region;
                            region.SetSize(size);
                            region.SetIndex(offset);
                            ImageUtils<ImageType>::setRegion(mask,region,1);
                            DeformationFieldPointerType diff=TransfUtils<ImageType>::subtract(updatedDeform,knownDeformation);
                            //m_ADE+=TransfUtils<ImageType>::computeDeformationNorm(diff);
                            m_ADE+=TransfUtils<ImageType>::computeDeformationNormMask(diff,mask);
                        }
                    }


                    //compute ADE
                    //compute inconsistency? baeh.
                    
                    //store updated deformation in the appropriate place
                    if (estDef){
                        //write full resolution deformation to disk
                        ostringstream outfile;
                        if (directory != ""){
                            //outfile<<directory<<"/estimatedLocalComposedDeformationError-FROM-"<<sourceID<<"-TO-"<<targetID<<".mha";
                            //ImageUtils<DeformationFieldType>::writeImage(outfile.str().c_str(),estimatedError);
                            ostringstream outfile2;
                            outfile2<<directory<<"/estimatedLocalComposedDeformation-FROM-"<<sourceID<<"-TO-"<<targetID<<".mha";
                            ImageUtils<DeformationFieldType>::writeImage(outfile2.str().c_str(),updatedDeform);
                        }

                        
                        typedef typename itk::DisplacementFieldJacobianDeterminantFilter<DeformationFieldType,double> DisplacementFieldJacobianDeterminantFilterType;
                        typename DisplacementFieldJacobianDeterminantFilterType::Pointer jacobianFilter = DisplacementFieldJacobianDeterminantFilterType::New();
                        jacobianFilter->SetInput(updatedDeform);
                        jacobianFilter->SetUseImageSpacingOff();
                        jacobianFilter->Update();
                        FloatImagePointerType jac=jacobianFilter->GetOutput();
                        double minJac = FilterUtils<FloatImageType>::getMin(jac);
                        LOGV(2)<<VAR(sourceID)<<" "<<VAR(targetID)<< " " << VAR(minJac) <<" " <<VAR(nCC)<<endl;
                        m_averageMinJac+=minJac;
                        if (minJac<m_minMinJacobian){
                            m_minMinJacobian=minJac;
                        }

                        //sample to correct resolution (downsample)
                        
                        if (m_bSplineInterpol){
                            updatedDeform=TransfUtils<ImageType>::bSplineInterpolateDeformationField(updatedDeform,this->m_ROI,m_smoothDeformationDownsampling);
                            //updatedDeform=TransfUtils<ImageType>::computeBSplineTransformFromDeformationField(updatedDeform,this->m_ROI);
                        }else{
                            updatedDeform=TransfUtils<ImageType>::linearInterpolateDeformationField(updatedDeform,this->m_ROI,m_smoothDeformationDownsampling);
                        }
                        if (m_metric == "gradient"){
                            (m_pairwiseGradients)[sourceID][targetID]=TransfUtils<ImageType>::createEmpty(updatedDeform);
                            double value;
                            computeMetricAndDerivative(targetImage, m_imageList[sourceID] ,updatedDeform ,   (m_pairwiseGradients)[sourceID][targetID] ,  value);
                        }

                        if (updateThisDeformation){//m_updateDeformations ||  !m_downSampledDeformationCache[sourceID][targetID].IsNotNull()){
                            
                            firstRun=true;// !m_downSampledDeformationCache[sourceID][targetID].IsNotNull();
                            m_downSampledDeformationCache[sourceID][targetID] = updatedDeform;

                        }else{

                            (m_updatedDeformationCache)[sourceID][targetID] = updatedDeform;
                            
#if 0
                            locallyUpdateDeformation( m_downSampledDeformationCache[sourceID][targetID],(m_updatedDeformationCache)[sourceID][targetID], m_pairwiseLocalWeightMaps[sourceID][targetID],m_updatedPairwiseLocalWeightMaps[sourceID][targetID]);
#endif

                            m_haveDeformationEstimate = true;
                            //resample deformation if resolution or origin was changed
                            if (this->m_ROI->GetLargestPossibleRegion().GetSize() != m_downSampledDeformationCache[sourceID][targetID]->GetLargestPossibleRegion().GetSize()
                                || this->m_ROI->GetOrigin() != m_downSampledDeformationCache[sourceID][targetID]->GetOrigin() )
                                {
                                    m_downSampledDeformationCache[sourceID][targetID]=TransfUtils<ImageType>::linearInterpolateDeformationField(deformation,this->m_ROI,m_smoothDeformationDownsampling);
                                }

                        }
                        ++count;
                    }

                }//target=source
            }
        }
        m_ADE/=count;
        m_dice/=count;
        m_TRE/=count;
        m_averageMinJac/=count;
        m_averageNCC/=count;
        //LOG<<VAR(m_averageMinJac)<<" "<<VAR(minMinJac)<<" "<<VAR(m_averageNCC)<<endl;
        if (m_updateDeformations || firstRun ){
            m_Inconsistency=TransfUtils<ImageType>::computeInconsistency(&m_downSampledDeformationCache,&m_imageIDList, &m_trueDeformations,m_maskList);
        }else{
            m_Inconsistency=TransfUtils<ImageType>::computeInconsistency(&m_updatedDeformationCache,&m_imageIDList, &m_trueDeformations,m_maskList);
        }
    }//DoALot
    
    
    
    double computeTRE(string targetLandmarks, string refLandmarks, DeformationFieldPointerType def,ImagePointerType reference){
        typedef typename  ImageType::DirectionType DirectionType;
        typedef typename itk::VectorLinearInterpolateImageFunction<DeformationFieldType> DefInterpolatorType;
        typedef typename DefInterpolatorType::ContinuousIndexType CIndexType;
        PointType p;
        p.Fill(0.0);
        typename DefInterpolatorType::Pointer defInterpol=DefInterpolatorType::New();
        defInterpol->SetInputImage(def);
        ifstream ifs(refLandmarks.c_str());
        int i=0;
        double TRE=0.0;
        int count=0;
        vector<PointType> landmarksReference, landmarksTarget;
        DirectionType refDir=reference->GetDirection();
        DirectionType targetDir=def->GetDirection();

        while ( not ifs.eof() ) {
            PointType point;
            for (int d=0;d<D;++d){
                ifs>>point[d];
                point[d]=point[d]*refDir[d][d];
            }
            //LOG<<point<<endl;
            landmarksReference.push_back(point);
            
        } 
        //std::cout<<"read "<<landmarksReference.size()<<" landmarks"<<std::endl;
        ifstream ifs2(targetLandmarks.c_str());
        i=0;
        for (;i<landmarksReference.size()-1;++i){
            PointType pointTarget;
            for (int d=0;d<D;++d){
                ifs2>>pointTarget[d];
                pointTarget[d]=pointTarget[d]*targetDir[d][d];
            }        
            IndexType indexTarget,indexReference;
            def->TransformPhysicalPointToIndex(pointTarget,indexTarget);
           
            PointType deformedReferencePoint;
            reference->TransformPhysicalPointToIndex(landmarksReference[i],indexReference);
                        
          
            CIndexType cindex;
            def->TransformPhysicalPointToContinuousIndex(pointTarget,cindex);
            if (def->GetLargestPossibleRegion().IsInside(cindex)){
                deformedReferencePoint= pointTarget+defInterpol->EvaluateAtContinuousIndex(cindex);
                double localError=(deformedReferencePoint - landmarksReference[i]).GetNorm();
                LOGI(2,std::cout<<"pt"<<i<<": "<<(localError)<<" ");
                TRE+=localError;
                ++count;
            }
        }
        LOGI(2,std::cout<<std::endl);
        return TRE/count;
    }


    double computeLandmarkRegistrationError(DeformationCacheType * deformations, map<string,string> landmarkFilenames,std::vector<string> imageIDs, ImageCacheType * images){

        int nImages=imageIDs.size();
        typedef typename  ImageType::DirectionType DirectionType;
        
        typedef typename itk::VectorLinearInterpolateImageFunction<DeformationFieldType> DefInterpolatorType;
        
        typedef typename DefInterpolatorType::ContinuousIndexType CIndexType;
        
        PointType p;
        p.Fill(0.0);
        
        double sumSquareError=0.0;
        int count = 0;

        for (int source=0;source<nImages;++source){
            string sourceID=imageIDs[source];
            for (int target=0;target<nImages;++target){
                if (source!=target){
                    string targetID=imageIDs[target];
                    DeformationFieldPointerType def=(deformations)[sourceID][targetID];
                    ImagePointerType reference=(images)[targetID];
                    DirectionType refDir=reference->GetDirection();

                    if (def->GetLargestPossibleRegion().GetSize() != reference->GetLargestPossibleRegion().GetSize()){
                        def=TransfUtils<ImageType>::linearInterpolateDeformationField(def,reference);
                    }
                    typename DefInterpolatorType::Pointer defInterpol=DefInterpolatorType::New();
        
                    defInterpol->SetInputImage(def);
                    DirectionType targetDir=def->GetDirection();
                    vector<PointType> landmarksReference, landmarksTarget;
                    string refLandmarks=landmarkFilenames[sourceID];
                    string targetLandmarks=landmarkFilenames[targetID];
                    ifstream ifs(refLandmarks.c_str());
                    int i=0;
                      
                    while ( not ifs.eof() ) {
                        PointType point;
                        for (int d=0;d<D;++d){
                            ifs>>point[d];
                            point[d]=point[d]*refDir[d][d];
                        }
                        //LOG<<point<<endl;
                        landmarksReference.push_back(point);
                          
                    } 
                    //std::cout<<"read "<<landmarksReference.size()<<" landmarks"<<std::endl;
                    ifstream ifs2(targetLandmarks.c_str());
                    i=0;
                    for (;i<landmarksReference.size()-1;++i){
                        PointType pointTarget;
                        for (int d=0;d<D;++d){
                            ifs2>>pointTarget[d];
                            pointTarget[d]=pointTarget[d]*targetDir[d][d];
                        }        
                        IndexType indexTarget,indexReference;
                        def->TransformPhysicalPointToIndex(pointTarget,indexTarget);
                        //LOG<<VAR(def->GetOrigin())<<endl;
                        //LOG<<VAR(pointTarget)<<" "<<VAR(indexTarget)<<endl;
                        PointType deformedReferencePoint;
                        reference->TransformPhysicalPointToIndex(landmarksReference[i],indexReference);
                          
                        //std::cout<<VAR(targetPoint)<<endl;
                        //deformedReferencePoint= pointTarget+def->GetPixel(indexTarget);
                        CIndexType cindex;
                        def->TransformPhysicalPointToContinuousIndex(pointTarget,cindex);
                        //LOG<<VAR(landmarksReference[i])<<" "<<VAR(indexReference)<<" "<<VAR(cindex)<<endl;
                        if (def->GetLargestPossibleRegion().IsInside(cindex)){
                            deformedReferencePoint= pointTarget+defInterpol->EvaluateAtContinuousIndex(cindex);
                              
                            //LOG<< VAR(pointTarget) << endl;
                            double localSquaredError=(deformedReferencePoint - landmarksReference[i]).GetNorm();
                           
                              
                            LOGI(2,std::cout<<"pt"<<i<<": "<<(localSquaredError)<<" ");
                            sumSquareError+=localSquaredError;
                            ++count;
                        }
                    }
                    LOGI(2,std::cout<<endl);
                      
                }
            }
        }
        return sumSquareError/count;
      
    }  

    void computeMetricAndDerivative(ImagePointerType img1, ImagePointerType img2, DeformationFieldPointerType def, DeformationFieldPointerType deriv, double & value){

        typedef typename itk::CorrelationImageToImageMetricv4<FloatImageType,FloatImageType> MetricType;
        //typedef typename itk::MeanSquaresImageToImageMetricv4<FloatImageType,FloatImageType> MetricType;
        typedef typename MetricType::Pointer MetricPointer;
        typedef typename MetricType::DerivativeType MetricDerivativeType;

        //typedef itk::RegistrationParameterScalesFromJacobian<MetricType> ScalesEstimatorType;
        typedef itk::RegistrationParameterScalesFromPhysicalShift<MetricType> ScalesEstimatorType;
        typedef typename ScalesEstimatorType::Pointer ScalesEstimatorPointer;

        m_resolutionFactor=4.0*def->GetLargestPossibleRegion().GetSize()[0]/img1->GetLargestPossibleRegion().GetSize()[0];
        m_resolutionFactor=min(1.0,m_resolutionFactor);
        FloatImagePointerType fimg1=FilterUtils<ImageType,FloatImageType>::LinearResample(img1,m_resolutionFactor,true);
        FloatImagePointerType fimg2=FilterUtils<ImageType,FloatImageType>::LinearResample(img2,m_resolutionFactor,true);
        typename TransfUtils<ImageType,float,double>::OutputDeformationFieldPointerType dblDef=TransfUtils<ImageType,float,double>::cast(def);

#if 0
        typedef typename  itk::DisplacementFieldTransform<double, D> DisplacementFieldTransformType;
        typedef typename DisplacementFieldTransformType::Pointer DisplacementFieldTransformPointer;
        DisplacementFieldTransformPointer defTransf=DisplacementFieldTransformType::New();
        defTransf->SetDisplacementField(dblDef);
        //    LOG<<defTransf<<endl;
#else

        typedef typename  itk::BSplineDeformableTransform<double, D,3> BSplineDeformableTransformType;
        typedef typename BSplineDeformableTransformType::Pointer BSplineDeformableTransformPointer;
        typename BSplineDeformableTransformType::ImagePointer paramImages[D];
         
        BSplineDeformableTransformPointer defTransf=BSplineDeformableTransformType::New();
        for (int d=0;d<D;++d){
            paramImages[d]=TransfUtils<ImageType,double,double,double>::getComponent(dblDef,d);
        }
        defTransf->SetCoefficientImages(paramImages);
#endif
        //LOG<<VAR(defTransf->GetNumberOfParameters())<<endl;
         
     
        MetricPointer metric=MetricType::New();
        metric->SetFixedImage(fimg1);
        metric->SetMovingImage(fimg2);
        metric->SetTransform(defTransf);
       
        metric->Initialize();
        MetricDerivativeType derivative;
        metric->GetValueAndDerivative(value, derivative);

        ScalesEstimatorPointer scalesEstimator=ScalesEstimatorType::New();
        scalesEstimator->SetMetric(metric);
        typename ScalesEstimatorType::ScalesType scales,localScales;
        scalesEstimator->EstimateScales(scales);
        //scalesEstimator->EstimateLocalScales(derivative,localScales);
        float learningRate;
        float stepScale;
        float maxStepSize;
        //modify gradient by scales
        for (int i=0;i<derivative.size();++i){
            LOGV(4)<<derivative[i]<<"  "<<i<<" "<<i%scales.size()<<" "<<scales[i%scales.size()]<<endl;
            derivative[i]/=scales[i%scales.size()];
        }
        stepScale=scalesEstimator->EstimateStepScale(derivative);
        //estimate learning rate
        maxStepSize=scalesEstimator->EstimateMaximumStepSize();
        learningRate=maxStepSize/stepScale;
        
        LOGV(2)<<VAR(value)<<endl;
        

        //deriv=TransfUtils<ImageType>::createEmpty(def);
        int numberOfPixels=deriv->GetBufferedRegion().GetNumberOfPixels();
        typedef typename itk::ImageRegionIterator<DeformationFieldType> DeformationIteratorType;
        DeformationIteratorType defIt(deriv,deriv->GetLargestPossibleRegion());
        for (int i=0;i<10;++i){
            int p=0;
            for (defIt.GoToBegin();!defIt.IsAtEnd();++defIt,++p){
                DeformationType disp;
                for (int d=0;d<D;++d){
                    LOGV(4)<<VAR(p+d*numberOfPixels)<<" "<<VAR(derivative[p+d*numberOfPixels])<<" "<<VAR(scales[p+d*numberOfPixels])<<" "<<VAR(stepScale)<<" "<<VAR(maxStepSize)<<" "<<VAR(learningRate)<<endl;
                    disp[d]=-derivative[p+d*numberOfPixels]*learningRate;///scales[p+d*numberOfPixels];
                }
                defIt.Set(disp);

            }
            typename TransfUtils<ImageType,float,double>::OutputDeformationFieldPointerType newDef=TransfUtils<ImageType,double>::add(dblDef,TransfUtils<ImageType,float,double>::cast(deriv));
            for (int d=0;d<D;++d){
                paramImages[d]=TransfUtils<ImageType,double,double,double>::getComponent(newDef,d);
            }
            defTransf->SetCoefficientImages(paramImages);
            metric->SetTransform(defTransf);
            metric->Initialize();
            double newValue=metric->GetValue();
            LOGV(1)<<VAR(i)<<" "<<VAR(value)<<" "<<VAR(newValue)<<" "<<VAR(learningRate)<<endl;
            if (newValue<value){
                return;
            }else{
                learningRate/=2;
            }
        }
        ImageUtils<DeformationFieldType>::writeImage("derivative.mha",deriv);
    }

    void locallyUpdateDeformation(DeformationFieldPointerType & def, DeformationFieldPointerType &updatedDef, FloatImagePointerType & similarity, FloatImagePointerType updatedSimilarity){
        typedef typename itk::ImageRegionIterator<DeformationFieldType> DeformationIteratorType;
        DeformationIteratorType defIt(def,def->GetLargestPossibleRegion());
        DeformationIteratorType updatedDefIt(updatedDef,def->GetLargestPossibleRegion());
        typedef typename itk::ImageRegionIterator<FloatImageType> FloatImageIteratorType;
        FloatImageIteratorType simIt(similarity,similarity->GetLargestPossibleRegion());
        FloatImageIteratorType updatedSimIt(updatedSimilarity,similarity->GetLargestPossibleRegion());
        defIt.GoToBegin();
        updatedDefIt.GoToBegin();
        simIt.GoToBegin();
        updatedSimIt.GoToBegin();
        for (;!defIt.IsAtEnd();++simIt,++updatedSimIt,++defIt,++updatedDefIt){
            
            if (updatedSimIt.Get()>=simIt.Get()){
                simIt.Set(updatedSimIt.Get());
                defIt.Set(updatedDefIt.Get());
            }

        }


    }

};
