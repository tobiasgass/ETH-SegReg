/**
 * @file   Registration-Propagation-MRF.h
 * @author Tobias Gass <tobiasgass@gmail.com>
 * @date   Thu Mar 12 12:26:30 2015
 * 
 * @brief  Registration propagation using MRF and optionally LWA fusion
 * 
 * 
 */

#pragma once

#include <stdio.h>
#include <iostream>

#include "ArgumentParser.h"
#include "Log.h"
#include <vector>
#include <map>
#include "itkImageRegionIterator.h"
#include "TransformationUtils.h"
#include "ImageUtils.h"
#include "FilterUtils.hpp"
#include <sstream>
#include "ArgumentParser.h"
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include "itkConstNeighborhoodIterator.h"
#include "itkDisplacementFieldTransform.h"
#include "itkNormalizedCorrelationImageToImageMetric.h"
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkMattesMutualInformationImageToImageMetric.h"
#include "itkNormalizedMutualInformationHistogramImageToImageMetric.h"
#include "itkLinearInterpolateImageFunction.h"
#include <itkAddImageFilter.h>
#include "Metrics.h"
#include "itkLabelOverlapMeasuresImageFilter.h"
#include "MRFRegistrationFuser.h"
#include <itkDisplacementFieldJacobianDeterminantFilter.h>
#include "SegmentationMapper.hxx"


namespace MRegFuse{

/**
 * @brief  Registration propagation in a set of N images using MRF and optionally LWA fusion
 */
template <class ImageType>
class RegistrationPropagationMRF{
public:
    typedef typename ImageType::PixelType PixelType;
    static const unsigned int D=ImageType::ImageDimension;
    typedef typename  ImageType::Pointer ImagePointerType;
    typedef typename  ImageType::IndexType IndexType;
    typedef typename  ImageType::PointType PointType;
    typedef typename  ImageType::OffsetType OffsetType;
    typedef typename  ImageType::SizeType SizeType;
    typedef typename  ImageType::ConstPointer ImageConstPointerType;
    typedef typename  ImageUtils<ImageType,double>::FloatImageType FloatImageType;
    typedef typename  FloatImageType::Pointer FloatImagePointerType;

    typedef typename  TransfUtils<ImageType>::DisplacementType DisplacementType;
    typedef typename  TransfUtils<ImageType>::DeformationFieldType DeformationFieldType;
    typedef typename  DeformationFieldType::Pointer DeformationFieldPointerType;
    typedef typename  itk::ImageRegionIterator<ImageType> ImageIteratorType;
    typedef typename  itk::ImageRegionIterator<FloatImageType> FloatImageIteratorType;
    typedef typename  itk::ImageRegionIterator<DeformationFieldType> DeformationIteratorType;
    typedef typename  itk::ConstNeighborhoodIterator<ImageType> ImageNeighborhoodIteratorType;
    typedef   ImageNeighborhoodIteratorType * ImageNeighborhoodIteratorPointerType;
    typedef typename  ImageNeighborhoodIteratorType::RadiusType RadiusType;

    typedef MRFRegistrationFuser<ImageType,double> RegistrationFuserType;
    typedef typename itk::DisplacementFieldJacobianDeterminantFilter<DeformationFieldType,double> DisplacementFieldJacobianDeterminantFilterType;

    typedef typename itk::AddImageFilter<DeformationFieldType,DeformationFieldType,DeformationFieldType> DeformationAddFilterType;
    typedef map<string,ImagePointerType> ImageCacheType;
    typedef map<string, map< string, string> > FileListCacheType;
    enum MetricType {MAD,NCC,MSD};
    enum WeightingType {UNIFORM,GLOBAL,LOCAL};
protected:
    double m_gamma;
    RadiusType m_patchRadius;
public:
    int run(int argc, char ** argv){
        feraiseexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
        ArgumentParser * as=new ArgumentParser(argc,argv);
        string trueDefListFilename="",landmarkFileList="",groundTruthSegmentationFileList="",deformationFileList,imageFileList,atlasSegmentationFileList,supportSamplesListFileName="",outputDir="",outputSuffix="",weightListFilename="";
        int verbose=0;
        int radius=3;
        double controlGridSpacingFactor=4;
        int maxHops=1;
        string metricName="NCC";
        string weightingName="local";
        bool lateFusion=false;
        bool dontCacheDeformations=false;
        double alpha=0;
        m_gamma=10;
        double m_pairwiseWeight=1.0;
        bool useHardConstraints=false;
        bool estimateMRF=false,estimateMean=false;
        int refineIter=0;
        string source="",target="";
        bool runEndless=false;
        bool indivCompare=false;
        int nKernels=20;
        int refineSeamIter=0;
        double smoothIncrease=1.2;
        bool useMaskForSSR=false;
        //as->parameter ("A",atlasSegmentationFileList , "list of atlas segmentations <id> <file>", true);
        as->option ("MRF", estimateMRF, "use MRF fusion");
        as->option ("mean", estimateMean, "use (local) mean fusion. Can be used in addition to MRF or stand-alone.");
        as->parameter ("T", deformationFileList, " list of deformations", true);
        as->parameter ("i", imageFileList, " list of  images", true);
        as->parameter ("true", trueDefListFilename, " list of TRUE deformations", false);
        //as->parameter ("W", weightListFilename,"list of weights for deformations",false);
        as->parameter ("metric", metricName,"metric to be used for global or local weighting, valid: NONE,SAD,MSD,NCC,MI,NMI",false);
        as->parameter ("weighting", weightingName,"internal weighting scheme {uniform,local,global}. non-uniform will only work with metric != NONE",false);
        as->parameter ("g", m_gamma,"gamma for exp(- metric/gamma)[SSD,SAD,..] or metric^gamma [NCC] ",false);
        as->parameter ("w", m_pairwiseWeight,"pairwise weight for MRF",false);
        as->parameter ("radius", radius,"patch radius for local metrics",false);
        as->parameter ("controlGridSpacing", controlGridSpacingFactor,"Resolution of the MRF control grid, obtained by dividing the input image resolution by this factor.",false);
        as->option ("hardConstraints", useHardConstraints,"Use hard constraints in the MRF to prevent folding.");
        as->parameter ("refineIter", refineIter,"refine MRF solution by adding the result as new labels to the MRF and re-solving until convergence.",false);
        as->parameter ("O", outputDir,"outputdirectory (will be created + no overwrite checks!)",false);
        as->parameter ("maxHops", maxHops,"maximum number of hops",false);
        as->parameter ("alpha", alpha,"pairwise balancing weight (deformation (alpha=0) vs label smoothness (alpha=1))",false);
        //  as->option ("dontCacheDeformations", dontCacheDeformations,"read deformations only when needed to save memory. higher IO load!");
        as->parameter ("groundTruthSegmentations",groundTruthSegmentationFileList , "list of groundTruth segmentations <id> <file>", false);
        as->parameter ("landmarks",landmarkFileList , "list of landmark files <id> <file>", false);       
        as->parameter ("source",source , "source ID, will only compute updated registrations for <source>", false);       
        as->parameter ("target",target , "target ID, will only compute updated registrations for <source>", false);       
        as->option ("noCaching", dontCacheDeformations, "do not cache Deformations. will yield a higher IO load as some deformations need to be read multiple times.");
        as->option ("runEndless", runEndless, "do not check for convergence.");
        as->option ("indivCompare", indivCompare, "individually compare pre- and post registration similarity, and only update if sim has improved or stayed the same.");
        as->parameter ("refineSeamIter", refineSeamIter,"refine MRF solution at seams by smoothing the result and fusing it with the original solution.",false);
        as->parameter ("smoothIncrease", smoothIncrease,"factor to increase smoothing with per iteration for SSR.",false);
        as->option ("useMask", useMaskForSSR,"only update pixels with negative jac dets (or in the vincinity of those) when using SSR.");
        //        as->option ("graphCut", graphCut,"use graph cuts to generate final segmentations instead of locally maximizing");
        //as->parameter ("smoothness", smoothness,"smoothness parameter of graph cut optimizer",false);
        as->parameter ("verbose", verbose,"get verbose output",false);
        as->help();
        as->parse();
       
        if (!estimateMRF && !estimateMean){
            LOG<<"Neither MRF nor mean estimation activated, will compute metrics for original deformations and exit"<<endl;
            maxHops=1;
            //estimateMRF=true;
        }
        LOG<<VAR(estimateMRF)<<" "<<VAR(estimateMean)<<endl;
        for (unsigned int i = 0; i < ImageType::ImageDimension; ++i) m_patchRadius[i] = radius;

        if (dontCacheDeformations)
            maxHops=1;

        mkdir(outputDir.c_str(),0755);
        logSetStage("IO");
        logSetVerbosity(verbose);
        
        MetricType metric;
        if (metricName=="MSD")
            metric=MSD;
        else if (metricName=="MAD")
            metric=MAD;
        else if (metricName=="NCC")
            metric=NCC;
        else{
            LOG<<"don't understand "<<metricName<<", aborting"<<endl;
	    exit(0);
        }
        WeightingType weighting;
        if (weightingName=="uniform" ){
            weighting=UNIFORM;}
        else if (weightingName=="global")
            weighting=GLOBAL;
        else if (weightingName=="local"){
            weighting=LOCAL;
           
        }
        else{
            LOG<<"don't understand "<<VAR(weightingName)<<", defaulting to uniform."<<endl;
            weighting=UNIFORM;
        }

        if (metric==MAD || metric==MSD){
            if (m_gamma ==0.0){
                weighting=UNIFORM;
            }
        }

        map<string,ImagePointerType> inputImages;
        typedef typename map<string, ImagePointerType>::iterator ImageListIteratorType;
        LOG<<"Reading input images."<<endl;
        std::vector<string> imageIDs;
     
        inputImages = ImageUtils<ImageType>::readImageList( imageFileList , imageIDs);
        
        LOGV(2)<<VAR(metric)<<" "<<VAR(weighting)<<endl;
        LOGV(2)<<VAR(m_gamma)<<" "<<VAR(lateFusion)<<" "<<VAR(m_patchRadius)<<endl;

        if (dontCacheDeformations){
            LOG<<"Reading deformation file names."<<endl;
        }else{
            LOG<<"CACHING all deformations!"<<endl;
        }
        map< string, map <string, DeformationFieldPointerType> > deformationCache, trueDeformations;
        map< string, map <string, string> > deformationFilenames,trueDeformationFilenames;
        map<string, map<string, float> > globalWeights;
        {
            ifstream ifs(deformationFileList.c_str());
            while (!ifs.eof()){
                string sourceID,targetID,defFileName;
                ifs >> sourceID;
                if (sourceID!=""){
                    ifs >> targetID;
                    ifs >> defFileName;
                    if (inputImages.find(sourceID)==inputImages.end() || inputImages.find(targetID)==inputImages.end() ){
                        LOGV(1)<<sourceID<<" or "<<targetID<<" not in image database, skipping"<<endl;
                        //exit(0);
                    }else{
                        if (!dontCacheDeformations){
                            LOGV(3)<<"Reading deformation "<<defFileName<<" for deforming "<<sourceID<<" to "<<targetID<<endl;
                            deformationCache[sourceID][targetID]=ImageUtils<DeformationFieldType>::readImage(defFileName);
                            globalWeights[sourceID][targetID]=1.0;
                        }else{
                            LOGV(3)<<"Reading filename "<<defFileName<<" for deforming "<<sourceID<<" to "<<targetID<<endl;
                            deformationFilenames[sourceID][targetID]=defFileName;
                            globalWeights[sourceID][targetID]=1.0;
                        }
                    }
                }
            }
        }
        if (trueDefListFilename!=""){
            ifstream ifs(trueDefListFilename.c_str());
            while (!ifs.eof()){
                string sourceID,targetID,defFileName;
                ifs >> sourceID;
                if (sourceID!=""){
                    ifs >> targetID;
                    ifs >> defFileName;
                    if (inputImages.find(sourceID)==inputImages.end() || inputImages.find(targetID)==inputImages.end() ){
                        LOGV(1)<<sourceID<<" or "<<targetID<<" not in image database, skipping"<<endl;
                        //exit(0);
                    }else{
                        if (!dontCacheDeformations){
                            LOGV(3)<<"Reading TRUE deformation "<<defFileName<<" for deforming "<<sourceID<<" to "<<targetID<<endl;
                            trueDeformations[sourceID][targetID]=ImageUtils<DeformationFieldType>::readImage(defFileName);
                          
                        }else{
                            trueDeformationFilenames[sourceID][targetID]=defFileName;

                        }
                    }
                }
            }
            
        }
        if (weightListFilename!=""){
            ifstream ifs(weightListFilename.c_str());
            while (!ifs.eof()){
                string sourceID,targetID;
                ifs >> sourceID;
                ifs >> targetID;
                if (inputImages.find(sourceID)==inputImages.end() || inputImages.find(targetID)==inputImages.end() ){
                    LOG << sourceID<<" or "<<targetID<<" not in image database while reading weights, skipping"<<endl;
                }else{
                    ifs >> globalWeights[sourceID][targetID];
                }
            }
        }

        //read landmark filenames
        map<string,string> m_landmarkFileList;
        if (landmarkFileList!=""){
            ifstream ifs(landmarkFileList.c_str());
            while (!ifs.eof()){
                string sourceID,landmarkFilename;
                ifs >> sourceID;
                ifs >>landmarkFilename;
                m_landmarkFileList[sourceID]=landmarkFilename;
            }
            
        }
        
        ImageCacheType m_groundTruthSegmentations;
        if (groundTruthSegmentationFileList!=""){
            std::vector<string> buff;
            m_groundTruthSegmentations=ImageUtils<ImageType>::readImageList(groundTruthSegmentationFileList,buff);
        }

        double error;
        double inconsistency;
        error=TransfUtils<ImageType>::computeError(&deformationCache,&trueDeformations,&imageIDs);
        //inconsistency = TransfUtils<ImageType>::computeInconsistency(&deformationCache,&imageIDs, &trueDeformations);
        int iter = 0;
        LOG<<VAR(iter)<<" "<<VAR(error)<<" "<<VAR(inconsistency)<<endl;
        
        double m_oldEnergy=std::numeric_limits<double>::max();
        double m_oldSimilarity=std::numeric_limits<double>::max();
        logSetStage("Zero Hop");
        LOGV(1)<<"Computing"<<std::endl;
        for (iter=1;iter<maxHops+1;++iter){
            map< string, map <string, DeformationFieldPointerType> > TMPdeformationCache;
            double m_dice=0.0;
            double m_volumeWeightedDice=0.0;
            double m_TRE=0.0;
            double m_energy=0.0;
            double m_similarity=0.0;
            double m_averageMinJac=0;
            double m_minMinJacobian=100000000;
            int count=0;
            for (ImageListIteratorType sourceImageIterator=inputImages.begin();sourceImageIterator!=inputImages.end();++sourceImageIterator){           
                //iterate over sources
                string sourceID= sourceImageIterator->first;
                //skip source images if only one source should be evaluated
                if (source != "" && sourceID!=source)
                    continue;
                for (ImageListIteratorType targetImageIterator=inputImages.begin();targetImageIterator!=inputImages.end();++targetImageIterator){                //iterate over targets
                    string targetID= targetImageIterator->first;
                    //skip target image if only one target should be evaluated
                    if (target !="" && target!=targetID)
                        continue;
                    if (targetID !=sourceID){
                        ++count;
                        DeformationFieldPointerType result;
                        DeformationFieldPointerType deformationSourceTarget;
                        if (dontCacheDeformations){
                            deformationSourceTarget = ImageUtils<DeformationFieldType>::readImage(deformationFilenames[sourceID][targetID]);
                        }else{
                            deformationSourceTarget = deformationCache[sourceID][targetID];
                        }
                        
                        double initialSimilarity;
                        ImagePointerType warpedSourceImage=TransfUtils<ImageType>::warpImage(sourceImageIterator->second,deformationSourceTarget);
                        switch(metric){
                        case NCC:
                            initialSimilarity=Metrics<ImageType,FloatImageType>::nCC(warpedSourceImage,targetImageIterator->second);
                            break;
                        case MSD:
                            initialSimilarity=Metrics<ImageType,FloatImageType>::msd(warpedSourceImage,targetImageIterator->second);
                            break;
                        case MAD:
                            initialSimilarity=Metrics<ImageType,FloatImageType>::mad(warpedSourceImage,targetImageIterator->second);
                            break;
                        }
                        
                        
                        if (estimateMRF || estimateMean){

                            RegistrationFuserType estimator;
                            estimator.setAlpha(alpha);
                            estimator.setPairwiseWeight(m_pairwiseWeight);
                            estimator.setGridSpacing(controlGridSpacingFactor);
                            estimator.setHardConstraints(useHardConstraints);
                       
                            GaussianEstimatorVectorImage<ImageType,double> meanEstimator;
                            FloatImagePointerType weightImage=addImage(weightingName,metric,estimator,meanEstimator,targetImageIterator->second,sourceImageIterator->second,deformationSourceTarget,estimateMean,estimateMRF,radius,m_gamma);
                            if (weightImage.IsNotNull() && outputDir!=""){
                                ostringstream oss;
                                oss<<outputDir<<"/lncc-"<<sourceID<<"-TO-"<<targetID<<".mha";
                                LOGI(2,ImageUtils<FloatImageType>::writeImage(oss.str(),weightImage));
                            }
                            for (ImageListIteratorType intermediateImageIterator=inputImages.begin();intermediateImageIterator!=inputImages.end();++intermediateImageIterator){                //iterate over intermediates
                                string intermediateID= intermediateImageIterator->first;
                                if (targetID != intermediateID && sourceID!=intermediateID){
                                    //get all deformations for full circle
                                    DeformationFieldPointerType deformationSourceIntermed;
                                    DeformationFieldPointerType deformationIntermedTarget;
                                    if (dontCacheDeformations){
                                        deformationSourceIntermed = ImageUtils<DeformationFieldType>::readImage(deformationFilenames[sourceID][intermediateID]);
                                        deformationIntermedTarget = ImageUtils<DeformationFieldType>::readImage(deformationFilenames[intermediateID][targetID]);
                                    }else{
                                        deformationSourceIntermed = deformationCache[sourceID][intermediateID];
                                        deformationIntermedTarget = deformationCache[intermediateID][targetID];
                                    }
                                    LOGV(3)<<"Adding "<<VAR(sourceID)<<" "<<VAR(targetID)<<" "<<VAR(intermediateID)<<endl;
                                    DeformationFieldPointerType indirectDef = TransfUtils<ImageType>::composeDeformations(deformationIntermedTarget,deformationSourceIntermed);
                                    FloatImagePointerType weightImage=addImage(weightingName,metric,estimator,meanEstimator,targetImageIterator->second,sourceImageIterator->second,indirectDef,estimateMean,estimateMRF,radius,m_gamma);
                                    if (weightImage.IsNotNull() && outputDir!=""){
                                        ostringstream oss;
                                        oss<<outputDir<<"/lncc-"<<sourceID<<"-TO-"<<targetID<<"-via-"<<intermediateID<<".mha";
                                        LOGI(4,ImageUtils<FloatImageType>::writeImage(oss.str(),weightImage));
                                    }

                                }//if
                            }//intermediate image
                            ImagePointerType labelImage;
                            
                            if (estimateMean)
                                meanEstimator.finalize();

                            if (! estimateMRF){
                                DeformationFieldPointerType meanDef=meanEstimator.getMean();
                                result=meanDef;
                            }else{

                                if (estimateMean){
                                    DeformationFieldPointerType meanDef=meanEstimator.getMean();
                                    addImage(weightingName,metric,estimator,meanEstimator,targetImageIterator->second,sourceImageIterator->second,meanDef,estimateMean,estimateMRF,radius,m_gamma);

                                }
                                
                                estimator.finalize();
                                double energy=estimator.solve();
                                result=estimator.getMean();
                                labelImage=estimator.getLabelImage();
                                LOGV(1)<<VAR(energy)<<endl;
                                if (refineSeamIter>0){


                                    typename DisplacementFieldJacobianDeterminantFilterType::Pointer jacobianFilter = DisplacementFieldJacobianDeterminantFilterType::New();
                                    jacobianFilter->SetInput(result);
                                    jacobianFilter->SetUseImageSpacingOff();
                                    jacobianFilter->Update();
                                    FloatImagePointerType jac=jacobianFilter->GetOutput();
                                    double minJac = FilterUtils<FloatImageType>::getMin(jac);
                                    if (minJac<0){
                                        if (outputDir!=""){
                                            ostringstream oss2;
                                            oss2<<outputDir<<"/jacobianDetWithNegVals-"<<sourceID<<"-TO-"<<targetID<<".mha";
                                            LOGI(3,ImageUtils<FloatImageType>::writeImage(oss2.str(),jac));
                                        }
                                    

                                        LOGV(1)<<"Refining seams by smoothing solution with maximally "<<nKernels<<" kernels.."<<endl;
                                    DeformationFieldPointerType originalFusionResult=result;
                                    RegistrationFuserType seamEstimator;
                                    double kernelBaseWidth=0.5;//1.0;//pow(-1.0*minJac,1.0/D);

                                    seamEstimator.setAlpha(alpha);
                                    //seamEstimator.setGridSpacing(1);
                                    seamEstimator.setGridSpacing(controlGridSpacingFactor);
                                    seamEstimator.setHardConstraints(useHardConstraints);
                                    seamEstimator.setAnisoSmoothing(false);
                                    //seamEstimator.setAlpha(pow(2.0,1.0*iter)*alpha);
                                    
                                    //hacky shit to avoid oversmoothing
                                    FloatImagePointerType weights=addImage(weightingName,metric,estimator,meanEstimator,targetImageIterator->second,sourceImageIterator->second,originalFusionResult,false,estimateMRF,radius,m_gamma);
                                    seamEstimator.addImage(estimator.getLowResResult(),weights);

                                    //kernelGammas= kernelBaseWidth/4,kbw/2,kbw,2*kbw,4*kbw
                                    int k=0;
                                    DeformationFieldPointerType smoothedResult=result;
                                    kernelBaseWidth=0.5;
                                    double exp=2;
                                    double previousGamma=0.0;
                                    double kernelGamma;
#ifdef USELOCALSIGMASFORDILATION
                                    ImagePointerType negJacMaskPrevious=FilterUtils<FloatImageType,ImageType>::binaryThresholdingHigh(jac,0.0);
                                    FloatImagePointerType localKernelWidths=ImageUtils<FloatImageType>::createEmpty(jac);
                                    localKernelWidths->FillBuffer(0.0);
#endif
                                    for (;k<nKernels;++k){
                                        //double kernelGamma=kernelBaseWidth*(k+1);//pow(2.0,1.0*(k));
                                        LOGV(3)<<VAR(k)<<endl;
                                        kernelGamma=kernelBaseWidth*pow(exp,1.0*(k));
                                        LOGV(3)<<VAR(kernelGamma)<<endl;

                                        double actualGamma=sqrt(pow(kernelGamma,2.0)-pow(previousGamma,2.0));
                                        LOGV(3)<<VAR(actualGamma)<<endl;
                                        smoothedResult=TransfUtils<ImageType>::gaussian(smoothedResult,actualGamma);
                                        previousGamma=actualGamma;
                                        LOGV(3)<<"Smoothed result with gaussian.."<<endl;
                                        addImage(weightingName,metric,seamEstimator,meanEstimator,targetImageIterator->second,sourceImageIterator->second,smoothedResult,false,estimateMRF,radius,m_gamma);
                                        typename DisplacementFieldJacobianDeterminantFilterType::Pointer jacobianFilter = DisplacementFieldJacobianDeterminantFilterType::New();
                                        jacobianFilter->SetInput(smoothedResult);
                                        jacobianFilter->Update();
                                        FloatImagePointerType jac=jacobianFilter->GetOutput();
                                        double minJac2 = FilterUtils<FloatImageType>::getMin(jac);
                                        LOGV(3)<<VAR(minJac2)<<endl;
#ifdef USELOCALSIGMASFORDILATION
                                        //get negative jacobian value locations
                                        ImagePointerType negJacMask=FilterUtils<FloatImageType,ImageType>::binaryThresholdingHigh(jac,0.0);
                                        //subtract and invert to get locations of removed negative JDs
                                        //ImagePointerType removedNegJacMask=FilterUtils<ImageType>::substract(negJacMaskPrevious,negJacMask));
                                        ImagePointerType removedNegJacMask=FilterUtils<ImageType>::binaryThresholding(FilterUtils<ImageType>::add(negJacMaskPrevious,negJacMask),1,1);
                                        //create image with gamma at locations where nJDs were removed
                                        FloatImagePointerType kernelWidthForRemovednJDs=ImageUtils<FloatImageType>::createEmpty(jac);
                                        kernelWidthForRemovednJDs->FillBuffer(3.0*kernelGamma);
                                        kernelWidthForRemovednJDs=ImageUtils<FloatImageType>::multiplyImageOutOfPlace(kernelWidthForRemovednJDs,FilterUtils<ImageType,FloatImageType>::cast(removedNegJacMask));
                                        //add to localKernelWidths image
                                        LOGI(3,ImageUtils<FloatImageType>::writeImage("localKernelWidths-New.nii",kernelWidthForRemovednJDs));
                                        LOGI(3,ImageUtils<FloatImageType>::writeImage("localKernelWidths-old.nii",localKernelWidths));
                                        
                                        localKernelWidths=FilterUtils<FloatImageType>::add(localKernelWidths,kernelWidthForRemovednJDs);
                                        
                                        negJacMaskPrevious=negJacMask;
#endif                                        
                                        if (minJac2>0.1)
                                            break;
                                    }
                                    //LOGI(3,ImageUtils<FloatImageType>::writeImage("localKernelWidths.nii",localKernelWidths));

                                    LOGV(3)<<VAR(minJac/kernelGamma)<<endl;
                                    LOGV(1)<<"Actual number of kernels: "<<VAR(k)<<endl;
                                    seamEstimator.setPairwiseWeight(m_pairwiseWeight);
                                    seamEstimator.finalize();
#ifdef USELOCALSIGMASFORDILATION
                                    energy=seamEstimator.solveUntilPosJacDet(refineSeamIter,smoothIncrease,useMaskForSSR,localKernelWidths);
#else
                                    energy=seamEstimator.solveUntilPosJacDet(refineSeamIter,smoothIncrease,useMaskForSSR,50);
#endif
                                    result=seamEstimator.getMean();
                                    //labelImage=seamEstimator.getLabelImage();
                                    }//neg jac
                                }//refine seams
                                

                                for (int iter=0;iter<refineIter;++iter){
                                    //estimator.setAlpha(pow(2.0,1.0*iter)*alpha);
                                    addImage(weightingName,metric,estimator,meanEstimator,targetImageIterator->second,sourceImageIterator->second,result,false,estimateMRF,radius,m_gamma);
                                    estimator.finalize();
                                    double newEnergy=estimator.solve();
                                    LOGV(1)<<VAR(iter)<<" "<<VAR(newEnergy)<<" "<<(energy-newEnergy)/energy<<endl;
                                    if (newEnergy >energy )
                                        break;
                                    result=estimator.getMean();
                                    if ( (energy-newEnergy)/energy < 1e-4) {
                                        LOGV(1)<<"refinement converged, stopping."<<endl;
                                        break;
                                    }
                                    energy=newEnergy;
                                    
                                }
                                estimator.setAlpha(alpha);

                                m_energy+=energy;
                            }

                            if (outputDir!=""){
                                ostringstream oss;
                                oss<<outputDir<<"/propagatedDeformation-"<<sourceID<<"-TO-"<<targetID<<".mha";
                                ImageUtils<DeformationFieldType>::writeImage(oss.str(),result);
                                if (estimateMRF && labelImage.IsNotNull()){
                                    ostringstream oss2;
                                    oss2<<outputDir<<"/fusionLabelImage-"<<sourceID<<"-TO-"<<targetID<<".nii";
                                    ImageUtils<ImageType>::writeImage(oss2.str(),labelImage);
                                }

                            }
                        }else{
                            result=deformationSourceTarget;
                        }//if (estimateMean || estimateMRF)

                        
                        double similarity;
                        warpedSourceImage=TransfUtils<ImageType>::warpImage(sourceImageIterator->second,result);
                        switch(metric){
                        case NCC:
                            similarity=Metrics<ImageType,FloatImageType>::nCC(warpedSourceImage,targetImageIterator->second);
                            break;
                        case MSD:
                            similarity=Metrics<ImageType,FloatImageType>::msd(warpedSourceImage,targetImageIterator->second);
                            break;
                        case MAD:
                            similarity=Metrics<ImageType,FloatImageType>::mad(warpedSourceImage,targetImageIterator->second);
                            break;
                        }

                        if (indivCompare && similarity>initialSimilarity){
                            //fall back to initial solution since similarity has actually decreased
                            similarity=initialSimilarity;
                            result=deformationSourceTarget;

                        }
                        m_similarity+=similarity;

                        if (maxHops>1 || !dontCacheDeformations){
                            TMPdeformationCache[sourceID][targetID]=result;
                        }

                        // compare landmarks
                        if (m_landmarkFileList.size()){
                            //hope that all landmark files are available :D
                            m_TRE+=TransfUtils<ImageType>::computeTRE(m_landmarkFileList[targetID], m_landmarkFileList[sourceID],result,targetImageIterator->second);
                        }
                        
                        if (m_groundTruthSegmentations[targetID].IsNotNull() && m_groundTruthSegmentations[sourceID].IsNotNull()){
                            ImagePointerType deformedSeg=TransfUtils<ImageType>::warpImage(  m_groundTruthSegmentations[sourceID] , result ,true);
                            SegmentationMapper<ImageType> segmentationMapper;

                            ImagePointerType groundTruthImg=segmentationMapper.FindMapAndApplyMap((m_groundTruthSegmentations)[targetID]);
                            ImagePointerType segmentedImg=segmentationMapper.ApplyMap(deformedSeg);
                            double dice=0.0;
                            double volumeWeightedDice=0.0;
                            double weightSum=0.0;
                            LOGV(3)<<"IndividDice "<<VAR(sourceID)<<" "<<VAR(targetID);
                            for (int i=1;i<segmentationMapper.getNumberOfLabels();++i){
                                typedef typename itk::LabelOverlapMeasuresImageFilter<ImageType> OverlapMeasureFilterType;
                                typename OverlapMeasureFilterType::Pointer filter = OverlapMeasureFilterType::New();
                                ImagePointerType binaryGT=FilterUtils<ImageType>::select(groundTruthImg,i);
                                filter->SetSourceImage(binaryGT);
                                filter->SetTargetImage(FilterUtils<ImageType>::select(segmentedImg,i));
                                filter->SetCoordinateTolerance(1e-4);
                                filter->Update();
                                dice+=filter->GetDiceCoefficient();
                                LOGI(3,std::cout<<" label: "<<segmentationMapper.GetInverseMappedLabel(i)<<" "<<filter->GetDiceCoefficient());
                                double weight=FilterUtils<ImageType>::sum(binaryGT);
                                volumeWeightedDice+=weight*filter->GetDiceCoefficient();
                                weightSum+=weight;
                            }
                            LOGI(3,std::cout<<endl);
                            dice/=(segmentationMapper.getNumberOfLabels()-1.0);
                            volumeWeightedDice/=weightSum;
                            LOGV(1)<<VAR(sourceID)<<" "<<VAR(targetID)<<" "<<VAR(dice)<<" "<<VAR(volumeWeightedDice)<<endl;
                            m_dice+=dice;
                            m_volumeWeightedDice+=volumeWeightedDice;
                            
                        }
                        
                    
                        //create mask of valid deformation region
                        
                        typename DisplacementFieldJacobianDeterminantFilterType::Pointer jacobianFilter = DisplacementFieldJacobianDeterminantFilterType::New();
                        jacobianFilter->SetInput(result);
                        jacobianFilter->SetUseImageSpacingOff();
                        jacobianFilter->Update();
                        FloatImagePointerType jac=jacobianFilter->GetOutput();
                        double minJac = FilterUtils<FloatImageType>::getMin(jac);
                        LOGV(2)<<VAR(sourceID)<<" "<<VAR(targetID)<< " " << VAR(minJac) <<endl;
                        if (outputDir!=""){
                            ostringstream oss2;
                            oss2<<outputDir<<"/jacobianDetsFinal-"<<sourceID<<"-TO-"<<targetID<<".mha";
                            LOGI(3,ImageUtils<FloatImageType>::writeImage(oss2.str(),jac));
                        }
                        m_averageMinJac+=minJac;
                        if (minJac<m_minMinJacobian){
                            m_minMinJacobian=minJac;
                        }
                        
                    }//if

                   
                }//target images
              
            }//source images
            m_dice/=count;
            m_volumeWeightedDice/=count;
            m_TRE/=count;
            m_energy/=count;
            m_similarity/=count;
            m_averageMinJac/=count;
            if (!dontCacheDeformations || maxHops>1){
                //error computation only available when all deformations are cached
                error=TransfUtils<ImageType>::computeError(&TMPdeformationCache,&trueDeformations,&imageIDs);
            }
            inconsistency = TransfUtils<ImageType>::computeInconsistency(&deformationCache,&imageIDs,&trueDeformations);
            LOG<<VAR(iter)<<" "<<VAR(error)<<" "<<VAR(inconsistency)<<" "<<VAR(m_TRE)<<" "<<VAR(m_dice)<<" "<<VAR(m_volumeWeightedDice)<<" "<<VAR(m_energy)<<" "<<VAR(m_similarity)<<" "<<VAR(m_averageMinJac)<<" "<<VAR(m_minMinJacobian)<<endl;
            if (!runEndless){
                if (m_similarity>m_oldSimilarity){
                    LOG<<"Similarity increased, stopping and returning previous estimate."<<endl;
                    break;
                }else if ( fabs((m_oldSimilarity-m_similarity)/m_oldSimilarity) < 1e-4) { //else if ( (m_oldEnergy-m_energy)/m_oldEnergy < 1e-2) {
                    LOG<<"Optimization converged, stopping."<<endl;
                    deformationCache=TMPdeformationCache;
                    break;
                }
            }
            m_oldEnergy=m_energy;
            m_oldSimilarity=m_similarity;
            deformationCache=TMPdeformationCache;
          

        }//hops
        
        LOG<<"Storing output."<<endl;
        for (ImageListIteratorType targetImageIterator=inputImages.begin();targetImageIterator!=inputImages.end();++targetImageIterator){
            string id= targetImageIterator->first;
        }
         return 1;
    }//run
protected:
    map<string,ImagePointerType> * readImageList(string filename){
        map<string,ImagePointerType> * result=new  map<string,ImagePointerType>;
        ifstream ifs(filename.c_str());
        if (!ifs){
            LOG<<"could not read "<<filename<<endl;
            exit(0);
        }
        while( ! ifs.eof() ) 
            {
                string imageID;
                ifs >> imageID;                
                if (imageID!=""){
                    ImagePointerType img;
                    string imageFileName ;
                    ifs >> imageFileName;
                    LOGV(3)<<"Reading image "<<imageFileName<< " with ID "<<imageID<<endl;
                    img=ImageUtils<ImageType>::readImage(imageFileName);
                    if (result->find(imageID)==result->end())
                        (*result)[imageID]=img;
                    else{
                        LOG<<"duplicate image ID "<<imageID<<", aborting"<<endl;
                        exit(0);
                    }
                }
            }
        return result;
    }        
  
    FloatImagePointerType addImage(string weighting, MetricType metric,RegistrationFuserType & estimator,  GaussianEstimatorVectorImage<ImageType,double> & meanEstimator, ImagePointerType targetImage, ImagePointerType sourceImage, DeformationFieldPointerType def, bool estimateMean, bool estimateMRF, double radius, double m_gamma){
        FloatImagePointerType metricImage;

        if (weighting=="global" || weighting=="local" || weighting=="globallocal"){
            ImagePointerType warpedSourceImage=TransfUtils<ImageType>::warpImage(sourceImage,def);
            if (weighting=="local" || weighting=="globallocal"){
                switch(metric){
                case NCC:
                    metricImage=Metrics<ImageType,FloatImageType>::efficientLNCC(warpedSourceImage,targetImage,radius,m_gamma);
                    break;
                case MSD:
                    metricImage=Metrics<ImageType,FloatImageType>::LSSDNorm(warpedSourceImage,targetImage,radius,m_gamma);
                    break;
                case MAD:
                    metricImage=Metrics<ImageType,FloatImageType>::LSADNorm(warpedSourceImage,targetImage,radius,m_gamma);
                    break;
                default:
                    metricImage=Metrics<ImageType,FloatImageType>::efficientLNCC(warpedSourceImage,targetImage,radius,m_gamma);
                }
            }else{
                metricImage=FilterUtils<ImageType,FloatImageType>::createEmpty(targetImage);
                metricImage->FillBuffer(1.0);
            }
            double globalWeight=1.0;
            if (weighting=="global" || weighting=="globallocal"){
                switch(metric){
                case NCC:
                    globalWeight=Metrics<ImageType,FloatImageType>::nCC(warpedSourceImage,targetImage);
                    break;
                case MSD:
                    globalWeight=Metrics<ImageType,FloatImageType>::nCC(warpedSourceImage,targetImage);
                    break;
                case MAD:
                    globalWeight=Metrics<ImageType,FloatImageType>::nCC(warpedSourceImage,targetImage);
                    break;
                default:
                    globalWeight=Metrics<ImageType,FloatImageType>::nCC(warpedSourceImage,targetImage);
                }
                LOGV(2)<<VAR(globalWeight)<<" "<<VAR(pow(globalWeight,m_gamma))<<endl;
                globalWeight=pow(globalWeight,m_gamma); 
                ImageUtils<FloatImageType>::multiplyImage(metricImage,globalWeight);
            }
            FilterUtils<FloatImageType>::lowerThresholding(metricImage,std::numeric_limits<float>::epsilon());
            if (estimateMRF)
                estimator.addImage(def,metricImage);
            if (estimateMean)
                meanEstimator.addImage(def,metricImage);
        }else{
            if (estimateMRF){
                //estimator.addImage(def);
                LOG<<"you should never come here.."<<endl;
            }
            if (estimateMean)
                meanEstimator.addImage(def);
        }
        return metricImage;
    }
    
    
    FloatImagePointerType replaceFirstImage(string weighting, MetricType metric,RegistrationFuserType & estimator,  GaussianEstimatorVectorImage<ImageType,double> & meanEstimator, ImagePointerType targetImage, ImagePointerType sourceImage, DeformationFieldPointerType def, bool estimateMean, bool estimateMRF, double radius, double m_gamma){
        
        FloatImagePointerType metricImage;
        if (weighting=="global" || weighting=="local" || weighting=="globallocal"){
            ImagePointerType warpedSourceImage=TransfUtils<ImageType>::warpImage(sourceImage,def);

            if (weighting=="local" || weighting=="globallocal"){
                switch(metric){
                case NCC:
                    metricImage=Metrics<ImageType,FloatImageType>::efficientLNCC(warpedSourceImage,targetImage,radius,m_gamma);
                    break;
                case MSD:
                    metricImage=Metrics<ImageType,FloatImageType>::LSSDNorm(warpedSourceImage,targetImage,radius,m_gamma);
                    break;
                case MAD:
                    metricImage=Metrics<ImageType,FloatImageType>::LSADNorm(warpedSourceImage,targetImage,radius,m_gamma);
                    break;
                default:
                    metricImage=Metrics<ImageType,FloatImageType>::efficientLNCC(warpedSourceImage,targetImage,radius,m_gamma);
                }
            }else{
                metricImage=FilterUtils<ImageType,FloatImageType>::createEmpty(targetImage);
                metricImage->FillBuffer(1.0);
            }
            double globalWeight=1.0;
            if (weighting=="global" || weighting=="globallocal"){
                switch(metric){
                case NCC:
                    globalWeight=Metrics<ImageType,FloatImageType>::nCC(warpedSourceImage,targetImage);
                    break;
                case MSD:
                    globalWeight=Metrics<ImageType,FloatImageType>::nCC(warpedSourceImage,targetImage);
                    break;
                case MAD:
                    globalWeight=Metrics<ImageType,FloatImageType>::nCC(warpedSourceImage,targetImage);
                    break;
                default:
                    globalWeight=Metrics<ImageType,FloatImageType>::nCC(warpedSourceImage,targetImage);
                }
                LOGV(2)<<VAR(globalWeight)<<" "<<VAR(pow(globalWeight,m_gamma))<<endl;
                globalWeight=pow(globalWeight,m_gamma); 
                ImageUtils<FloatImageType>::multiplyImage(metricImage,globalWeight);
            }
            FilterUtils<FloatImageType>::lowerThresholding(metricImage,std::numeric_limits<float>::epsilon());
            if (estimateMRF)
                estimator.replaceFirstImage(def,metricImage);
            if (estimateMean)
                meanEstimator.addImage(def,metricImage);
        }else{
            if (estimateMRF)
                estimator.addImage(def);
            if (estimateMean)
                meanEstimator.addImage(def);
        }
        return metricImage;
    }
   
};//class

}//namespace
