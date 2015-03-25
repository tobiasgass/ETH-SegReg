/**
 * @file   Registration-Fusion-MRF.h
 * @author Tobias Gass <tobiasgass@gmail.com>
 * @date   Thu Mar 12 11:39:15 2015
 * 
 * @brief  This file implements a class to fuse registrations using MRFs
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
#include <string>
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
#include "itkHistogramMatchingImageFilter.h"

namespace MRegFuse{

  /**
   * @brief  Registration fusion of multiple registration hypotheses between two images,  using MRF and optionally LWA fusion
   */
  template <class ImageType>
    class RegistrationFusionMRF{
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
      string inputDeformationFilenames,targetFileName,sourceFileName,trueDefFilename="",targetSegmentationFilename="",sourceSegmentationFilename="",targetLandmarkFilename="",sourceLandmarkFilename="";
      int verbose=0;
      double pWeight=1.0;
      int radius=3;
      double controlGridSpacingFactor=4;
      int maxHops=1;
      bool uniformUpdate=true;
      string metricName="NCC";
      string weightingName="local";
      bool lateFusion=false;
      bool dontCacheDeformations=false;
      bool graphCut=false;
      double smoothness=1.0;
      double alpha=0;
      m_gamma=10;
      double m_pairwiseWeight=1.0;
      bool useHardConstraints=false;
      bool m_refineSolution=false;
      bool estimateMRF=false,estimateMean=false;
      int refineIter=0;
      int refineSeamIter=0;
      string outputFilename;
      bool histNorm=false;
      string outputDir="./";
      bool anisoSmoothing=false;
      int nKernels=20;
      double smoothIncrease=1.2;
      bool useMaskForSSR=false;
      //as->parameter ("A",atlasSegmentationFileList , "list of atlas segmentations <id> <file>", true);
      as->option ("MRF", estimateMRF, "use MRF fusion");
      as->option ("mean", estimateMean, "use (local) mean fusion. Can be used in addition to MRF or stand-alone.");
      as->parameter ("T", inputDeformationFilenames, "list of deformations (comma separated)", true);
      as->parameter ("t", targetFileName, " target image filename", true);
      as->parameter ("s", sourceFileName, " source image filename", true);
      as->parameter ("true", trueDefFilename, "  TRUE deformation filename", false);
      as->parameter ("o", outputFilename , "output filename prefix ", false);

      as->parameter ("st",targetSegmentationFilename , "target segmentation filename", false);
      as->parameter ("ss",sourceSegmentationFilename , "source segmentation filename", false);
      as->parameter ("lt",targetLandmarkFilename , " target landmark filename", false);       
      as->parameter ("ls",sourceLandmarkFilename , " source landmark filename", false);       

      //as->parameter ("W", weightListFilename,"list of weights for deformations",false);
      as->parameter ("metric", metricName,"metric to be used for global or local weighting, valid: NONE,SAD,MSD,NCC,MI,NMI",false);
      as->parameter ("weighting", weightingName,"internal weighting scheme {uniform,local,global}. non-uniform will only work with metric != NONE",false);
      as->parameter ("g", m_gamma,"gamma for exp(- metric/gamma)[SSD,SAD,..] or metric^gamma [NCC]",false);
      as->parameter ("w", m_pairwiseWeight,"pairwise weight for MRF",false);
      as->parameter ("radius", radius,"patch radius for local metrics",false);
      as->parameter ("controlGridSpacing", controlGridSpacingFactor,"Resolution of the MRF control grid, obtained by dividing the input image resolution by this factor.",false);
      as->option ("hardConstraints", useHardConstraints,"Use hard constraints in the MRF to prevent folding.");
      as->parameter ("refineIter", refineIter,"refine MRF solution by adding the result as new labels to the MRF and re-solving until convergence.",false);
      as->parameter ("refineSeamIter", refineSeamIter,"refine MRF solution at seams by smoothing the result and fusing it with the original solution.",false);
      as->parameter ("smoothIncrease", smoothIncrease,"factor to increase smoothing with per iteration for SSR.",false);
      as->option ("useMask", useMaskForSSR,"only update pixels with negative jac dets (or in the vincinity of those) when using SSR.");
      as->parameter ("O", outputDir,"outputdirectory (will be created + no overwrite checks!)",false);
      as->parameter ("maxHops", maxHops,"maximum number of hops",false);
      as->parameter ("alpha", alpha,"pairwise balancing weight (deformation (alpha=0) vs label smoothness (alpha=1))",false);
      as->option ("dontCacheDeformations", dontCacheDeformations,"read deformations only when needed to save memory. higher IO load!");
      as->option ("histNorm", histNorm,"Match source histogram to target histogram!");
      as->option ("anisoSmooth", anisoSmoothing,"Anisotropically penalize unsmooth deformations, using statistics from the input");
      
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

      ImagePointerType targetImage=ImageUtils<ImageType>::readImage(targetFileName);
      ImagePointerType sourceImage=ImageUtils<ImageType>::readImage(sourceFileName);

      if (histNorm){
	typedef itk::HistogramMatchingImageFilter<ImageType,ImageType> HEFilterType;
	typename HEFilterType::Pointer IntensityEqualizeFilter = HEFilterType::New();
	IntensityEqualizeFilter->SetReferenceImage(targetImage  );
	IntensityEqualizeFilter->SetInput( sourceImage );
	IntensityEqualizeFilter->SetNumberOfHistogramLevels( 100);
	IntensityEqualizeFilter->SetNumberOfMatchPoints( 15);
	IntensityEqualizeFilter->ThresholdAtMeanIntensityOn();
	IntensityEqualizeFilter->Update();
	sourceImage=IntensityEqualizeFilter->GetOutput();
      }
      //split comma separated list
      std::stringstream   myStream( inputDeformationFilenames );
      char          temp[1000];
      const char * cdelim=",";
      char delim=*cdelim;
      std::vector<std::string> inputDeformationFilenameList;
      while( myStream.getline(temp, 1000,delim) ){ // getline( myStream , temp, delim ) ) {
	inputDeformationFilenameList.push_back( temp );
	LOGV(1)<<temp<<endl;
      }

      int nDeformations=inputDeformationFilenameList.size();
      RegistrationFuserType estimator;
      estimator.setAlpha(alpha);
      estimator.setPairwiseWeight(m_pairwiseWeight);
      estimator.setGridSpacing(controlGridSpacingFactor);
      estimator.setHardConstraints(useHardConstraints);
      estimator.setAnisoSmoothing(anisoSmoothing);
      GaussianEstimatorVectorImage<ImageType,double> meanEstimator;
      DeformationFieldPointerType result;
      for (int i=0;i<nDeformations;++i){
	DeformationFieldPointerType def=ImageUtils<DeformationFieldType>::readImage(inputDeformationFilenameList[i]);
	FloatImagePointerType weightImage=addImage(weightingName,metric,estimator,meanEstimator,targetImage,sourceImage,def,estimateMean,estimateMRF,radius,m_gamma);
	if (weightImage.IsNotNull() && outputDir!=""){
	  ostringstream oss;
	  string fName=inputDeformationFilenameList[i];
	  replace(fName.begin(), fName.end(), '/', '_' );
	  LOG<<VAR(fName)<<endl;
	  oss<<outputDir<<"/"<<outputFilename<<"-weights-"<<fName;
	  LOGI(2,ImageUtils<FloatImageType>::writeImage(oss.str(),weightImage));
	}
      }
      double m_energy=0.0;
      double relativeClosenessToLB;
      ImagePointerType labelImage;
      if (estimateMean)
	meanEstimator.finalize();
        
      if (! estimateMRF){
	DeformationFieldPointerType meanDef=meanEstimator.getMean();
	result=meanDef;
      }else{
            
	if (estimateMean){
	  DeformationFieldPointerType meanDef=meanEstimator.getMean();
	  addImage(weightingName,metric,estimator,meanEstimator,targetImage,sourceImage,meanDef,estimateMean,estimateMRF,radius,m_gamma);
                
	}
            
	estimator.finalize();
	double energy=estimator.solve();
	labelImage=estimator.getLabelImage();
	result=estimator.getMean();
	relativeClosenessToLB=estimator.getRelativeLB();

	LOGV(1)<<VAR(energy)<<endl;
	if (refineSeamIter>0){

	  typename DisplacementFieldJacobianDeterminantFilterType::Pointer jacobianFilter = DisplacementFieldJacobianDeterminantFilterType::New();
	  jacobianFilter->SetInput(result);
	  jacobianFilter->SetUseImageSpacingOff();
	  jacobianFilter->Update();
	  FloatImagePointerType jac=jacobianFilter->GetOutput();
	  double minJac = FilterUtils<FloatImageType>::getMin(jac);
	  if (minJac<0.0){
	    if (outputDir!=""){
	      ostringstream oss2;
	      oss2<<outputDir<<"/"<<outputFilename<<"-jacobianDetWithNegVals.mha";
	      LOGI(3,ImageUtils<FloatImageType>::writeImage(oss2.str(),jac));
	    }
                
	    LOG<<"Refining seams by smoothing solution with "<<nKernels<<" kernels.."<<endl;
	    DeformationFieldPointerType originalFusionResult=result;
	    RegistrationFuserType seamEstimator;
	    double kernelBaseWidth=0.5;//1.0;//pow(-1.0*minJac,1.0/D);

	    seamEstimator.setAlpha(alpha);
	    seamEstimator.setGridSpacing(controlGridSpacingFactor);
	    seamEstimator.setHardConstraints(useHardConstraints);
	    seamEstimator.setAnisoSmoothing(false);
	    //seamEstimator.setAlpha(pow(2.0,1.0*iter)*alpha);
 
	    //hacky shit to avoid oversmoothing
	    FloatImagePointerType weights=addImage(weightingName,metric,estimator,meanEstimator,targetImage,sourceImage,originalFusionResult,false,estimateMRF,radius,m_gamma);
	    seamEstimator.addImage(estimator.getLowResResult(),weights);
	    //kernelSigmas= kernelBaseWidth/4,kbw/2,kbw,2*kbw,4*kbw
	    int k=0;
	    double kernelSigma;
	    double previousSigma=0.0;
	    kernelBaseWidth=0.5;
	    double exp=2;
	    DeformationFieldPointerType smoothedResult=result;
#ifdef USELOCALSIGMASFORDILATION
	    ImagePointerType negJacMaskPrevious=FilterUtils<FloatImageType,ImageType>::binaryThresholdingHigh(jac,0.0);
	    FloatImagePointerType localKernelWidths=ImageUtils<FloatImageType>::duplicate(jac);
	    localKernelWidths->FillBuffer(0.0);
#endif
	    for (;k<nKernels;++k){
                    
	      kernelSigma=kernelBaseWidth*pow(exp,1.0*(k));
	      double actualSigma=sqrt(pow(kernelSigma,2.0)-pow(previousSigma,2.0));
	      LOGV(3)<<VAR(kernelSigma)<<" "<<VAR(actualSigma)<<endl;
	      smoothedResult=TransfUtils<ImageType>::gaussian(smoothedResult,actualSigma);
	      previousSigma=actualSigma;
	      addImage(weightingName,metric,seamEstimator,meanEstimator,targetImage,sourceImage,smoothedResult,false,estimateMRF,radius,m_gamma);
	      typename DisplacementFieldJacobianDeterminantFilterType::Pointer jacobianFilter = DisplacementFieldJacobianDeterminantFilterType::New();
	      jacobianFilter->SetInput(smoothedResult);
	      jacobianFilter->SetUseImageSpacingOff();
	      jacobianFilter->Update();
	      FloatImagePointerType jac=jacobianFilter->GetOutput();
	      if (outputDir!=""){
		ostringstream oss2;
		oss2<<outputDir<<"/"<<outputFilename<<"-jacobianDetWithNegVals-sigma"<<actualSigma<<".mha";
		LOGV(3)<<"Saving jac to " << oss2.str()<<endl;
		LOGI(3,ImageUtils<FloatImageType>::writeImage(oss2.str(),jac));
	      }
	      double minJac2 = FilterUtils<FloatImageType>::getMin(jac);
	      LOGV(3)<<VAR(minJac2)<<endl;
#ifdef USELOCALSIGMASFORDILATION
	      //get negative jacobian value locations
	      ImagePointerType negJacMask=FilterUtils<FloatImageType,ImageType>::binaryThresholdingHigh(jac,0.0);
	      //subtract and invert to get locations of removed negative JDs
	      //ImagePointerType removedNegJacMask=(FilterUtils<ImageType>::substract(negJacMaskPrevious,negJacMask));
	      ImagePointerType removedNegJacMask=FilterUtils<ImageType>::binaryThresholding(FilterUtils<ImageType>::add(negJacMaskPrevious,negJacMask),1,1);
	      //ImagePointerType removedNegJacMask=FilterUtils<ImageType>::binaryThresholdingHigh(FilterUtils<ImageType>::add(negJacMaskPrevious,negJacMask),1);
	      //create image with sigma at locations where nJDs were removed
	      FloatImagePointerType kernelWidthForRemovednJDs=ImageUtils<FloatImageType>::createEmpty(jac);
	      kernelWidthForRemovednJDs->FillBuffer(3.0*kernelSigma);
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
	    LOG<<"Actual number of kernels: "<<VAR(k)<<endl;
	    LOGV(3)<<VAR(minJac)<<" "<<VAR(minJac/kernelSigma)<<" "<<VAR(kernelSigma)<<endl;
#ifdef USELOCALSIGMASFORDILATION
	    LOGI(3,ImageUtils<FloatImageType>::writeImage("localKernelWidths.nii",localKernelWidths));
#endif
	    seamEstimator.setPairwiseWeight(m_pairwiseWeight);
	    seamEstimator.finalize();
#ifdef USELOCALSIGMASFORDILATION
	    energy=seamEstimator.solveUntilPosJacDet(refineSeamIter,smoothIncrease,useMaskForSSR,localKernelWidths);
#else
	    energy=seamEstimator.solveUntilPosJacDet(refineSeamIter,smoothIncrease,useMaskForSSR,50);
#endif
	    result=seamEstimator.getMean();
	  }//actual neg jac
	}//refine seams
	for (int iter=0;iter<refineIter;++iter){
	  //estimator.setAlpha(pow(2.0,1.0*iter)*alpha);
	  addImage(weightingName,metric,estimator,meanEstimator,targetImage,sourceImage,result,false,estimateMRF,radius,m_gamma);
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
	oss<<outputDir<<"/"<<outputFilename<<"-avgDeformation.mha";
	if (result.IsNotNull())
	  ImageUtils<DeformationFieldType>::writeImage(oss.str(),result);
	if (estimateMRF && labelImage.IsNotNull()){
	  ostringstream oss2;
	  oss2<<outputDir<<"/"<<outputFilename<<"-labelImage.nii";
	  LOGI(1,ImageUtils<ImageType>::writeImage(oss2.str(),labelImage));
	}
            
      }
      double m_dice=0.0;
      double m_TRE=0.0;
      double m_similarity=0.0;
      double m_averageMinJac=0;
      double m_minMinJacobian=100000000;
      if (targetLandmarkFilename!="" && sourceLandmarkFilename!="" ){
	//hope that all landmark files are available :D
	m_TRE+=TransfUtils<ImageType>::computeTRE(targetLandmarkFilename, sourceLandmarkFilename,result,targetImage);
      }
        
      if (targetSegmentationFilename!="" && sourceSegmentationFilename!=""){
	ImagePointerType targetSegmentation=ImageUtils<ImageType>::readImage(targetSegmentationFilename);
	ImagePointerType sourceSegmentation=ImageUtils<ImageType>::readImage(sourceSegmentationFilename);
	ImagePointerType deformedSeg=TransfUtils<ImageType>::warpImage(  sourceSegmentation , result ,true);
	SegmentationMapper<ImageType> segmentationMapper;
            
	ImagePointerType groundTruthImg=segmentationMapper.FindMapAndApplyMap(targetSegmentation);
	ImagePointerType segmentedImg=segmentationMapper.ApplyMap(deformedSeg);
	double dice=0.0;
	for (int i=1;i<segmentationMapper.getNumberOfLabels();++i){
	  typedef typename itk::LabelOverlapMeasuresImageFilter<ImageType> OverlapMeasureFilterType;
	  typename OverlapMeasureFilterType::Pointer filter = OverlapMeasureFilterType::New();
	  filter->SetSourceImage(FilterUtils<ImageType>::select(groundTruthImg,i));
	  filter->SetTargetImage(FilterUtils<ImageType>::select(segmentedImg,i));
	  filter->SetCoordinateTolerance(1e-4);
	  filter->Update();
	  dice+=filter->GetDiceCoefficient();
	}
	dice/=(segmentationMapper.getNumberOfLabels()-1.0);
	//LOGV(1)<<VAR(sourceID)<<" "<<VAR(targetID)<<" "<<VAR(dice)<<endl;
	m_dice+=dice;
      }
      double similarity;
      ImagePointerType warpedSourceImage=TransfUtils<ImageType>::warpImage(sourceImage,result);
      switch(metric){
      case NCC:
	similarity=Metrics<ImageType,FloatImageType>::nCC(warpedSourceImage,targetImage);
	break;
      case MSD:
	similarity=Metrics<ImageType,FloatImageType>::msd(warpedSourceImage,targetImage);
	break;
      case MAD:
	similarity=Metrics<ImageType,FloatImageType>::mad(warpedSourceImage,targetImage);
	break;
      }
      typename DisplacementFieldJacobianDeterminantFilterType::Pointer jacobianFilter = DisplacementFieldJacobianDeterminantFilterType::New();
      jacobianFilter->SetInput(result);
      jacobianFilter->SetUseImageSpacingOff();
      jacobianFilter->Update();
      FloatImagePointerType jac=jacobianFilter->GetOutput();
      if (outputDir!="" && jac.IsNotNull()){
	ostringstream filename;
	filename<<outputDir<<"/"<<outputFilename<<"-jacobian.mha";
	LOGI(1,ImageUtils<FloatImageType>::writeImage(filename.str(),jac));
      }
      double minJac = FilterUtils<FloatImageType>::getMin(jac);
      double maxJac = FilterUtils<FloatImageType>::getMax(jac);
      //LOGV(2)<<VAR(sourceID)<<" "<<VAR(targetID)<< " " << VAR(minJac) <<" " <<VAR(nCC)<<endl;
      m_averageMinJac+=minJac;
      if (minJac<m_minMinJacobian){
	m_minMinJacobian=minJac;
      }
      double error=0;
        
      if (trueDefFilename!=""){
	DeformationFieldPointerType trueDef=ImageUtils<DeformationFieldType>::readImage(trueDefFilename);


	ImagePointerType mask=TransfUtils<ImageType>::createEmptyImage(trueDef);
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
	DeformationFieldPointerType diff=TransfUtils<ImageType>::subtract(result,trueDef);
	//error=TransfUtils<ImageType>::computeDeformationNormMask(diff,mask);
	error=TransfUtils<ImageType>::computeDeformationNorm(diff);
      }
      LOG <<VAR(error)<<" "<<VAR(m_TRE)<<" "<<VAR(m_dice)<<" "<<VAR(m_energy)<<" "<<VAR(relativeClosenessToLB)<<" "<<VAR(similarity)<<" "<<VAR(minJac)<<" "<<VAR(maxJac)<<endl;

      
    }//run
  protected:
   
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
	  LOG<<"should not be called like that!"<<endl;
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
