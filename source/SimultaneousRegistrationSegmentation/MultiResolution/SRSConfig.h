/**
 * @file   SRSConfig.h
 * @author Tobias Gass <tobiasgass@gmail.com>
 * @date   Fri Mar  6 10:32:41 2015
 * 
 * @brief  This class holds the configuration parameters for SRS and allows for reading them from the command line. Config file support is deprecated
 * 
 * 
 */

#ifndef SRSCONFIG_H_
#define SRSCONFIG_H_
#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <iterator>
#include "Log.h"

#include "ArgumentParser.h"

namespace SRS{
  
  class SRSConfig{
  public:
    std::string targetFilename,atlasFilename,targetGradientFilename, outputDeformedSegmentationFilename,atlasSegmentationFilename, outputDeformedFilename,deformableFilename,defFilename, segmentationOutputFilename, atlasGradientFilename,atlasMaskFilename;
    std::string segmentationProbsFilename, pairWiseProbsFilename, targetAnatomyPriorFilename,affineBulkTransform,bulkTransformationField,ROIFilename,groundTruthSegmentationFilename;
    std::string targetRGBImageFilename,atlasRGBImageFilename;
    std::string logFileName,segmentationUnaryProbFilename;
    int auxiliaryLabel;/// Label to tell SRS that this is not a target anatomy label.
    double pairwiseRegistrationWeight;
    double pairwiseSegmentationWeight;
    int displacementSampling;
    double unaryWeight;
    int maxDisplacement;
    double unaryRegistrationWeight;
    double unarySegmentationWeight;
    double pairwiseCoherenceWeight;
    int nSegmentations;
    int verbose;
    int * levels;
    int nLevels;
    int startTiling;
    int iterationsPerLevel;
    bool train;
    double displacementRescalingFactor;
    double scale,asymmetry;
    double segmentationScalingFactor;
    int optIter;
    double downScale;
    double pairwiseContrastWeight;
    int nSubsamples;
    double alpha;
    int imageLevels;
    bool computeMultilabelAtlasSegmentation;
    bool useTargetAnatomyPrior;
    bool segment,regist,coherence;
    double thresh_UnaryReg,thresh_PairwiseReg;
    bool log_UnaryReg,log_PairwiseReg;
    double displacementScaling;
    bool evalContinuously;
    bool TRW,GCO,OPENGM;
    bool fullRegPairwise;
    double coherenceMultiplier;
    bool dontNormalizeRegUnaries;
    double ARSTolerance;
    bool centerImages;
    double toleranceBase;
    bool penalizeOutside;
    bool initWithMoments;
    bool normalizePotentials;
    bool cachePotentials;
    double segDistThresh;
    double theta;
    bool linearDeformationInterpolation;
    bool histNorm;
    bool normalizeImages;
    bool useLowResBSpline;
    std::string atlasLandmarkFilename,targetLandmarkFilename;
    std::vector<int> nRegSamples;
    std::vector<double> resamplingFactors;
    int nSegmentationLevels;
    std::string solver;
  private:
    ArgumentParser * as;
  public:
    SRSConfig(){
      auxiliaryLabel=1;
      atlasLandmarkFilename="";
      targetLandmarkFilename="";
      useLowResBSpline=false;
      normalizeImages=false;
      defFilename="";
      outputDeformedSegmentationFilename="deformedAtlasSegmentation.nii";
      outputDeformedFilename="deformedAtlas.nii";
      segmentationOutputFilename="targetSegmentation.nii";
      groundTruthSegmentationFilename="";
      atlasMaskFilename="";
      pairwiseRegistrationWeight=0;
      pairwiseSegmentationWeight=0;
      displacementSampling=-1;
      unaryWeight=1;
      maxDisplacement=4;
      unaryRegistrationWeight=1;
      unarySegmentationWeight=0;
      pairwiseCoherenceWeight=0;
      nSegmentations=2;
      verbose=0;
      nLevels=4;
      startTiling=3;
      iterationsPerLevel=4;
      train=false;
      segmentationProbsFilename="segmentation.bin";
      pairWiseProbsFilename="pairwise.bin";
      displacementRescalingFactor=0.618;
      scale=0.5;
      asymmetry=0;
      downScale=1;
      pairwiseContrastWeight=1;
      nSubsamples=-1;
      alpha=0;
      imageLevels=-1;
      affineBulkTransform="";
      bulkTransformationField="";
      computeMultilabelAtlasSegmentation=false;
      useTargetAnatomyPrior=false;
      segment=false;
      regist=false;
      coherence=false;
      optIter=10;
      thresh_UnaryReg=std::numeric_limits<double>::max();
      thresh_PairwiseReg=std::numeric_limits<double>::max();;
      log_UnaryReg=false;
      log_PairwiseReg=false;
      displacementScaling=1.0;
      evalContinuously=false;
      logFileName="";
      TRW=false;
      GCO=false;
      fullRegPairwise=false;
      coherenceMultiplier=1.0;
      dontNormalizeRegUnaries=false;
      ARSTolerance=-1.0;
      centerImages=false;
      segmentationScalingFactor=1.0;
      toleranceBase=2.0;
      penalizeOutside=false;
      initWithMoments=false;
      normalizePotentials=false;
      cachePotentials=false;
      segDistThresh=-1.0;
      targetRGBImageFilename="";
      atlasRGBImageFilename="";
      segmentationUnaryProbFilename="";
      theta=0;
      linearDeformationInterpolation=false;
      histNorm=false;
      nSegmentationLevels=1;
      solver="GCO";
    }
    ~SRSConfig(){
      delete as;
      //delete levels;
    }
    void parseParams(int argc, char** argv){
      as=new ArgumentParser(argc, argv);
      parse();
    }
    void copyFrom(SRSConfig c){
      targetFilename=c.targetFilename;
      atlasFilename=c.atlasFilename;
      targetGradientFilename=c.targetGradientFilename;
      atlasGradientFilename=c.atlasGradientFilename;
      outputDeformedSegmentationFilename=c.outputDeformedSegmentationFilename;
      atlasSegmentationFilename=c.atlasSegmentationFilename;
      outputDeformedFilename=c.outputDeformedFilename;
      deformableFilename=c.deformableFilename;
      defFilename=c.defFilename;
      segmentationOutputFilename=c.segmentationOutputFilename;
      segmentationProbsFilename=c.segmentationProbsFilename;
      pairWiseProbsFilename=c.pairWiseProbsFilename;

      pairwiseRegistrationWeight=c.pairwiseRegistrationWeight;
      pairwiseSegmentationWeight=c.pairwiseSegmentationWeight;
      unaryRegistrationWeight=c.unaryRegistrationWeight;
      unarySegmentationWeight=c.unarySegmentationWeight;
      pairwiseCoherenceWeight=c.pairwiseCoherenceWeight;

      displacementSampling=c.displacementSampling;
      unaryWeight=c.unaryWeight;
      maxDisplacement=c.maxDisplacement;
      nSegmentations=c.nSegmentations;
      verbose=c.verbose;
      levels=c.levels;
      nLevels=c.nLevels;
      startTiling=c.startTiling;
      train=c.train;
      displacementRescalingFactor=c.displacementRescalingFactor;
      scale=c.scale;
      asymmetry=c.asymmetry;
      optIter=c.optIter;
      downScale=c.downScale;
      pairwiseContrastWeight=c.pairwiseContrastWeight;
      nSubsamples=c.nSubsamples;
      alpha=c.alpha;                          
      imageLevels=c.imageLevels;
      useTargetAnatomyPrior=c.useTargetAnatomyPrior;
      thresh_UnaryReg=c.thresh_UnaryReg;
      thresh_PairwiseReg=c.thresh_PairwiseReg;
    }
    void parseFile(std::string filename){
      std::ostringstream streamm;
      std::ifstream is(filename.c_str());
      if (!is){
	LOG<<"could not open "<<filename<<" for reading"<<std::endl;
	exit(10);
      }
      std::string buff;
      is >> buff;
      streamm<<buff;
      while (!is.eof()){
	streamm<<" ";
	is >>buff;
	streamm<<buff;
	//			LOG<<streamm.str()<<std::endl;
	//			LOG<<buff<<std::endl;

      }
      //as=ArgumentParser(streamm.str().c_str());
      //parse();
    }
    void parse(){
      bool optionalParameter=true;
      std::string filename="";
      as->parameter ("configFile", filename, "read config from file, additional command line parameters overwrite config file settings. (filename)", false);
      if (filename!=""){
	SRSConfig fromFile;
	fromFile.parseFile(filename);
	copyFrom(fromFile);
      }
      std::string regSampleString="";
      //input filenames
      //mandatory
      as->parameter ("t", targetFilename, "target image (file name)", true);
      as->parameter ("roi", ROIFilename, " image to set target ROI from (file name)", false);
      as->parameter ("a", atlasFilename, "atlas image (file name)", false);
      as->parameter ("sa", atlasSegmentationFilename, "atlas segmentation image (file name)", false);

      as->parameter ("rgbt", targetRGBImageFilename, "RGB version of targetImage for segmentation unaries", false,optionalParameter);
      as->parameter ("rgba", atlasRGBImageFilename, "RGB version of atlas image for segmentation unaries ", false,optionalParameter);

      //optional
      as->parameter ("ma", atlasMaskFilename, "atlas mask image for masking the registration unary potential (file name)", false);

      as->parameter ("lma", atlasLandmarkFilename, "atlas landmark file name", false);
      as->parameter ("lmt", targetLandmarkFilename, "target landmark file name", false);

      as->parameter ("gt", targetGradientFilename, "target gradient image (file name)", false,optionalParameter);
      as->parameter ("ga", atlasGradientFilename, "atlas gradient image (file name)", false,optionalParameter);
      as->parameter ("targetAnatomyPriorFilename", targetAnatomyPriorFilename, "targetAnatomy prior image (file name)", false,optionalParameter);
      as->parameter ("affineBulkTransform", affineBulkTransform, "affine bulk transfomr", false);
      as->parameter ("bulkTransformationFiled", bulkTransformationField, "bulk transformation field", false);
      as->option ("moments", initWithMoments, "initialize deformation with moments ");
      as->option ("center", centerImages, "initialize deformation with centeringTransfom ");

      as->parameter ("ta", outputDeformedFilename, "output image (file name)", false);
      as->parameter ("tsa", outputDeformedSegmentationFilename, "output image (file name)", false);
      as->parameter ("st", segmentationOutputFilename, "output segmentation image (file name)", false);
      as->parameter ("T", defFilename,"deformation field filename", false);
      //weights
      as->parameter ("rp", pairwiseRegistrationWeight,"weight for pairwise registration potentials", false);
      as->parameter ("sp", pairwiseSegmentationWeight,"weight for pairwise segmentation potentials", false);
      as->parameter ("cp", pairwiseCoherenceWeight,"weight for coherence potential", false);
      as->parameter ("cm", coherenceMultiplier,"multiplier for increasing the coherence weight after one ars or srs iteration", false);


      as->parameter ("ru", unaryRegistrationWeight,"weight for registration unary", false);
      as->option ("dontNormRegUn", dontNormalizeRegUnaries,"deactivate normalization of registration unary potentials across multiresolution hirarchies",optionalParameter);
      as->parameter ("su", unarySegmentationWeight,"weight for segmentation unary", false);
      as->option ("fullRegistrationSmoothing",fullRegPairwise ,"Regularise composed deformation instead of regularizing just the current update. Leads to non-submodular function, and is thus not usable with GC optimization.");
      as->option ("penalizeOutside",penalizeOutside ,"Penalize registrations falling outside of moving image.",optionalParameter);
      as->option ("normalizePotentials",normalizePotentials ,"divide all potentials by the total number of the respective potential. This balances forces in the two-layer SRS graph (somewhat).",optionalParameter);
      as->option ("cachePotentials"  ,cachePotentials,"Cache all potential function values before calling the optimizer. requires more memory, but will speed up things!.",optionalParameter);
      as->option ("normalizeImages",normalizeImages ,"Normalize images to zero mean and unit variance. NO CHECK IF PIXELTYPE IS INTEGER!",optionalParameter);
      as->option ("useLowResBSpline",useLowResBSpline ,"Only upsample deformation field to the resolution used in registration unary computation. Speeds up the process a bit, looses some accuracy. DOES NOT WORK/HAVE ANY EFFECT WHEN SRS IS USED!",optionalParameter);

      as->parameter ("auxLabel", auxiliaryLabel,"label of auxiliary anatomy in atlas segmentation. less adherence is enforced in SRS to this label in comparison to other foreground labels. When unset (default), is assumed to be 1 in case their are more than 2 labels in total.", false);


      //thresholds
      as->parameter ("tru", thresh_UnaryReg,"threshold for unary registration potential.", false,optionalParameter);
      as->parameter ("trp", thresh_PairwiseReg,"threshold for pairwise registration potential (factor of max distance).", false,optionalParameter);
      as->parameter ("lru", log_UnaryReg,"negative log metric unary registration potential.", false,optionalParameter);
      as->parameter ("lrp", log_PairwiseReg,"negative log metric for pairwise registration potential.", false,optionalParameter);
      as->parameter ("tsc", segDistThresh,"if the distance to target segmentation label is greater than this threshold, the node is excluded from the segmentation graph. Only works in SRS, not in seg only currently.", false,optionalParameter);

      //other params
      as->parameter ("max", maxDisplacement,"number of displacement samples per axis", false);
      as->parameter ("samp",regSampleString,"displacement sampling hierarchy, eg 6x4x2 for 3 levels", false);
      as->parameter ("nLevels", nLevels,"number of grid multiresolution pyramid levels", false);
      imageLevels=nLevels;
      as->parameter ("nImageLevels", imageLevels,"number of image multiresolution  levels", false);
      as->parameter ("nSegmentationLevels", nSegmentationLevels,"number of segmentation multiresolution  levels", false);
      as->parameter ("startlevel", startTiling,"start tiling", false);
      as->parameter ("iterationsPerLevel", iterationsPerLevel,"iterationsPerLevel", false);
      as->parameter ("optIter", optIter,"max iterations of optimizer", false);
      as->parameter ("r",displacementRescalingFactor,"displacementRescalingFactor", false);
      as->parameter ("asymmetry",asymmetry,"asymmetry in segreg potential", false,optionalParameter);
      as->parameter ("displacementScaling",displacementScaling,"Scaling of displacement labels relative to image spacing. WARNING: if set larger than 1, diffeomorphic registrations are no longer guaranteed!", false,optionalParameter);
      as->parameter ("toleranceBase",toleranceBase,"Base for computing the coherence tolerance at different l evels of the grid pyramid.", false,optionalParameter);

      as->parameter ("segmentationProbs", segmentationProbsFilename,"segmentation probabilities  filename (legacy?)", false,optionalParameter);

      as->parameter ("segmentationUnaryProbs", segmentationUnaryProbFilename,"segmentation unaries probabilities  filename", false,optionalParameter);

        
      as->parameter ("pairwiseProbs", pairWiseProbsFilename,"pairwise segmentation probabilities filename", false,optionalParameter);
      as->option ("train", train,"train classifier (and save), if not set data will be read from the given files");
      as->option ("useTargetAnatomyPrior", useTargetAnatomyPrior,"compute and use a targetAnatomy prior. Currently only works with bone CT images.",optionalParameter);

      std::vector<int> tmp_levels(6,-1);
      as->parameter ("l0", tmp_levels[0],"divisor for level 0", false,optionalParameter);
      as->parameter ("l1", tmp_levels[1],"divisor for level 1", false,optionalParameter);
      as->parameter ("l2", tmp_levels[2],"divisor for level 2", false,optionalParameter);
      as->parameter ("l3", tmp_levels[3],"divisor for level 3", false,optionalParameter);
      as->parameter ("l4", tmp_levels[4],"divisor for level 4", false,optionalParameter);
      as->parameter ("l5", tmp_levels[5],"divisor for level 5", false,optionalParameter);
      as->parameter ("scale", scale,"downscaling factor for images in the registration multiresolution pyramid ", false);
      as->parameter ("segmentationScalingFactor", segmentationScalingFactor,"downsampling factor for segmentation multiresolution pyramid", false);
      as->parameter ("downScale", downScale,"downSample ALL  images by an isotropic factor",false);
      as->parameter ("nSegmentations",nSegmentations ,"number of segmentation labels (>=2)", false);
      as->option ("computeMultilabelAtlasSegmentation",computeMultilabelAtlasSegmentation ,"compute multilabel atlas segmentation from original atlas segmentation. will overwrite nSegmentations.",optionalParameter);

      as->option ("evalContinuously",evalContinuously ,"evaluate optimization at each step. slower, but also returns actual energy and changes in labellings during each iteration.,optionalParamete");
      //as->option ("GCO",GCO ,"Use (alpha expansion) graph cuts instead of TRW-S for optimization.");
      as->parameter ("solver",solver ,"choose solver for optimization (TRWS,GCO,OPENGM).",false);
      as->option ("linearDeformationInterpolation",linearDeformationInterpolation ,"Use linear interpolation for deformation field upsampling.");
      as->option ("histNorm",histNorm ,"Use histogram normalization to adapt the atlas intensity distribution to the target.");
         
      as->parameter ("nSubsamples",nSubsamples ,"number of subsampled registration labels per node (default=1)", false,optionalParameter);
      as->parameter ("pairwiseContrast",pairwiseContrastWeight ,"weight of contrast in pairwise segmentation potential (if not trained) (>=1)", false,optionalParameter);
      as->parameter ("alpha",alpha ,"generic weight (0)", false);
      as->parameter ("theta",theta ,"theta (offset) for segmentation pairwise potential", false,optionalParameter);
      as->parameter ("log",logFileName ,"cache output and flush to file at the end of the program", false);
      as->parameter ("groundTruth", groundTruthSegmentationFilename, "groundtruth segmentation of target, allows evaluation of result(file name)", false,optionalParameter);

      as->parameter ("verbose", verbose,"get verbose output",false);
      std::list<int> bla;
      //		as->values<int> (back_inserter(bla),"descr",nLevels);
      //	as->help();
      //as->defaultErrorHandling();
      as->parse();
	
      if (displacementSampling==-1) displacementSampling=maxDisplacement;
	
      levels=new int[nLevels+1];
      if (tmp_levels[0]==-1){
	levels[0]=startTiling;
      }else{
	levels[0]=tmp_levels[0];
      }
      for (int i=1;i<nLevels+1;++i){
	if (tmp_levels[i]==-1){
	  levels[i]=levels[i-1]*2-1;
	}else{
	  levels[i]=tmp_levels[i];
	}
      }
      nRegSamples=std::vector<int>(std::max(nLevels,1),maxDisplacement);
      if (regSampleString!=""){
	std::stringstream ss(regSampleString);
	int i;
	int c=0;
	while (ss >> i && c<nLevels)
	  {
	    nRegSamples[c]=i;
	    ++c; 
	    if (ss.peek() == 'x')
	      ss.ignore();
	  }
	for (;c<nLevels;++c){
	  nRegSamples[c]=i;
	}
            
      }

      resamplingFactors.push_back(1);
      resamplingFactors.push_back(0.5);
      resamplingFactors.push_back(0.3);
      resamplingFactors.push_back(0.25);
      resamplingFactors.push_back(0.2);
      resamplingFactors.push_back(0.17);
      resamplingFactors.push_back(0.15);
      resamplingFactors.push_back(0.15);
      resamplingFactors.push_back(0.15);        
      resamplingFactors.push_back(0.15);
      if (imageLevels<0)imageLevels=nLevels;
      
      coherence= (pairwiseCoherenceWeight>0);
      segment=pairwiseSegmentationWeight>0 ||  unarySegmentationWeight>0 || coherence;
      regist= pairwiseRegistrationWeight>0||  unaryRegistrationWeight>0|| coherence;

      if (solver == "GCO"){
	GCO=true;
      }else if (solver == "TRWS"){
	TRW=true;
      }else if (solver == "OPENGM"){
	OPENGM=true;
      }else{
	LOG<<"Choosen solver is "<<solver<<", will default to OPENGM if "<<solver<<" is not applicable."<<std::endl;
	OPENGM=true;
      }

    }
  };

}//namespace
#endif /* CONFIG_H_ */
