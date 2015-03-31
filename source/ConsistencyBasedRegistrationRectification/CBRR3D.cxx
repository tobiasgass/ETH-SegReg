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
#include <itkSubtractImageFilter.h>
#include "itkFixedPointInverseDeformationFieldImageFilter.h"
#include "itkMemoryProbesCollectorBase.h"

#include "SolverConsistencyCBRR.h"

using namespace CBRR;

//using namespace std;
typedef short PixelType; 
static const unsigned int D=3 ;
typedef itk::Image<PixelType,D> ImageType;
typedef   ImageType::Pointer ImagePointerType;
typedef   ImageType::IndexType IndexType;
typedef   ImageType::PointType PointType;
typedef   ImageType::OffsetType OffsetType;
typedef   ImageType::SizeType SizeType;
typedef   ImageType::ConstPointer ImageConstPointerType;
typedef   ImageType::ConstPointer ConstImagePointerType;
typedef   ImageUtils<ImageType,double>::FloatImageType FloatImageType;
typedef   FloatImageType::Pointer FloatImagePointerType;


typedef   TransfUtils<ImageType>::DisplacementType DisplacementType;
typedef   TransfUtils<ImageType>::DeformationFieldType DeformationFieldType;
typedef   DeformationFieldType::Pointer DeformationFieldPointerType;
typedef   itk::ImageRegionIterator<ImageType> ImageIteratorType;
typedef   itk::ImageRegionIterator<FloatImageType> FloatImageIteratorType;
typedef   itk::ImageRegionIterator<DeformationFieldType> DeformationIteratorType;
typedef   itk::ConstNeighborhoodIterator<ImageType> ImageNeighborhoodIteratorType;
typedef   ImageNeighborhoodIteratorType * ImageNeighborhoodIteratorPointerType;
typedef   ImageNeighborhoodIteratorType::RadiusType RadiusType;

typedef  itk::ImageRegionIteratorWithIndex<DeformationFieldType> DeformationFieldIterator;
typedef  itk::AddImageFilter<DeformationFieldType,DeformationFieldType,DeformationFieldType> DeformationAddFilterType;
typedef  DeformationAddFilterType::Pointer DeformationAddFilterPointer;
typedef  itk::SubtractImageFilter<DeformationFieldType,DeformationFieldType,DeformationFieldType> DeformationSubtractFilterType;

typedef  itk::FixedPointInverseDeformationFieldImageFilter<DeformationFieldType,DeformationFieldType> InverseDeformationFieldFilterType;
typedef  InverseDeformationFieldFilterType::Pointer InverseDeformationFieldFilterPointerType;
enum MetricType {NONE,MAD,NCC,MI,NMI,MSD};
enum WeightingType {UNIFORM,GLOBAL,LOCAL};
enum SolverType {LOCALDEFORMATIONANDERROR};

typedef map<string, map< string, string> > FileListCacheType;
typedef  map<string,ImagePointerType> ImageCacheType;
typedef  map<string, ImagePointerType>::iterator ImageListIteratorType;
typedef map< string, map <string, DeformationFieldPointerType> > DeformationCacheType;



int main(int argc, char ** argv){

  


    double m_sigma;
    RadiusType m_patchRadius;
    feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    ArgumentParser * as=new ArgumentParser(argc,argv);
    string maskFileList="",groundTruthSegmentationFileList="",landmarkFileList="",deformationFileList,imageFileList,atlasSegmentationFileList,supportSamplesListFileName="",outputDir="",outputSuffix="",weightListFilename="",trueDefListFilename="",ROIFilename="";
    int verbose=0;
    double pWeight=1.0;
    int radius=4;
    int maxHops=1;
    int maxLevels=1;
    bool updateDeformations=false;
    bool locallyUpdateDeformations=false;
    string metricName="NCC";
    string weightingName="uniform";
    bool lateFusion=false;
    bool dontCacheDeformations=false;
    bool graphCut=false;
    double smoothness=0.0;
    double lambda=0.0;
    double resamplingFactor=4.0;
    double imageResamplingFactor=-1.0;
    m_sigma=4;
    string solverName="localnorm";
    double wwd=0.0,winput=1.0,wsmooth=0.0,wcons=1.0,wwdelta=0.0,wsmoothum=0,wsdelta=0.0,m_exponent=10.0,wwInconsistencyError=0.0,wErrorStatistics=0.0,wSymmetry=0.0;
    bool nearestneighb=false;
    double shearing = 1.0;
    double circWeightScaling = 1.0;
    double scalingFactorForConsistentSegmentation = 1.0;
    double oracle = 0;
    string localSimMetric="lncc";
    bool evalLowResolutionDeformationss=false;
    bool roiShift=false;
    bool smoothDownsampling=false;
    bool bSplineResampling=false;
    bool filterMetricWithGradient=false;
    bool lineSearch=false;
    bool useConstraints=false;
    bool updateDeformationsGlobalWeight=false;
    string optimizer="csd:100";
    double tolerance=1e-2;
    bool useTaylor=false;
    bool lowResSim=false;
    bool normalizeForces=false;
    int maxTripletOcc=100000;
    as->parameter ("i", imageFileList, " list of  images", true);
    as->parameter ("T", deformationFileList, " list of deformations", true);
    as->parameter ("true", trueDefListFilename, " list of TRUE deformations", false);
    as->parameter ("ROI", ROIFilename, "file containing a ROI on which to perform erstimation", false);
    as->parameter ("resamplingFactor", resamplingFactor,"lower resolution by a factor",false);
    as->parameter ("optimizer", optimizer,"optimizer for lsq problem. optional number of iterations, eg lbfgsx100.opt in {lsqlin,cg,csd,,lbfgs,cgd}",false);
    as->parameter ("imageResamplingFactor", imageResamplingFactor,"lower image resolution by a different factor. This will lead to having more equations for the regularization than there are variables, with the chosen interpolation affecting the interpolation.",false);
    as->parameter ("winp", winput,"weight for adherence to input registration",false);
    as->parameter ("wcons", wcons,"weight consistency penalty",false);
    as->parameter ("wsmooth", wsmooth,"weight for smoothness of deformation (first-order derivative)",false);
    as->parameter ("maxOcc", maxTripletOcc,"maximal number of triplets in which a pairwise registration can occur.",false);

    as->parameter ("A",atlasSegmentationFileList , "list of atlas segmentations <id> <file>", false);
    as->parameter ("groundTruthSegmentations",groundTruthSegmentationFileList , "list of groundTruth segmentations <id> <file> for immediate DICE evaluation", false);
    as->parameter ("landmarks",landmarkFileList , "list of landmark files <id> <file> for immediate TRE evaluation", false);


    as->parameter ("masks", maskFileList, " list of  binary masks used to compute inconsistency", false);
    as->parameter ("solver", solverName,"solver used {globalnorm,localnorm,localerror,localcomposederror,localdeformationanderror}",false);
    as->parameter ("tol", tolerance, " stopping criterion on the relative change of the inconsistency", false);
    as->parameter ("s", m_sigma," kernel width for lncc",false);
    as->parameter ("exp",m_exponent ,"exponent for local similarity weights",false);


    as->parameter ("O", outputDir,"outputdirectory (will be created + no overwrite checks!)",false);
    as->parameter ("maxHops", maxHops,"maximum number of hops per level",false);
    as->parameter ("maxLevels", maxLevels,"maximum number of multi-resolution levels",false);
    as->option ("useTaylor", useTaylor,"use something similar to first order taylor approximation for inconsistency terms.");
    as->option ("lowResSim", lowResSim,"compute local similarities/gradients only at ROI resolution instead of image resolution.");
    as->option ("smoothDownsampling", smoothDownsampling,"Smooth deformation before downsampling. will capture errors between grid points, but will miss other inconsistencies due to the smoothing.");
    as->option ("bSpline", bSplineResampling,"Use bSlpines for resampling the deformation fields. A lot slower, especially in 3D.");
    as->option ("lineSearch", lineSearch,"Use (simple) line search to determine update step width, based on global NCC.");
    as->option ("normalizeForces", normalizeForces,"divide inconsistency and regularization equation weights by their respective average to equalize the forces.");
    as->option ("useConstraints", useConstraints,"Use hard constraints to prevent folding. Tearing might currently still occur.");


    as->parameter ("metric",localSimMetric ,"metric to be used for local sim computation (none,lncc, lsad, lssd,localautocorrelation).",false);
    as->option ("filterMetricWithGradient", filterMetricWithGradient,"Multiply local metric with target and warped source image gradients to filter out smooth regions.");

    as->option ("updateDeformations", updateDeformations," use estimate of previous iteration in next one.");
    as->option ("updateDeformationsGlobalWeight", updateDeformationsGlobalWeight," use estimate of previous iteration in next one IF global similarity improved.");
    as->option ("locallyUpdateDeformations", locallyUpdateDeformations," locally use better (in terms of similarity) from initial and prior Deformation estimate as target in next iteration.");
    as->option ("evalLowResolutionDeformations", evalLowResolutionDeformationss," Use only the (upsampled) low resolution deformation for further processing. This is faster (ofc), but less accurate.");

    as->parameter ("shearing",shearing ,"reduction coefficient for shearing potentials in spatial smoothing",false);
    as->parameter ("circScale", circWeightScaling,"scaling of circ weight per iteration ",false);
    as->option ("nearestneighb", nearestneighb," use nearestneighb interpolation (instead of NN) when building equations for circles.");
    as->parameter ("segmentationConsistencyScaling",scalingFactorForConsistentSegmentation,"factor for increasing the weight on consistency for segmentated pixels",false);

    as->parameter ("wsmoothum", wsmoothum,"EXPERIMENTAL: weight for def1 in circle",false);
    as->parameter ("wsmoothym", wSymmetry,"EXPERIMENTAL: weight for def1 in circle",false);
    as->parameter ("wwincerr",wwInconsistencyError ,"EXPERIMENTAL: weight for def1 in circle",false);
    as->parameter ("wErrorStatistics",wErrorStatistics ,"EXPERIMENTAL: weight for error variable being forced to be similar to the inconsitency statistics",false);
    as->option ("roiShift", roiShift,"EXPERIMENTAL: Shift ROI by half spacing after each iteration, to sample from different points.");

    as->parameter ("wwd", wwd,"EXPERIMENTAL: weight for def1 in circle",false);
    as->parameter ("wsdelta", wsdelta,"EXPERIMENTAL: weight for def1 in circle",false);
    as->parameter ("wwdelta", wwdelta,"EXPERIMENTAL: weight for def1 in circle",false);
    //        as->option ("graphCut", graphCut,"use graph cuts to generate final segmentations instead of locally maximizing");
    //as->parameter ("smoothness", smoothness,"smoothness parameter of graph cut optimizer",false);
    as->parameter ("ORACLE", oracle," oracle=1:use true deformation for indexing variables in loops.CHEATING!!. oracle=2: additianlly use true def as initial values. oracle = 3: use true def adherence  ",false);

    as->parameter ("verbose", verbose,"get verbose output",false);
    as->parse();
    
    
    if (imageResamplingFactor<0){
        imageResamplingFactor=resamplingFactor;
    }
    //late fusion is only well defined for maximal 1 hop.
    //it requires to explicitly compute all n!/(n-nHops) deformation paths to each image and is therefore infeasible for nHops>1
    //also strange to implement
    if (lateFusion)
        maxHops==min(maxHops,1);

    for (unsigned int i = 0; i < ImageType::ImageDimension; ++i) m_patchRadius[i] = radius;

     if (updateDeformationsGlobalWeight) updateDeformations=true;

    logSetStage("IO");
    logSetVerbosity(verbose);
        
    MetricType metric;
  

    
    //read images and image IDs
    ImageCacheType inputImages;
    std::vector<string> imageIDs;
    LOG<<"Reading input images."<<endl;
    inputImages = ImageUtils<ImageType>::readImageList( imageFileList, imageIDs );
    int nImages = inputImages.size();

    ImageCacheType * inputMasks=NULL;
    if (maskFileList!=""){
        inputMasks=new ImageCacheType;
        std::vector<string> buff;
        LOG<<"Reading input masks."<<endl;
        (*inputMasks) = ImageUtils<ImageType>::readImageList( maskFileList, buff );
    }
        
    //read landmark filenames
    map<string,string> landmarkList;
    if (landmarkFileList!=""){
        ifstream ifs(landmarkFileList.c_str());
        while (!ifs.eof()){
            string sourceID,landmarkFilename;
            ifs >> sourceID;
            ifs >>landmarkFilename;
            landmarkList[sourceID]=landmarkFilename;
        }

    }
    
    FileListCacheType  deformationFilenames, trueDeformationFilenames;

    //read  target deformations filenames
    
    ifstream ifs(deformationFileList.c_str());
    while (!ifs.eof()){
        string sourceID,targetID,defFileName;
        ifs >> sourceID;
        if (sourceID!=""){
            ifs >> targetID;
            ifs >> defFileName;
            if (inputImages.find(sourceID)==inputImages.end() || inputImages.find(targetID)==inputImages.end() ){
                LOGV(3)<<sourceID<<" or "<<targetID<<" not in image database, skipping"<<endl;
                //exit(0);
            }else{
                LOGV(3)<<"Reading deformation "<<defFileName<<" for deforming "<<sourceID<<" to "<<targetID<<endl;
                deformationFilenames[sourceID][targetID]=defFileName;
            }
        }
    }
    
    //read  true deformation filenames
    if (trueDefListFilename!=""){
        ifstream ifs(trueDefListFilename.c_str());
        while (!ifs.eof()){
            string sourceID,targetID,defFileName;
            ifs >> sourceID;
            if (sourceID!=""){
                ifs >> targetID;
                ifs >> defFileName;
                if (inputImages.find(sourceID)==inputImages.end() || inputImages.find(targetID)==inputImages.end() ){
                    LOGV(3)<<sourceID<<" or "<<targetID<<" not in image database, skipping"<<endl;
                    //exit(0);
                }else{
                    trueDeformationFilenames[sourceID][targetID]=defFileName;
                }  
            }
        }
    }




   
    //create ROI, either from first image or from a file
    ImagePointerType origReference=ImageUtils<ImageType>::duplicate( (inputImages)[imageIDs[0]]);
    ImagePointerType ROI,grid;
    if (ROIFilename!="") {
        ROI=ImageUtils<ImageType>::readImage(ROIFilename);
    }else{
        ROI=origReference;
    }
    //resample ROI ? should/could be done within CBRR?
    if (resamplingFactor>1.0){
        ROI=FilterUtils<ImageType>::LinearResample(ROI,1.0/imageResamplingFactor,false);
	
    }
    if (outputDir!=""){
        mkdir(outputDir.c_str(),0755);
    }
    grid=FilterUtils<ImageType>::LinearResample(origReference,1.0/resamplingFactor,false);
    
    //load atlas and/or groundtruth segmentations (or none)
    ImageCacheType atlasSegmentations ;
    if (atlasSegmentationFileList!=""){
        std::vector<string> buff;
        atlasSegmentations=ImageUtils<ImageType>::readImageList(atlasSegmentationFileList,buff);
    }
    ImageCacheType groundTruthSegmentations;
    if (groundTruthSegmentationFileList!=""){
        std::vector<string> buff;
        groundTruthSegmentations=ImageUtils<ImageType>::readImageList(groundTruthSegmentationFileList,buff);
    }

    //create solver
    SolverConsistencyCBRR<ImageType> * solver;
    solver= new SolverConsistencyCBRR<ImageType>;
    
    solver->setOracle(oracle);
    solver->setWeightFullCircleEnergy(wwd);
    solver->setWeightDeformationSmootheness(wsmooth);
    solver->setWeightTransformationSimilarity(winput);
    solver->setWeightTransformationSymmetry(wSymmetry);
    solver->setWeightErrorNorm(wwdelta);
    solver->setWeightErrorSmootheness(wsdelta);
    solver->setWeightCircleNorm(wcons); 
    solver->setWeightSum(wsmoothum); 
    solver->setWeightInconsistencyError(wwInconsistencyError); 
    solver->setWeightErrorStatistics(wErrorStatistics); 
    solver->setUpdateDeformations(updateDeformations); 
    solver->setLocallyUpdateDeformations(locallyUpdateDeformations);
    solver->setSmoothDeformationDownsampling(smoothDownsampling);
    solver->setBSplineInterpol(bSplineResampling);
    solver->setLowResEval(evalLowResolutionDeformationss);
    solver->setLinearInterpol(!nearestneighb);
    solver->setSigma(m_sigma);
    solver->setLocalWeightExp(m_exponent);
    solver->setShearingReduction(shearing);
    solver->setMetric(localSimMetric);
    solver->setFilterMetricWithGradient(filterMetricWithGradient);
    solver->setLineSearch(lineSearch);
    solver->setUseConstraints(useConstraints);

    solver->setDeformationFilenames(deformationFilenames);
    solver->setTrueDeformationFilenames(trueDeformationFilenames);
    solver->setLandmarkFilenames(landmarkList);
    solver->setAtlasSegmentations(atlasSegmentations);
    solver->setGroundTruthSegmentations(groundTruthSegmentations);
    solver->setImageIDs(imageIDs);
    solver->setImages(inputImages);
    solver->setMasks(inputMasks);
    solver->setROI(ROI);
    solver->setUseTaylor(useTaylor);
    solver->setLowResSim(lowResSim);
    solver->setNormalizeForces(normalizeForces);
    solver->setMaxTripletOccs(maxTripletOcc);
    solver->setGrid(grid);
    solver->setOptimizer(optimizer);
    solver->Initialize();
    int c=1;

    for (int level=0;level<maxLevels;++level){
        int iter = 0;
        double error=solver->getADE();
        double inconsistency=solver->getInconsistency();
        double TRE=solver->getTRE();
        double dice=solver->getDice();
        double oldInconsistency=inconsistency;
        double minJac=solver->getMinJac();
        double averageNCC=solver->getAverageNCC();
        double stdJac=solver->getAverageJacSTD();
        double averageLNCC=solver->getAverageLNCC();
        LOG<<VAR(iter)<<" "<<VAR(error)<<" "<<VAR(inconsistency)<<" "<<VAR(TRE)<<" "<<VAR(dice)<<" "<<VAR(averageNCC)<<" "<<VAR(averageLNCC)<<" "<<VAR(minJac)<<" "<<VAR(stdJac)<<endl;
        for (iter=1;iter<maxHops+1;++iter){
            if (! iter % 2){
                LOGV(2)<<"Increasing image resolution for bspline registration"<<endl;
                solver->doubleImageResolution();
            }
            itk::MemoryProbesCollectorBase memorymeter;
            //memorymeter.Start( "CBRR complete" );

            solver->createSystem();
            solver->solve();
            //compute and store results. For efficiency reasons, CBRR computes all metrics in one go within this routine.
            solver->storeResult("");
            if (roiShift){
                //shift ROI by half spacing to get different sampling in next iteration
                ROI->SetOrigin(ROI->GetOrigin()+ pow(-1.0,1.0*(iter-1))*0.5*ROI->GetSpacing());
                LOGV(1)<<VAR(ROI->GetOrigin())<<endl;
            }
            solver->ComputeAndEvaluateResults(outputDir);
            //memorymeter.Stop( "CBRR complete" );
            //memorymeter.Report( std::cout );
            error=solver->getADE();
            inconsistency=solver->getInconsistency();
            TRE=solver->getTRE();
            dice=solver->getDice();
            minJac=solver->getMinJac();
            averageNCC=solver->getAverageNCC();
            averageLNCC=solver->getAverageLNCC();

            stdJac=solver->getAverageJacSTD();


            if (false && updateDeformations){
                solver->setWeightTransformationSimilarity(winput*pow(1.2,1.0*iter),true);
                ++c;
            }
            if (iter == maxHops){
                //double resolution
            }
            if (iter >1 && (tolerance*(oldInconsistency-inconsistency)<-0.1)){
                LOG<<"inconsistency increased("<<inconsistency<<" , stopping refinement."<<endl;
                break;
            }
            LOG<<VAR(iter)<<" "<<VAR(error)<<" "<<VAR(inconsistency)<<" "<<VAR(TRE)<<" "<<VAR(dice)<<" "<<VAR(averageNCC)<<" "<<VAR(averageLNCC)<<" "<<VAR(minJac)<<" "<<VAR(stdJac)<<endl;
            if (iter >1 && oldInconsistency>0.0 && fabs(oldInconsistency-inconsistency)/oldInconsistency <=tolerance){
                LOG<<"Convergence reached, stopping refinement."<<endl;
                break;
            }
          
            

            oldInconsistency=inconsistency;
            
        }
        
        if (level !=maxLevels-1){
            //resample ROI for next level with increased resolution
            ROI=FilterUtils<ImageType>::LinearResample(ROI,2.0,false,true);
            grid=FilterUtils<ImageType>::LinearResample(grid,2.0,false);
            solver->setROI(ROI);
            solver->setGrid(grid);
            solver->Initialize();
            solver->ComputeAndEvaluateResults();
        }

    }//levels
    
    
    return 1;
}//main
