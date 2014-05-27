#include <stdio.h>
#include <iostream>

#include "argstream.h"
#include "Log.h"
#include <vector>
#include <map>
#include "itkImageRegionIterator.h"
#include "TransformationUtils.h"
#include "ImageUtils.h"
#include "FilterUtils.hpp"
#include <sstream>
#include "argstream.h"
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
#include "CLERCIndependentDimensions.h"

//using namespace std;
typedef unsigned char PixelType;
static const unsigned int D=2 ;
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

double computeError( map< string, map <string, DeformationFieldPointerType> >  & defs,  map< string, map <string, DeformationFieldPointerType> > & trueDefs){
    
    return 0.0;

}
  





int main(int argc, char ** argv){

  


    double m_sigma;
    RadiusType m_patchRadius;
    feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    argstream * as=new argstream(argc,argv);
    string maskFileList="",groundTruthSegmentationFileList="",landmarkFileList="",deformationFileList,imageFileList,atlasSegmentationFileList,supportSamplesListFileName="",outputDir="",outputSuffix="",weightListFilename="",trueDefListFilename="",ROIFilename="";
    int verbose=0;
    double pWeight=1.0;
    int radius=3;
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
    double resamplingFactor=8.0;
    m_sigma=10;
    string solverName="localnorm";
    double wwd=0.0,wwt=1.0,wws=0.0,wwcirc=1.0,wwdelta=0.0,wwsum=0,wsdelta=0.0,m_exponent=1.0,wwInconsistencyError=0.0,wErrorStatistics=0.0,wSymmetry=0.0;
    bool nearestneighb=false;
    double shearing = 1.0;
    double circWeightScaling = 1.0;
    double scalingFactorForConsistentSegmentation = 1.0;
    bool oracle = false;
    string localSimMetric="lncc";
    bool evalLowResolutionDeformationss=false;
    bool roiShift=false;
    bool smoothDownsampling=false;
    bool bSplineResampling=false;
    bool filterMetricWithGradient=false;
    bool lineSearch=false;
    bool useConstraints=false;
    double annealing=1.0;
    int robustLSQIter=0;
    double robustFitTuningParam=2.12;
    (*as) >> parameter ("T", deformationFileList, " list of deformations", true);
    (*as) >> parameter ("true", trueDefListFilename, " list of TRUE deformations", false);
    (*as) >> parameter ("ROI", ROIFilename, "file containing a ROI on which to perform erstimation", false);

    (*as) >> parameter ("i", imageFileList, " list of  images", true);
    (*as) >> parameter ("masks", maskFileList, " list of  binary masks used to compute inconsistency", false);
    (*as) >> parameter ("solver", solverName,"solver used {globalnorm,localnorm,localerror,localcomposederror,localdeformationanderror}",false);
    (*as) >> parameter ("s", m_sigma," kernel width for lncc",false);
    (*as) >> parameter ("O", outputDir,"outputdirectory (will be created + no overwrite checks!)",false);
    (*as) >> parameter ("maxHops", maxHops,"maximum number of hops per level",false);
    (*as) >> parameter ("maxLevels", maxLevels,"maximum number of multi-resolution levels",false);

    (*as) >> parameter ("wwd", wwd,"weight for def1 in circle",false);
    (*as) >> parameter ("wwt", wwt,"weight for def1 in circle",false);
    (*as) >> parameter ("wws", wws,"weight for def1 in circle",false);
    (*as) >> parameter ("wsdelta", wsdelta,"weight for def1 in circle",false);
    (*as) >> parameter ("wwdelta", wwdelta,"weight for def1 in circle",false);
    (*as) >> parameter ("wwcirc", wwcirc,"weight for def1 in circle",false);
    (*as) >> parameter ("wwsum", wwsum,"weight for def1 in circle",false);
    (*as) >> parameter ("wwsym", wSymmetry,"weight for def1 in circle",false);
    (*as) >> parameter ("wwincerr",wwInconsistencyError ,"weight for def1 in circle",false);
    (*as) >> parameter ("wErrorStatistics",wErrorStatistics ,"weight for error variable being forced to be similar to the inconsitency statistics",false);
    (*as) >> parameter ("robustLSQIter",robustLSQIter ,"iterations for robust least square reweighting based on bisquare weights. !",false);
    (*as) >> parameter ("robustFitTuningParam",robustFitTuningParam ,"Tuning parameter for robust lsq fitting. bigger value will lead to weaker penalization of outliers, and vice versa!",false);
    (*as) >> option ("roiShift", roiShift,"Shift ROI by half spacing after each iteration, to sample from different points.");
    (*as) >> option ("smoothDownsampling", smoothDownsampling,"Smooth deformation before downsampling. will capture errors between grid points, but will miss other inconsistencies due to the smoothing.");
    (*as) >> option ("bSpline", bSplineResampling,"Use bSlpines for resampling the deformation fields. A lot slower, especially in 3D.");
    (*as) >> option ("lineSearch", lineSearch,"Use (simple) line search to determine update step width, based on global NCC.");
    (*as) >> option ("useConstraints", useConstraints,"Use hard constraints to prevent folding. Tearing might currently still occur.");


    (*as) >> parameter ("metric",localSimMetric ,"metric to be used for local sim computation (lncc, lsad, lssd,localautocorrelation).",false);
    (*as) >> option ("filterMetricWithGradient", filterMetricWithGradient,"Multiply local metric with target and warped source image gradients to filter out smooth regions.");

    (*as) >> option ("updateDeformations", updateDeformations," use estimate of previous iteration in next one.");
    (*as) >> parameter ("annealing",annealing ,"increase regularization by annealing factor in each iteration.",false);

    (*as) >> option ("locallyUpdateDeformations", locallyUpdateDeformations," locally use better (in terms of similarity) from initial and prior Deformation estimate as target in next iteration.");
    (*as) >> option ("evalLowResolutionDeformations", evalLowResolutionDeformationss," Use only the (upsampled) low resolution deformation for further processing. This is faster (ofc), but less accurate.");

    (*as) >> parameter ("exp",m_exponent ,"exponent for local similarity weights",false);
    (*as) >> parameter ("shearing",shearing ,"reduction coefficient for shearing potentials in spatial smoothing",false);
    (*as) >> parameter ("circScale", circWeightScaling,"scaling of circ weight per iteration ",false);
    (*as) >> option ("nearestneighb", nearestneighb," use nearestneighb interpolation (instead of NN) when building equations for circles.");
    (*as) >> parameter ("segmentationConsistencyScaling",scalingFactorForConsistentSegmentation,"factor for increasing the weight on consistency for segmentated pixels",false);
    (*as) >> parameter ("A",atlasSegmentationFileList , "list of atlas segmentations <id> <file>", false);
    (*as) >> parameter ("groundTruthSegmentations",groundTruthSegmentationFileList , "list of groundTruth segmentations <id> <file>", false);
    (*as) >> parameter ("landmarks",landmarkFileList , "list of landmark files <id> <file>", false);

    //        (*as) >> option ("graphCut", graphCut,"use graph cuts to generate final segmentations instead of locally maximizing");
    //(*as) >> parameter ("smoothness", smoothness,"smoothness parameter of graph cut optimizer",false);
    (*as) >> parameter ("resamplingFactor", resamplingFactor,"lower resolution by a factor",false);
    (*as) >> option ("ORACLE", oracle," use true deformation for indexing variables in loops.CHEATING!!.");

    (*as) >> parameter ("verbose", verbose,"get verbose output",false);
    (*as) >> help();
    as->defaultErrorHandling();
       
    //late fusion is only well defined for maximal 1 hop.
    //it requires to explicitly compute all n!/(n-nHops) deformation paths to each image and is therefore infeasible for nHops>1
    //also strange to implement
    if (lateFusion)
        maxHops==min(maxHops,1);

    for (unsigned int i = 0; i < ImageType::ImageDimension; ++i) m_patchRadius[i] = radius;

    ++robustLSQIter;
 
    logSetStage("IO");
    logSetVerbosity(verbose);
        
    MetricType metric;
  
    //only one solver
    SolverType solverType=LOCALDEFORMATIONANDERROR;
    if (solverName=="localdeformationanderror"){
        solverType=LOCALDEFORMATIONANDERROR;
    } 

    
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
    ImagePointerType ROI;
    if (ROIFilename!="") {
        ROI=ImageUtils<ImageType>::readImage(ROIFilename);
    }else{
        ROI=origReference;
    }
    //resample ROI ? should/could be done within CLERC?
    if (resamplingFactor>1.0)
        ROI=FilterUtils<ImageType>::LinearResample(ROI,1.0/resamplingFactor,false);
    
    if (outputDir!=""){
        mkdir(outputDir.c_str(),0755);
    }

    
    //load atlas and/or groundtruth segmentations (or none)
    ImageCacheType atlasSegmentations ;
    if (atlasSegmentationFileList!=""){
        std::vector<string> buff;
        atlasSegmentations=ImageUtils<ImageType>::readImageList(atlasSegmentationFileList,buff);
    }
    LOGV(1)<<VAR(atlasSegmentations.size())<<endl;
    ImageCacheType groundTruthSegmentations;
    if (groundTruthSegmentationFileList!=""){
        std::vector<string> buff;
        groundTruthSegmentations=ImageUtils<ImageType>::readImageList(groundTruthSegmentationFileList,buff);
    }

    //create solver
    CLERCIndependentDimensions<ImageType> * solver;
    
    switch(solverType){
    case LOCALDEFORMATIONANDERROR:
        //solver= new AquircLocalDeformationAndErrorSolver<ImageType>;
        solver= new CLERCIndependentDimensions<ImageType>;
        break;
    }
    
    solver->setOracle(oracle);
    solver->setWeightFullCircleEnergy(wwd);
    solver->setWeightDeformationSmootheness(wws);
    solver->setWeightTransformationSimilarity(wwt);
    solver->setWeightTransformationSymmetry(wSymmetry);
    solver->setWeightErrorNorm(wwdelta);
    solver->setWeightErrorSmootheness(wsdelta);
    solver->setWeightCircleNorm(wwcirc); 
    solver->setWeightSum(wwsum); 
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
    solver->setRobustLSQIter(robustLSQIter);
    solver->setRobustFitTuningParam(robustFitTuningParam);
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
        LOG<<VAR(iter)<<" "<<VAR(error)<<" "<<VAR(inconsistency)<<" "<<VAR(TRE)<<" "<<VAR(dice)<<" "<<VAR(averageNCC)<<" "<<VAR(minJac)<<endl;
        for (iter=1;iter<maxHops+1;++iter){
            if (false && ! iter % 5){
                solver->doubleImageResolution();
            }
            solver->createSystem();
            solver->solve();
            //compute and store results. For efficiency reasons, CLERC computes all metrics in one go within this routine.
            solver->storeResult("");
            if (roiShift){
                //shift ROI by half spacing to get different sampling in next iteration
                ROI->SetOrigin(ROI->GetOrigin()+ pow(-1.0,1.0*(iter-1))*0.5*ROI->GetSpacing());
                LOGV(1)<<VAR(ROI->GetOrigin())<<endl;
            }
            solver->DoALot(outputDir);

            error=solver->getADE();
            inconsistency=solver->getInconsistency();
            TRE=solver->getTRE();
            dice=solver->getDice();
            minJac=solver->getMinJac();
            averageNCC=solver->getAverageNCC();
            LOG<<VAR(iter)<<" "<<VAR(error)<<" "<<VAR(inconsistency)<<" "<<VAR(TRE)<<" "<<VAR(dice)<<" "<<VAR(averageNCC)<<" "<<VAR(minJac)<<endl;
            solver->setWeightTransformationSimilarity(wwt*pow(annealing,1.0*iter),true);
            ++c;
                
            if (iter == maxHops){
                //double resolution
            }
            if (fabs(oldInconsistency-inconsistency)/oldInconsistency <=1e-2){
                LOG<<"Convergence reached, stopping refinement."<<endl;
                break;
            }
            oldInconsistency=inconsistency;
            
        }
        
        if (level !=maxLevels-1){
            //resample ROI for next level with increased resolution
            ROI=FilterUtils<ImageType>::LinearResample(ROI,2.0,false,true);
            solver->setROI(ROI);
            solver->DoALot();
        }

    }//levels
    
    
    return 1;
}//main
