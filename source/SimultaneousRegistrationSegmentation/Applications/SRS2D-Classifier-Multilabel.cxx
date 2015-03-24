/**
 * @file   Multiresolution-SRS2D.cxx
 * @author gasst <gasst@ETHSEGREG>
 * @date   Thu Mar  5 13:15:31 2015
 * 
 * @brief  Example fuer 2D SRS using learned potentials
 * 
 * 
 */
#include <stdio.h>
#include <iostream>

#include "ArgumentParser.h"

#include "SRSConfig.h"
#include "HierarchicalSRSImageToImageFilter.h"
#include "Graph.h"
#include "FastGraph.h"
#include "BaseLabel.h"
#include "Potential-Registration-Unary.h"
#include "Potential-Registration-Pairwise.h"
#include "Potential-Segmentation-Unary.h"
#include "Potential-Coherence-Pairwise.h"
#include "Potential-Segmentation-Pairwise.h"
#include "Log.h"
#include "Preprocessing.h"
#include "TransformationUtils.h"

#include "Classifier-Segmentation-Unary-GMM.h"
#include "Classifier-Segmentation-Unary-RandomForest.h"
#include "Classifier-Segmentation-Pairwise-RandomForest.h"

using namespace std;
using namespace SRS;
using namespace itk;
int main(int argc, char ** argv)
{
	feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
	SRSConfig filterConfig;
	filterConfig.parseParams(argc,argv);
    if (filterConfig.logFileName!=""){
        mylog.setCachedLogging();
    }
    logSetStage("Init");

    //define types.
	
    typedef unsigned char PixelType;
	const unsigned int D=2;
	typedef Image<PixelType,D> ImageType;
    typedef ImageType::Pointer ImagePointerType;
    typedef ImageType::ConstPointer ImageConstPointerType;
	typedef TransfUtils<ImageType>::DisplacementType DisplacementType;
    typedef SparseRegistrationLabelMapper<ImageType,DisplacementType> LabelMapperType;
    //typedef SemiSparseRegistrationLabelMapper<ImageType,DisplacementType> LabelMapperType;
    //typedef DenseRegistrationLabelMapper<ImageType,DisplacementType> LabelMapperType;
    typedef TransfUtils<ImageType>::DeformationFieldType DeformationFieldType;
    typedef DeformationFieldType::Pointer DeformationFieldPointerType;



    //unary seg
    typedef ClassifierSegmentationUnaryGMMMultilabel<ImageType> ClassifierType;
    typedef UnaryPotentialNewSegmentationClassifier< ImageType, ClassifierType > SegmentationUnaryPotentialType;

    // //pairwise seg
    typedef ClassifierSegmentationPairwiseRandomForestWithGradient<ImageType> SegmentationSmoothnessClassifierType;
    typedef CachingPairwisePotentialSegmentationClassifier<ImageType,SegmentationSmoothnessClassifierType> SegmentationPairwisePotentialType;
    
    // //reg
    //typedef FastUnaryPotentialRegistrationSAD< LabelMapperType, ImageType > RegistrationUnaryPotentialType;
    typedef FastUnaryPotentialRegistrationNCC< ImageType > RegistrationUnaryPotentialType;
    
    typedef PairwisePotentialRegistration< ImageType > RegistrationPairwisePotentialType;
    
    typedef PairwisePotentialCoherence< ImageType > CoherencePairwisePotentialType;
 
#define POTENTIALINHERITANCE
#ifdef POTENTIALINHERITANCE
    typedef FastGraphModel<ImageType>        GraphType;
#else
    typedef FastGraphModel<ImageType,RegistrationUnaryPotentialType,RegistrationPairwisePotentialType,SegmentationUnaryPotentialType,SegmentationPairwisePotentialType,CoherencePairwisePotentialType>        GraphType;
#endif
    
    typedef HierarchicalSRSImageToImageFilter<GraphType>        FilterType;    
    //create filter
    FilterType::Pointer filter=FilterType::New();
    filter->setConfig(&filterConfig);
    
    logSetStage("Instantiate Potentials");
    

    RegistrationUnaryPotentialType::Pointer unaryRegistrationPot=RegistrationUnaryPotentialType::New();
    SegmentationUnaryPotentialType::Pointer unarySegmentationPot=SegmentationUnaryPotentialType::New();
    RegistrationPairwisePotentialType::Pointer pairwiseRegistrationPot=RegistrationPairwisePotentialType::New();
    SegmentationPairwisePotentialType::Pointer pairwiseSegmentationPot=SegmentationPairwisePotentialType::New();
    CoherencePairwisePotentialType::Pointer pairwiseCoherencePot=CoherencePairwisePotentialType::New();
#ifdef POTENTIALINHERITANCE

    filter->setUnaryRegistrationPotentialFunction(static_cast<typename FastUnaryPotentialRegistrationNCC<ImageType>::Pointer>(unaryRegistrationPot));
    filter->setPairwiseRegistrationPotentialFunction(static_cast<typename PairwisePotentialRegistration<ImageType>::Pointer>(pairwiseRegistrationPot));
    filter->setUnarySegmentationPotentialFunction(static_cast<typename UnaryPotentialSegmentation<ImageType>::Pointer>(unarySegmentationPot));
    filter->setPairwiseCoherencePotentialFunction(static_cast<typename PairwisePotentialCoherence<ImageType>::Pointer>(pairwiseCoherencePot));
    filter->setPairwiseSegmentationPotentialFunction(static_cast<typename PairwisePotentialSegmentation<ImageType>::Pointer>(pairwiseSegmentationPot));
#else
    filter->setUnaryRegistrationPotentialFunction((unaryRegistrationPot));
    filter->setPairwiseRegistrationPotentialFunction((pairwiseRegistrationPot));
    filter->setUnarySegmentationPotentialFunction((unarySegmentationPot));
    filter->setPairwiseCoherencePotentialFunction((pairwiseCoherencePot));
    filter->setPairwiseSegmentationPotentialFunction((pairwiseSegmentationPot));

#endif

    logUpdateStage("IO");
    logSetVerbosity(filterConfig.verbose);
    LOG<<"Loading target image :"<<filterConfig.targetFilename<<std::endl;
    ImagePointerType targetImage=ImageUtils<ImageType>::readImage(filterConfig.targetFilename);
#if 0
    if (filterConfig.normalizeImages){
        targetImage=FilterUtils<ImageType>::normalizeImage(targetImage);
    }
#endif

    if (!targetImage) {LOG<<"failed!"<<endl; exit(0);}
    LOG<<"Loading atlas image :"<<filterConfig.atlasFilename<<std::endl;
    ImagePointerType atlasImage;
    if (filterConfig.atlasFilename!="") {
        atlasImage=ImageUtils<ImageType>::readImage(filterConfig.atlasFilename);
#if 0
        if (filterConfig.normalizeImages){
            atlasImage=FilterUtils<ImageType>::normalizeImage(atlasImage);
        }
#endif
    }
    if (!atlasImage) {LOG<<"Warning: no atlas image loaded!"<<endl;
        LOG<<"Loading atlas segmentation image :"<<filterConfig.atlasSegmentationFilename<<std::endl;}
    ImagePointerType atlasSegmentation;
    if (filterConfig.atlasSegmentationFilename !="")atlasSegmentation=ImageUtils<ImageType>::readImage(filterConfig.atlasSegmentationFilename);
    if (!atlasSegmentation) {LOG<<"Warning: no atlas segmentation loaded!"<<endl; }
    
    ImagePointerType targetAnatomyPrior;
    if (filterConfig.targetAnatomyPriorFilename !="") {
        targetAnatomyPrior=ImageUtils<ImageType>::readImage(filterConfig.targetAnatomyPriorFilename);
        filterConfig.useTargetAnatomyPrior=true;
    }

    ImagePointerType atlasMaskImage=NULL;
    if (filterConfig.atlasMaskFilename!="") atlasMaskImage=ImageUtils<ImageType>::readImage(filterConfig.atlasMaskFilename);

    logResetStage;
    logSetStage("Preprocessing");

    if (filterConfig.histNorm){
        // Histogram match the images
        typedef itk::HistogramMatchingImageFilter<ImageType,ImageType> HEFilterType;
        HEFilterType::Pointer IntensityEqualizeFilter = HEFilterType::New();
        IntensityEqualizeFilter->SetReferenceImage(targetImage  );
        IntensityEqualizeFilter->SetInput( atlasImage );
        IntensityEqualizeFilter->SetNumberOfHistogramLevels( 100);
        IntensityEqualizeFilter->SetNumberOfMatchPoints( 15);
        IntensityEqualizeFilter->ThresholdAtMeanIntensityOn();
        IntensityEqualizeFilter->Update();
        atlasImage=IntensityEqualizeFilter->GetOutput();

    }

    //preprocessing 1: gradients
    ImagePointerType targetGradient, atlasGradient;
    if (filterConfig.segment){
       
  
        if (filterConfig.useTargetAnatomyPrior && ! targetAnatomyPrior.IsNotNull() ){
            //targetAnatomyPrior=Preprocessing<ImageType>::computeSoftTargetAnatomyEstimate(targetImage);
            LOG<<"NOT YET IMPLEMENTED: Preprocessing<ImageType>::computeSoftTargetAnatomyEstimate"<<endl;
            exit(0);
        }
        //preprocessing 2: multilabel
        if (filterConfig.computeMultilabelAtlasSegmentation){
            atlasSegmentation=FilterUtils<ImageType>::computeMultilabelSegmentation(atlasSegmentation);
            filterConfig.nSegmentations=5;//TODO!!!!
        }
    }
    logResetStage;

    ImagePointerType originalTargetImage=targetImage,originalAtlasImage=atlasImage,originalAtlasSegmentation=atlasSegmentation;
    //preprocessing 3: downscaling

    if (filterConfig.downScale<1){
        double sigma=1;
        double scale=filterConfig.downScale;
        LOG<<"Resampling images from "<< targetImage->GetLargestPossibleRegion().GetSize()<<" by a factor of"<<scale<<endl;
        targetImage=FilterUtils<ImageType>::LinearResample(targetImage,scale,true);
        if (atlasImage.IsNotNull()) atlasImage=FilterUtils<ImageType>::LinearResample(atlasImage,scale,true);
        if (atlasMaskImage.IsNotNull()) atlasMaskImage=FilterUtils<ImageType>::NNResample(atlasMaskImage,scale,false);
        if (atlasSegmentation.IsNotNull()) {
            atlasSegmentation=FilterUtils<ImageType>::NNResample((atlasSegmentation),scale,false);
            //ImageUtils<ImageType>::writeImage("testA.nii",atlasSegmentation);
        }
        if (filterConfig.segment){
            LOGV(3)<<"Resampling gradient images and anatomy prior by factor of "<<scale<<endl;
            targetGradient=FilterUtils<ImageType>::LinearResample(((ImageConstPointerType)targetGradient),scale,true);
            atlasGradient=FilterUtils<ImageType>::LinearResample(((ImageConstPointerType)atlasGradient),scale,true);
            //targetGradient=FilterUtils<ImageType>::NNResample(FilterUtils<ImageType>::gaussian((ImageConstPointerType)targetGradient,sigma),scale);
            //atlasGradient=FilterUtils<ImageType>::NNResample(FilterUtils<ImageType>::gaussian((ImageConstPointerType)atlasGradient,sigma),scale);
            if (filterConfig.useTargetAnatomyPrior){
                targetAnatomyPrior=FilterUtils<ImageType>::NNResample((targetAnatomyPrior),scale,false);
            }
            
        }
    }
    if (filterConfig.segment){
        if (atlasGradient.IsNotNull()) 
            LOGI(10,ImageUtils<ImageType>::writeImage("atlassheetness.nii",atlasGradient));
        LOGI(10,ImageUtils<ImageType>::writeImage("targetsheetness.nii",targetGradient));
    }


    logResetStage;
    filter->setTargetImage(targetImage);
    filter->setTargetGradient(targetGradient);
    filter->setAtlasImage(atlasImage);
    filter->setAtlasMaskImage(atlasMaskImage);
    filter->setAtlasGradient(atlasGradient);
    filter->setAtlasSegmentation(atlasSegmentation);
    if (filterConfig.useTargetAnatomyPrior){
        filter->setTargetAnatomyPrior(targetAnatomyPrior);
    }
    logSetStage("Bulk transforms");

    if (filterConfig.affineBulkTransform!=""){
        TransfUtils<ImageType>::AffineTransformPointerType affine=TransfUtils<ImageType>::readAffine(filterConfig.affineBulkTransform);
        LOGI(8,ImageUtils<ImageType>::writeImage("def.nii",TransfUtils<ImageType>::affineDeformImage(originalAtlasImage,affine,originalTargetImage)));
        //DeformationFieldPointerType transf=TransfUtils<ImageType>::affineToDisplacementField(affine,originalTargetImage);
        DeformationFieldPointerType transf=TransfUtils<ImageType>::affineToDisplacementField(affine,targetImage);
        LOGI(8,ImageUtils<ImageType>::writeImage("def2.nii",TransfUtils<ImageType>::warpImage((ImageType::ConstPointer)originalAtlasImage,transf)));
        filter->setBulkTransform(transf);
    }
    else if (filterConfig.bulkTransformationField!=""){
        filter->setBulkTransform(ImageUtils<DeformationFieldType>::readImage(filterConfig.bulkTransformationField));
    }else if (filterConfig.initWithMoments){
        //LOG<<" NOT NOT NOT Computing transform to move image centers on top of each other.."<<std::endl;
        LOG<<"initializing deformation using moments.."<<std::endl;
        DeformationFieldPointerType transf=TransfUtils<ImageType>::computeCenteringTransform(originalTargetImage,originalAtlasImage);
        filter->setBulkTransform(transf);
       
    }
    logResetStage;//bulk transforms
    originalTargetImage=NULL;
    originalAtlasImage=NULL;
    // compute SRS
    clock_t FULLstart = clock();
    filter->Init();
    logResetStage; //init

    filter->Update();
    logSetStage("Finalizing");
    clock_t FULLend = clock();
    float t = (float) ((double)(FULLend - FULLstart) / CLOCKS_PER_SEC);
    LOG<<"Finished computation after "<<t<<" seconds"<<std::endl;
    LOG<<"Unaries: "<<tUnary<<" Optimization: "<<tOpt<<std::endl;	
    LOG<<"Pairwise: "<<tPairwise<<std::endl;

    
    //process outputs
    ImagePointerType targetSegmentationEstimate=filter->getTargetSegmentationEstimate();
    DeformationFieldPointerType finalDeformation=filter->getFinalDeformation();
    
    delete filter;
    if (filterConfig.atlasFilename!="") originalAtlasImage=ImageUtils<ImageType>::readImage(filterConfig.atlasFilename);
    if (filterConfig.targetFilename!="") originalTargetImage=ImageUtils<ImageType>::readImage(filterConfig.targetFilename);

    //upsample?
    if (filterConfig.downScale<1){
        LOG<<"Upsampling Images.."<<endl;
       
        //it would probably be far better to create a surface for each label, 'upsample' that surface, and then create a binary volume for each surface which are merged in a last step
        if (targetSegmentationEstimate){
#if 0            
            targetSegmentationEstimate=FilterUtils<ImageType>::NNResample(targetSegmentationEstimate,originalTargetImage,false);
#else
            targetSegmentationEstimate=FilterUtils<ImageType>::upsampleSegmentation(targetSegmentationEstimate,originalTargetImage);
#endif
        }
    }

    if (targetSegmentationEstimate.IsNotNull()){

        ImageUtils<ImageType>::writeImage(filterConfig.segmentationOutputFilename,targetSegmentationEstimate);
    }
    
    if (finalDeformation.IsNotNull() ) {
        
        if (filterConfig.defFilename!="")
            ImageUtils<DeformationFieldType>::writeImage(filterConfig.defFilename,finalDeformation);
        if (filterConfig.linearDeformationInterpolation){
            finalDeformation=TransfUtils<ImageType>::linearInterpolateDeformationField(finalDeformation,(ImageConstPointerType)originalTargetImage,false);
         }else{
            finalDeformation=TransfUtils<ImageType>::bSplineInterpolateDeformationField(finalDeformation,(ImageConstPointerType)originalTargetImage);
        }
        
        LOG<<"Deforming Images.."<<endl;
        ImagePointerType deformedAtlasImage=TransfUtils<ImageType>::warpImage((ImageConstPointerType)originalAtlasImage,finalDeformation);
        ImageUtils<ImageType>::writeImage(filterConfig.outputDeformedFilename,deformedAtlasImage);
        LOGV(20)<<"Final SAD: "<<ImageUtils<ImageType>::sumAbsDist((ImageConstPointerType)deformedAtlasImage,(ImageConstPointerType)targetImage)<<endl;
        
        if (originalAtlasSegmentation.IsNotNull()){
            ImagePointerType deformedAtlasSegmentation=TransfUtils<ImageType>::warpImage((ImageConstPointerType)originalAtlasSegmentation,finalDeformation,true);
            ImageUtils<ImageType>::writeImage(filterConfig.outputDeformedSegmentationFilename,deformedAtlasSegmentation);
        }
        
        
    }
    
    OUTPUTTIMER;
    if (filterConfig.logFileName!=""){
        mylog.flushLog(filterConfig.logFileName);
    }
  
    return 1;
}
