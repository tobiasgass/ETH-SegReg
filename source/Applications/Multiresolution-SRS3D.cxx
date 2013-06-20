#include <stdio.h>
#include <iostream>

#include "argstream.h"

#include "SRSConfig.h"
#include "HierarchicalSRSImageToImageFilter.h"
#include "Graph.h"
#include "SubsamplingGraph.h"
#include "FastRegistrationGraph.h"
#include "BaseLabel.h"
#include "Potential-Registration-Unary.h"
#include "Potential-Registration-Pairwise.h"
#include "Potential-Segmentation-Unary.h"
#include "Potential-Coherence-Pairwise.h"
#include "Potential-Segmentation-Pairwise.h"
#include "MRF-TRW-S.h"
#include "Log.h"
#include "Preprocessing.h"
#include "TransformationUtils.h"
#include "NewClassifier.h"
using namespace std;
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
	
    typedef short int PixelType;
	const unsigned int D=3;
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
    //typedef SegmentationClassifierGradient<ImageType> ClassifierType;
    //    typedef SegmentationGenerativeClassifierGradient<ImageType> ClassifierType;
    //typedef     SegmentationGaussianClassifierGradient<ImageType> ClassifierType;
    //typedef SegmentationClassifier<ImageType> ClassifierType;
    //typedef HandcraftedBoneSegmentationClassifierMarcel<ImageType> ClassifierType;
    //typedef UnaryPotentialSegmentationClassifier< ImageType, ClassifierType > SegmentationUnaryPotentialType;
    //typedef UnaryPotentialSegmentationBoneMarcel< ImageType > SegmentationUnaryPotentialType;
    typedef UnaryPotentialSegmentationUnsignedBoneMarcel< ImageType > SegmentationUnaryPotentialType;
    // //typedef     UnaryPotentialSegmentation< ImageType > SegmentationUnaryPotentialType;
    
    // //typedef SegmentationRandomForestClassifier<ImageType> ClassifierType;
    // //typedef SegmentationGMMClassifier<ImageType> ClassifierType;
    // //typedef UnaryPotentialNewSegmentationClassifier< ImageType, ClassifierType > SegmentationUnaryPotentialType;

    // //pairwise seg
    // typedef SmoothnessClassifierGradient<ImageType> SegmentationSmoothnessClassifierType;
    // //typedef SmoothnessClassifierGradientContrast<ImageType> SegmentationSmoothnessClassifierType;
    // //typedef SmoothnessClassifierFullMultilabelPosterior<ImageType> SegmentationSmoothnessClassifierType;
    // //typedef CachingPairwisePotentialSegmentationClassifier<ImageType,SegmentationSmoothnessClassifierType> SegmentationPairwisePotentialType;
    typedef PairwisePotentialSegmentationMarcel<ImageType> SegmentationPairwisePotentialType;
    
    // //reg
    //typedef FastUnaryPotentialRegistrationSAD< LabelMapperType, ImageType > RegistrationUnaryPotentialType;
    typedef FastUnaryPotentialRegistrationNCC< LabelMapperType, ImageType > RegistrationUnaryPotentialType;
    // //typedef FastUnaryPotentialRegistrationSSD< LabelMapperType, ImageType > RegistrationUnaryPotentialType;
    // //typedef UnaryPotentialRegistrationNCCWithBonePrior< LabelMapperType, ImageType > RegistrationUnaryPotentialType;
    // //typedef UnaryPotentialRegistrationNCCWithDistanceBonePrior< LabelMapperType, ImageType > RegistrationUnaryPotentialType;
    
    typedef PairwisePotentialRegistration< LabelMapperType, ImageType > RegistrationPairwisePotentialType;
    
    typedef PairwisePotentialCoherence< ImageType > CoherencePairwisePotentialType;
    // //typedef PairwisePotentialSigmoidCoherence< ImageType > CoherencePairwisePotentialType;
    // //typedef PairwisePotentialCoherenceBinary< ImageType > CoherencePairwisePotentialType;
    // //typedef PairwisePotentialBoneCoherence<  ImageType > CoherencePairwisePotentialType;
    // //typedef FastRegistrationGraphModel<
    // //    typedef SortedSubsamplingGraphModel<
    typedef FastGraphModel<
        ImageType,
        RegistrationUnaryPotentialType,
        RegistrationPairwisePotentialType,
        SegmentationUnaryPotentialType,
        SegmentationPairwisePotentialType,
        CoherencePairwisePotentialType,
        LabelMapperType>        GraphType;
    
    typedef HierarchicalSRSImageToImageFilter<GraphType>        FilterType;    
    //create filter
    FilterType::Pointer filter=FilterType::New();
    filter->setConfig(&filterConfig);
    logSetStage("IO");
    logSetVerbosity(filterConfig.verbose);
    LOG<<"Loading target image :"<<filterConfig.targetFilename<<std::endl;
    ImagePointerType targetImage=ImageUtils<ImageType>::readImage(filterConfig.targetFilename);
    if (!targetImage) {LOG<<"failed!"<<endl; exit(0);}
    LOG<<"Loading atlas image :"<<filterConfig.atlasFilename<<std::endl;
    ImagePointerType atlasImage;
    if (filterConfig.atlasFilename!="") atlasImage=ImageUtils<ImageType>::readImage(filterConfig.atlasFilename);
    if (!atlasImage) {LOG<<"Warning: no atlas image loaded!"<<endl;
        LOG<<"Loading atlas segmentation image :"<<filterConfig.atlasSegmentationFilename<<std::endl;}
    ImagePointerType atlasSegmentation;
    if (filterConfig.atlasSegmentationFilename !="")atlasSegmentation=ImageUtils<ImageType>::readImage(filterConfig.atlasSegmentationFilename);
    if (!atlasSegmentation) {LOG<<"Warning: no atlas segmentation loaded!"<<endl; }
    logResetStage;
    logSetStage("Preprocessing");
    //preprocessing 1: gradients
    ImagePointerType targetGradient, atlasGradient;
    ImagePointerType tissuePrior;
    if (filterConfig.segment){
        if (filterConfig.targetGradientFilename!=""){
            targetGradient=(ImageUtils<ImageType>::readImage(filterConfig.targetGradientFilename));
        }else{
            targetGradient=Preprocessing<ImageType>::computeSheetness(targetImage);
            LOGI(8,ImageUtils<ImageType>::writeImage("targetsheetness.nii",targetGradient));
           
        }
        if (filterConfig.atlasGradientFilename!=""){
            atlasGradient=(ImageUtils<ImageType>::readImage(filterConfig.atlasGradientFilename));
        }else{
            if (atlasImage.IsNotNull()){
                atlasGradient=Preprocessing<ImageType>::computeSheetness(atlasImage);
                LOGI(8,ImageUtils<ImageType>::writeImage("atlassheetness.nii",atlasGradient));
            }
        }
  
        if (filterConfig.useTissuePrior){
            tissuePrior=Preprocessing<ImageType>::computeSoftTissueEstimate(targetImage);
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
        if (atlasSegmentation.IsNotNull()) {
            atlasSegmentation=FilterUtils<ImageType>::NNResample((atlasSegmentation),scale,false);
            //ImageUtils<ImageType>::writeImage("testA.nii",atlasSegmentation);
        }
        if (filterConfig.segment){
            targetGradient=FilterUtils<ImageType>::LinearResample(((ImageConstPointerType)targetGradient),scale,true);
            atlasGradient=FilterUtils<ImageType>::LinearResample(((ImageConstPointerType)atlasGradient),scale,true);
            //targetGradient=FilterUtils<ImageType>::NNResample(FilterUtils<ImageType>::gaussian((ImageConstPointerType)targetGradient,sigma),scale);
            //atlasGradient=FilterUtils<ImageType>::NNResample(FilterUtils<ImageType>::gaussian((ImageConstPointerType)atlasGradient,sigma),scale);
            if (filterConfig.useTissuePrior){
                tissuePrior=FilterUtils<ImageType>::LinearResample((targetImage),scale,true);
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
    filter->setAtlasGradient(atlasGradient);
    filter->setAtlasSegmentation(atlasSegmentation);
    if (filterConfig.useTissuePrior){
        filter->setTissuePrior(tissuePrior);
    }
    logSetStage("Bulk transforms");

    if (filterConfig.affineBulkTransform!=""){
        TransfUtils<ImageType>::AffineTransformPointerType affine=TransfUtils<ImageType>::readAffine(filterConfig.affineBulkTransform);
        ImageUtils<ImageType>::writeImage("def.nii",TransfUtils<ImageType>::affineDeformImage(originalAtlasImage,affine,originalTargetImage));
        DeformationFieldPointerType transf=TransfUtils<ImageType>::affineToDisplacementField(affine,originalTargetImage);
        ImageUtils<ImageType>::writeImage("def2.nii",TransfUtils<ImageType>::warpImage((ImageType::ConstPointer)originalAtlasImage,transf));
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

    //upsample?
    if (filterConfig.downScale<1){
        LOG<<"Upsampling Images.."<<endl;
        if (finalDeformation.IsNotNull() ) finalDeformation=TransfUtils<ImageType>::bSplineInterpolateDeformationField(finalDeformation,(ImageConstPointerType)originalTargetImage);
        //this is more or less f***** up
        //it would probably be far better to create a surface for each label, 'upsample' that surface, and then create a binary volume for each surface which are merged in a last step
        if (targetSegmentationEstimate){
            typedef ImageUtils<ImageType>::FloatImageType FloatImageType;
            typedef ImageUtils<ImageType>::FloatImagePointerType FloatImagePointerType;
            LOGI(6,ImageUtils<ImageType>::writeImage("targetSegmentationEstimateLow.nii",targetSegmentationEstimate));
            FloatImagePointerType distanceMap=FilterUtils<ImageType,FloatImageType>::distanceMapByFastMarcher(FilterUtils<ImageType>::binaryThresholdingLow(targetSegmentationEstimate,1),1);
            LOGI(6,ImageUtils<FloatImageType>::writeImage("distnaceMapLow.nii",distanceMap));
            distanceMap=FilterUtils<FloatImageType>::LinearResample(distanceMap,FilterUtils<ImageType,FloatImageType>::cast(originalTargetImage),false);
            LOGI(6,ImageUtils<FloatImageType>::writeImage("distnaceMaphigh.nii",distanceMap));
            targetSegmentationEstimate=FilterUtils<FloatImageType,ImageType>::binaryThresholdingHigh(distanceMap,0.5);
            //targetSegmentationEstimate=FilterUtils<ImageType>::round(FilterUtils<ImageType>::NNResample((targetSegmentationEstimate),scale));
        }
    }

    if (finalDeformation.IsNotNull() ){
        LOG<<"Deforming Images.."<<endl;
        if (filterConfig.defFilename!="")
            ImageUtils<DeformationFieldType>::writeImage(filterConfig.defFilename,finalDeformation);

        ImagePointerType deformedAtlasSegmentation=TransfUtils<ImageType>::warpImage((ImageConstPointerType)originalAtlasSegmentation,finalDeformation,true);
        ImagePointerType deformedAtlasImage=TransfUtils<ImageType>::warpImage((ImageConstPointerType)originalAtlasImage,finalDeformation);
        ImageUtils<ImageType>::writeImage(filterConfig.outputDeformedFilename,deformedAtlasImage);
        ImageUtils<ImageType>::writeImage(filterConfig.outputDeformedSegmentationFilename,deformedAtlasSegmentation);
        LOG<<"Final SAD: "<<ImageUtils<ImageType>::sumAbsDist((ImageConstPointerType)deformedAtlasImage,(ImageConstPointerType)targetImage)<<endl;

    }
    
   
    if (targetSegmentationEstimate.IsNotNull()){
        ImageUtils<ImageType>::writeImage(filterConfig.segmentationOutputFilename,targetSegmentationEstimate);
    }
    
    OUTPUTTIMER;
    if (filterConfig.logFileName!=""){
        mylog.flushLog(filterConfig.logFileName);
    }
  
    return 1;
}
