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
#include "TransformationUtils.h"
#include "NewClassifier.h"
#include <google/heap-profiler.h>

using namespace std;
using namespace itk;

int main(int argc, char ** argv)
{
    //HeapProfilerStart("SRS2D");
	//feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
	SRSConfig filterConfig;
	filterConfig.parseParams(argc,argv);
    if (filterConfig.logFileName!=""){
        mylog.setCachedLogging();
    }
    logSetStage("Init");
	//define types.
    typedef unsigned char PixelType;
	const   unsigned int D=2;
	typedef Image<PixelType,D> ImageType;
    typedef ImageType::Pointer ImagePointerType;
    typedef ImageType::ConstPointer ImageConstPointerType;
	typedef TransfUtils<ImageType>::DisplacementType DisplacementType;
    //typedef SparseRegistrationLabelMapper<ImageType,DisplacementType> LabelMapperType;
    //typedef SemiSparseRegistrationLabelMapper<ImageType,DisplacementType> LabelMapperType;
    typedef DenseRegistrationLabelMapper<ImageType,DisplacementType> LabelMapperType;
    typedef TransfUtils<ImageType>::DeformationFieldType DeformationFieldType;
    typedef DeformationFieldType::Pointer DeformationFieldPointerType;



    //unary segmentation potential choices

    //typedef SegmentationClassifierGradient<ImageType> ClassifierType;
    //typedef SegmentationGenerativeClassifierGradient<ImageType> ClassifierType;
    //typedef     SegmentationGaussianClassifierGradient<ImageType> ClassifierType;
    //typedef SegmentationClassifier<ImageType> ClassifierType;
    //typedef UnaryPotentialSegmentationClassifier< ImageType, ClassifierType > SegmentationUnaryPotentialType;
    //typedef UnaryPotentialSegmentationBoneMarcel< ImageType > SegmentationUnaryPotentialType;
    //typedef     UnaryPotentialSegmentation< ImageType > SegmentationUnaryPotentialType;
    //typedef SegmentationRandomForestClassifier<ImageType> ClassifierType;
    
    typedef SegmentationGMMClassifier<ImageType> ClassifierType;
    typedef UnaryPotentialNewSegmentationClassifier< ImageType, ClassifierType > SegmentationUnaryPotentialType;


    //pairwise segmentation potential choices

    //    typedef UnaryPotentialSegmentation< ImageType > SegmentationUnaryPotentialType;
    typedef SmoothnessClassifierGradient<ImageType> SegmentationSmoothnessClassifierType;
    //typedef SmoothnessClassifierGradientContrast<ImageType> SegmentationSmoothnessClassifierType;
    //typedef SmoothnessClassifierFullMultilabelPosterior<ImageType> SegmentationSmoothnessClassifierType;
    typedef CachingPairwisePotentialSegmentationClassifier<ImageType,SegmentationSmoothnessClassifierType> SegmentationPairwisePotentialType;
    //typedef PairwisePotentialSegmentationMarcel<ImageType> SegmentationPairwisePotentialType;
    
    //registration unary potential
    //typedef UnaryPotentialRegistrationSAD< LabelMapperType, ImageType > RegistrationUnaryPotentialType;
    typedef FastUnaryPotentialRegistrationNCC< LabelMapperType, ImageType > RegistrationUnaryPotentialType;
    //typedef FastUnaryPotentialRegistrationSAD< LabelMapperType, ImageType > RegistrationUnaryPotentialType;
    //typedef FastUnaryPotentialRegistrationSSD< LabelMapperType, ImageType > RegistrationUnaryPotentialType;
    //typedef FastUnaryPotentialRegistrationNMI< LabelMapperType, ImageType > RegistrationUnaryPotentialType;
    //typedef UnaryPotentialRegistrationNCC< LabelMapperType, ImageType > RegistrationUnaryPotentialType;
    //typedef UnaryPotentialRegistrationNCCWithBonePrior< LabelMapperType, ImageType > RegistrationUnaryPotentialType;
    //typedef UnaryPotentialRegistrationNCCWithDistanceBonePrior< LabelMapperType, ImageType > RegistrationUnaryPotentialType;

    //Registration pairwise potential
    typedef PairwisePotentialRegistration< LabelMapperType, ImageType > RegistrationPairwisePotentialType;
    //Coherence potential
    typedef PairwisePotentialCoherence< ImageType > CoherencePairwisePotentialType;
    //typedef PairwisePotentialSigmoidCoherence< ImageType > CoherencePairwisePotentialType;
    //typedef PairwisePotentialCoherenceBinary< ImageType > CoherencePairwisePotentialType;
    //typedef PairwisePotentialBoneCoherence<  ImageType > CoherencePairwisePotentialType;

    //Graph
    typedef FastGraphModel<
        //typedef GraphModel<
        ImageType,
        RegistrationUnaryPotentialType,
        RegistrationPairwisePotentialType,
        SegmentationUnaryPotentialType,
        SegmentationPairwisePotentialType,
        CoherencePairwisePotentialType,
        LabelMapperType>        GraphType;
    
    //Final filter
	typedef HierarchicalSRSImageToImageFilter<GraphType>        FilterType;    
	//create filter
    FilterType::Pointer filter=FilterType::New();
    filter->setConfig(&filterConfig);
    logSetStage("IO");
    logSetVerbosity(filterConfig.verbose);
    LOG<<"Loading target image :"<<filterConfig.targetFilename<<std::endl;
    ImagePointerType targetImage=ImageUtils<ImageType>::readImage(filterConfig.targetFilename);
    if (filterConfig.ROIFilename  != ""){
        ImagePointerType roi=ImageUtils<ImageType>::readImage(filterConfig.ROIFilename);
        targetImage=FilterUtils<ImageType>::NNResample(targetImage,roi,false);
    }
    if (!targetImage) {LOG<<"failed!"<<endl; exit(0);}
    LOG<<"Loading atlas image :"<<filterConfig.atlasFilename<<std::endl;
    ImagePointerType atlasImage=ImageUtils<ImageType>::readImage(filterConfig.atlasFilename);
    if (!atlasImage) {LOG<<"failed!"<<endl; exit(0);
        LOG<<"Loading atlas segmentation image :"<<filterConfig.atlasSegmentationFilename<<std::endl;}
    ImagePointerType atlasSegmentation=ImageUtils<ImageType>::readImage(filterConfig.atlasSegmentationFilename);
    atlasSegmentation=filter->fixSegmentationImage(atlasSegmentation,filterConfig.nSegmentations);
    logResetStage;//IO
    if (!atlasSegmentation) {LOG<<"failed!"<<endl; exit(0);}
    logSetStage("Preprocessing");
    //preprocessing 1: gradients
    ImagePointerType targetGradient, atlasGradient;
    ImagePointerType tissuePrior;
    if (filterConfig.segment){
        if (filterConfig.targetGradientFilename!=""){
            targetGradient=(ImageUtils<ImageType>::readImage(filterConfig.targetGradientFilename));
        }else{
            targetGradient=targetImage;
            //ImageUtils<ImageType>::writeImage("targetsheetness.png",targetGradient);
        }
        if (filterConfig.atlasGradientFilename!=""){
            atlasGradient=(ImageUtils<ImageType>::readImage(filterConfig.atlasGradientFilename));
        }else{
            atlasGradient=atlasImage;
            //ImageUtils<ImageType>::writeImage("atlassheetness.png",atlasGradient);
        }
        
        if (filterConfig.useTissuePrior){
            filterConfig.useTissuePrior=false;
            LOG<<"Tissue prior not implemented for 2D, setting to false"<<endl;
            //tissuePrior=Preprocessing<ImageType>::computeSoftTissueEstimate(targetImage);
        }
        //preprocessing 2: multilabel
        if (filterConfig.computeMultilabelAtlasSegmentation){
            atlasSegmentation=FilterUtils<ImageType>::computeMultilabelSegmentation(atlasSegmentation);
            filterConfig.nSegmentations=5;//TODO!!!!
        }
    }
  
    ImagePointerType originalTargetImage=targetImage,originalAtlasImage=atlasImage,originalAtlasSegmentation=atlasSegmentation;
    //preprocessing 3: downscaling
    if (filterConfig.downScale<1){
        double sigma=1;
        double scale=filterConfig.downScale;
        LOG<<"Resampling images from "<< targetImage->GetLargestPossibleRegion().GetSize()<<" by a factor of"<<scale<<endl;
        //targetImage=FilterUtils<ImageType>::LinearResample(FilterUtils<ImageType>::gaussian((ImageConstPointerType)targetImage,sigma),scale, false);
        targetImage=FilterUtils<ImageType>::LinearResample(targetImage,scale, true);
        atlasImage=FilterUtils<ImageType>::LinearResample(atlasImage,scale,true);

        atlasSegmentation=FilterUtils<ImageType>::NNResample((atlasSegmentation),scale,false);
        if (filterConfig.segment){
            targetGradient=FilterUtils<ImageType>::LinearResample((targetGradient),scale,true);
            atlasGradient=FilterUtils<ImageType>::LinearResample((atlasGradient),scale,true);
            //targetGradient=FilterUtils<ImageType>::NNResample(FilterUtils<ImageType>::gaussian((ImageConstPointerType)targetGradient,sigma),scale);
            //atlasGradient=FilterUtils<ImageType>::NNResample(FilterUtils<ImageType>::gaussian((ImageConstPointerType)atlasGradient,sigma),scale);
            if (filterConfig.useTissuePrior){
                tissuePrior=FilterUtils<ImageType>::LinearResample(targetImage,scale,true);
            }
        }
    }
    logResetStage;//Preprocessing
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
        ImageUtils<ImageType>::writeImage("def.png",TransfUtils<ImageType>::affineDeformImage(originalAtlasImage,affine,originalTargetImage));
        DeformationFieldPointerType transf=TransfUtils<ImageType>::affineToDisplacementField(affine,originalTargetImage);
        ImageUtils<ImageType>::writeImage("def2.png",TransfUtils<ImageType>::warpImage((ImageType::ConstPointer)originalAtlasImage,transf));
        filter->setBulkTransform(transf);
    }
    else if (filterConfig.bulkTransformationField!=""){
        filter->setBulkTransform(ImageUtils<DeformationFieldType>::readImage(filterConfig.bulkTransformationField));
    }else if (filterConfig.centerImages){
        LOG<<"Computing transform to move image centers on top of each other.."<<std::endl;
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
    LOG<<"RegUnaries: "<<tUnary<<" Optimization: "<<tOpt<<std::endl;	
    LOG<<"RegPairwise: "<<tPairwise<<std::endl;

    
    //process outputs
    ImagePointerType targetSegmentationEstimate=filter->getTargetSegmentationEstimate();
    DeformationFieldPointerType finalDeformation=filter->getFinalDeformation();
    
    logSetStage("Finalizing");
    //upsample?
    if (filterConfig.downScale<1){
        LOG<<"Upsampling Images.."<<endl;
        finalDeformation=TransfUtils<ImageType>::bSplineInterpolateDeformationField(finalDeformation,(ImageConstPointerType)originalTargetImage);
        //this is more or less f***** up
        //it would probably be far better to create a surface for each label, 'upsample' that surface, and then create a binary volume for each surface which are merged in a last step
        if (targetSegmentationEstimate){
            typedef ImageUtils<ImageType>::FloatImageType FloatImageType;
            targetSegmentationEstimate=FilterUtils<ImageType>::round(FilterUtils<ImageType>::NNResample((targetSegmentationEstimate),originalTargetImage,false));
        }
    }
    LOG<<"Deforming Images.."<<endl;

    if (filterConfig.regist && filterConfig.defFilename!=""){
        ImageUtils<DeformationFieldType>::writeImage(filterConfig.defFilename,finalDeformation);
    }
    
    if (filterConfig.regist){
        ImagePointerType deformedAtlasSegmentation=TransfUtils<ImageType>::warpSegmentationImage(originalAtlasSegmentation,finalDeformation);
        ImagePointerType deformedAtlasImage=TransfUtils<ImageType>::warpImage(originalAtlasImage,finalDeformation);
        
        ImageUtils<ImageType>::writeImage(filterConfig.outputDeformedFilename,deformedAtlasImage);
        ImageUtils<ImageType>::writeImage(filterConfig.outputDeformedSegmentationFilename,filter->makePngFromLabelImage((ImageConstPointerType)(ImageConstPointerType)deformedAtlasSegmentation,LabelMapperType::nSegmentations));
        LOG<<"Final SAD: "<<ImageUtils<ImageType>::sumAbsDist((ImageConstPointerType)deformedAtlasImage,(ImageConstPointerType)targetImage)<<endl;
    }
    if (filterConfig.segment){
        ImageUtils<ImageType>::writeImage(filterConfig.segmentationOutputFilename,filter->makePngFromLabelImage((ImageConstPointerType)targetSegmentationEstimate,LabelMapperType::nSegmentations));
    }
    
    OUTPUTTIMER;
    if (filterConfig.logFileName!=""){
        mylog.flushLog(filterConfig.logFileName);
    }
    //HeapProfilerStop();

    return 1;
}
