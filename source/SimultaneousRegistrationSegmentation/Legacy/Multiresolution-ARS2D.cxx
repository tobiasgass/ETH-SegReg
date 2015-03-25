#include <stdio.h>
#include <iostream>

#include "ArgumentParser.h"

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
#include "SegmentationTools.hxx"
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
    typedef unsigned char PixelType;
	const unsigned int D=2;
	typedef Image<PixelType,D> ImageType;
    typedef ImageType::Pointer ImagePointerType;
    typedef ImageType::ConstPointer ImageConstPointerType;
	typedef TransfUtils<ImageType>::DisplacementType DisplacementType;
    //typedef SparseRegistrationLabelMapper<ImageType,DisplacementType> LabelMapperType;
    typedef DenseRegistrationLabelMapper<ImageType,DisplacementType> LabelMapperType;

    typedef TransfUtils<ImageType>::DeformationFieldType DeformationFieldType;
    typedef DeformationFieldType::Pointer DeformationFieldPointerType;



    //unary seg
    //typedef SegmentationClassifierGradient<ImageType> ClassifierType;
    //    typedef SegmentationGenerativeClassifierGradient<ImageType> ClassifierType;
    //typedef     SegmentationGaussianClassifierGradient<ImageType> ClassifierType;
    //typedef SegmentationClassifier<ImageType> ClassifierType;
    //typedef UnaryPotentialSegmentationClassifier< ImageType, ClassifierType > SegmentationUnaryPotentialType;
    //typedef UnaryPotentialSegmentationBoneMarcel< ImageType > SegmentationUnaryPotentialType;
    //typedef     UnaryPotentialSegmentation< ImageType > SegmentationUnaryPotentialType;
    
    //typedef SegmentationRandomForestClassifier<ImageType> ClassifierType;
    typedef SegmentationGMMClassifier<ImageType> ClassifierType;
    typedef UnaryPotentialNewSegmentationClassifier< ImageType, ClassifierType > SegmentationUnaryPotentialType;

    //pairwise seg
    //    typedef UnaryPotentialSegmentation< ImageType > SegmentationUnaryPotentialType;
    typedef SmoothnessClassifierGradient<ImageType> SegmentationSmoothnessClassifierType;
    //typedef SmoothnessClassifierGradientContrast<ImageType> SegmentationSmoothnessClassifierType;
    //typedef SmoothnessClassifierFullMultilabelPosterior<ImageType> SegmentationSmoothnessClassifierType;
    typedef CachingPairwisePotentialSegmentationClassifier<ImageType,SegmentationSmoothnessClassifierType> SegmentationPairwisePotentialType;
    //typedef PairwisePotentialSegmentationMarcel<ImageType> SegmentationPairwisePotentialType;
    
    //reg
    //typedef UnaryPotentialRegistrationSAD< LabelMapperType, ImageType > RegistrationUnaryPotentialType;
    typedef FastUnaryPotentialRegistrationNCC< LabelMapperType, ImageType > RegistrationUnaryPotentialType;
    //typedef FastUnaryPotentialRegistrationSAD< LabelMapperType, ImageType > RegistrationUnaryPotentialType;
    //typedef FastUnaryPotentialRegistrationNMI< LabelMapperType, ImageType > RegistrationUnaryPotentialType;
    //typedef UnaryPotentialRegistrationNCC< LabelMapperType, ImageType > RegistrationUnaryPotentialType;
    //typedef UnaryPotentialRegistrationNCCWithBonePrior< LabelMapperType, ImageType > RegistrationUnaryPotentialType;
    //typedef UnaryPotentialRegistrationNCCWithDistanceBonePrior< LabelMapperType, ImageType > RegistrationUnaryPotentialType;
    typedef PairwisePotentialRegistration< LabelMapperType, ImageType > RegistrationPairwisePotentialType;
    typedef PairwisePotentialCoherence< ImageType > CoherencePairwisePotentialType;
    //typedef PairwisePotentialCoherenceBinary< ImageType > CoherencePairwisePotentialType;
    //typedef PairwisePotentialBoneCoherence<  ImageType > CoherencePairwisePotentialType;
    //typedef FastRegistrationGraphModel<
    //    typedef SortedSubsamplingGraphModel<
    typedef FastGraphModel<
        //typedef GraphModel<
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
    logSetVerbosity(filterConfig.verbose>0?filterConfig.verbose:-1);
    LOG<<"Loading target image :"<<filterConfig.targetFilename<<std::endl;
    ImagePointerType targetImage=ImageUtils<ImageType>::readImage(filterConfig.targetFilename);
    if (!targetImage) {LOG<<"failed!"<<endl; exit(0);}
    if (filterConfig.ROIFilename  != ""){
        ImagePointerType roi=ImageUtils<ImageType>::readImage(filterConfig.ROIFilename);
        targetImage=FilterUtils<ImageType>::NNResample(targetImage,roi,false);
    }
    LOG<<"Loading atlas image :"<<filterConfig.atlasFilename<<std::endl;
    ImagePointerType atlasImage=ImageUtils<ImageType>::readImage(filterConfig.atlasFilename);
    if (!atlasImage) {LOG<<"failed!"<<endl; exit(0);
        LOG<<"Loading atlas segmentation image :"<<filterConfig.atlasSegmentationFilename<<std::endl;}
    ImagePointerType atlasSegmentation=ImageUtils<ImageType>::readImage(filterConfig.atlasSegmentationFilename);
    atlasSegmentation=filter->fixSegmentationImage(atlasSegmentation,filterConfig.nSegmentations);

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
            //ImageUtils<ImageType>::writeImage("targetsheetness.nii",targetGradient);
        }
        if (filterConfig.atlasGradientFilename!=""){
            atlasGradient=(ImageUtils<ImageType>::readImage(filterConfig.atlasGradientFilename));
        }else{
            atlasGradient=atlasImage;
            //ImageUtils<ImageType>::writeImage("atlassheetness.nii",atlasGradient);
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
        targetImage=FilterUtils<ImageType>::LinearResample(((ImageConstPointerType)targetImage),scale,true);
        atlasImage=FilterUtils<ImageType>::LinearResample(((ImageConstPointerType)atlasImage),scale,true);
        atlasSegmentation=FilterUtils<ImageType>::NNResample((atlasSegmentation),scale,false);
        if (filterConfig.segment){
            targetGradient=FilterUtils<ImageType>::LinearResample(((ImageConstPointerType)targetGradient),scale,true);
            atlasGradient=FilterUtils<ImageType>::LinearResample(((ImageConstPointerType)atlasGradient),scale,true);
            //targetGradient=FilterUtils<ImageType>::NNResample(FilterUtils<ImageType>::gaussian((ImageConstPointerType)targetGradient,sigma),scale);
            //atlasGradient=FilterUtils<ImageType>::NNResample(FilterUtils<ImageType>::gaussian((ImageConstPointerType)atlasGradient,sigma),scale);
            if (filterConfig.useTissuePrior){
                tissuePrior=FilterUtils<ImageType>::LinearResample(((ImageConstPointerType)(targetImage)),scale,true);
            }
        }
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
    DeformationFieldPointerType transf=NULL;
    if (filterConfig.affineBulkTransform!=""){
        TransfUtils<ImageType>::AffineTransformPointerType affine=TransfUtils<ImageType>::readAffine(filterConfig.affineBulkTransform);
        ImageUtils<ImageType>::writeImage("def.nii",TransfUtils<ImageType>::affineDeformImage(originalAtlasImage,affine,originalTargetImage));
        transf=TransfUtils<ImageType>::affineToDisplacementField(affine,originalTargetImage);
        ImageUtils<ImageType>::writeImage("def2.nii",TransfUtils<ImageType>::warpImage((ImageType::ConstPointer)originalAtlasImage,transf));
        filter->setBulkTransform(transf);
    }
    else if (filterConfig.bulkTransformationField!=""){
        transf=(ImageUtils<DeformationFieldType>::readImage(filterConfig.bulkTransformationField));
    }else if (filterConfig.centerImages){
        LOG<<"Computing transform to move image centers on top of each other.."<<std::endl;
         transf=TransfUtils<ImageType>::computeCenteringTransform(originalTargetImage,originalAtlasImage);
      
    }
    
    // compute SRS
    clock_t FULLstart = clock();
    filter->Init();
    double tmpSegU,tmpSegP,tmpCoh;
    int tmpSegL,tmpRegL;
    //store original parameters
    tmpSegU=filterConfig.unarySegmentationWeight;
    tmpSegP=filterConfig.pairwiseSegmentationWeight;
    tmpCoh=filterConfig.pairwiseCoherenceWeight;
    tmpRegL=filterConfig.maxDisplacement;
    tmpSegL=filterConfig.nSegmentations;
    double coherenceMultiplier=filterConfig.coherenceMultiplier;
    filterConfig.coherenceMultiplier=1.0;
    DeformationFieldPointerType intermediateDeformation;
    ImagePointerType intermediateSegmentation;
    double lastSegEnergy=10000;
    double lastRegEnergy=10000;
    logResetStage;
    logSetStage("ARS iteration");
    for (int iteration=0;iteration<10;++iteration){    
        filterConfig.ARSTolerance= pow(filterConfig.toleranceBase,max(0,4-iteration-1)+1);
        filter->setBulkTransform(transf);
        if (iteration == 0){
            //start with pure registration by setting seg and coh weights to zero
            filterConfig.unarySegmentationWeight=0;
            filterConfig.pairwiseSegmentationWeight=0;
            filterConfig.pairwiseCoherenceWeight=0;
        }else{
            filter->setTargetSegmentation(intermediateSegmentation);
        }
        filterConfig.nSegmentations=1;
        filter->Update();
        double regEnergy=filter->getEnergy();
        intermediateDeformation=filter->getFinalDeformation();

        //transf=intermediateDeformation;

        filterConfig.nSegmentations=tmpSegL;
        //reset weights
        filterConfig.unarySegmentationWeight= tmpSegU;
        filterConfig.pairwiseSegmentationWeight=tmpSegP;
        filterConfig.pairwiseCoherenceWeight=tmpCoh*pow(coherenceMultiplier,iteration);

        filter->setTargetSegmentation(NULL);
        filter->setBulkTransform(intermediateDeformation);
        filterConfig.maxDisplacement=0;
        filter->Update();
        double segEnergy=filter->getEnergy();
        intermediateSegmentation=filter->getTargetSegmentationEstimate();
        filterConfig.maxDisplacement= tmpRegL;
        LOG<<" Iteration :"<<iteration<<" "<<VAR(regEnergy)<<" "<<VAR(segEnergy)<<endl;      
        if (filterConfig.groundTruthSegmentationFilename!=""){
            ImagePointerType groundTruth=ImageUtils<ImageType>::readImage(filterConfig.groundTruthSegmentationFilename);
            groundTruth=filter->fixSegmentationImage(groundTruth,filterConfig.nSegmentations);
            int maxLabel=FilterUtils<ImageType>::getMax(groundTruth);
            double dice,hd,msd;
            SegmentationTools<ImageType>::computeOverlap(groundTruth, intermediateSegmentation, dice,msd,hd,maxLabel,true);
            LOG<<"Iteration :"<<iteration<<" "<<VAR(dice)<<" "<<VAR(msd)<<" "<<VAR(hd)<<endl;

        }

    
        bool converged=false;
        if (iteration >0 && fabs(lastSegEnergy-segEnergy)/lastSegEnergy< 1e-4 && fabs (lastRegEnergy-regEnergy)/lastRegEnergy < 1e-4){
            converged=true;
        }
        lastSegEnergy=segEnergy;
        lastRegEnergy=regEnergy;
        if (filterConfig.verbose>=5){
            //store intermediate results
            std::string suff;
            if (ImageType::ImageDimension==2){
                suff=".nii";
            }
            if (ImageType::ImageDimension==3){
                suff=".nii";
            }
            ostringstream deformedSegmentationFilename;
            deformedSegmentationFilename<<filterConfig.outputDeformedSegmentationFilename<<"-arsIter"<<iteration<<suff;
            ostringstream tmpSegmentationFilename;
            tmpSegmentationFilename<<filterConfig.segmentationOutputFilename<<"-arsIter"<<iteration<<suff;
            ImagePointerType deformedAtlasSegmentation=TransfUtils<ImageType>::warpImage(originalAtlasSegmentation,intermediateDeformation,true);
            ImagePointerType tmpSeg=intermediateSegmentation;
            if (ImageType::ImageDimension==2){
                tmpSeg=filter->makePngFromLabelImage(tmpSeg, tmpSegL);
                deformedAtlasSegmentation=filter->makePngFromLabelImage(deformedAtlasSegmentation,tmpSegL);
            }
            ImageUtils<ImageType>::writeImage(deformedSegmentationFilename.str().c_str(),deformedAtlasSegmentation);
            ImageUtils<ImageType>::writeImage(tmpSegmentationFilename.str().c_str(),tmpSeg);

        }
        if (converged) break;
    }
    logResetStage;
    logSetStage("Finalizing");
    clock_t FULLend = clock();
    float t = (float) ((double)(FULLend - FULLstart) / CLOCKS_PER_SEC);
    LOG<<"Finished computation after "<<t<<" seconds"<<std::endl;
    LOG<<"RegUnaries: "<<tUnary<<" Optimization: "<<tOpt<<std::endl;	
    LOG<<"RegPairwise: "<<tPairwise<<std::endl;

    
    //process outputs
    ImagePointerType targetSegmentationEstimate=filter->getTargetSegmentationEstimate();
    DeformationFieldPointerType finalDeformation=filter->getFinalDeformation();
    
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

    if (filterConfig.defFilename!=""){
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
    return 1;
}
