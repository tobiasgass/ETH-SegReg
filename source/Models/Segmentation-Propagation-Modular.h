
#pragma once

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
#include "bgraph.h"
#include <sstream>
#include "argstream.h"
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include "itkConstNeighborhoodIterator.h"
using namespace std;

template <class ImageType, int nSegmentationLabels>
class SegmentationPropagationModular{
public:
    typedef typename ImageType::PixelType PixelType;
    static const unsigned int D=ImageType::ImageDimension;
    typedef typename  ImageType::Pointer ImagePointerType;
    typedef typename  ImageType::IndexType IndexType;
    typedef typename  ImageType::PointType PointType;
    typedef typename  ImageType::OffsetType OffsetType;
    typedef typename  ImageType::SizeType SizeType;
    typedef typename  ImageType::ConstPointer ImageConstPointerType;
    typedef typename  ImageUtils<ImageType>::FloatImageType FloatImageType;
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

    typedef itk::Vector<float,nSegmentationLabels> ProbabilisticPixelType;
    typedef itk::Image<ProbabilisticPixelType,D> ProbabilisticVectorImageType;
    typedef typename ProbabilisticVectorImageType::Pointer ProbabilisticVectorImagePointerType;
    typedef typename itk::ImageRegionIterator<ProbabilisticVectorImageType> ProbImageIteratorType;

    enum MetricType {NONE,SAD,NCC,MI,NMI,SSD};
    enum WeightingType {UNIFORM,GLOBAL,LOCAL};
public:
    int run(int argc, char ** argv)
    {
        feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
        argstream * as=new argstream(argc,argv);
        string deformationFileList,imageFileList,atlasSegmentationFileList,supportSamplesListFileName="",outputDir=".",outputSuffix="",weightListFilename="";
        int verbose=0;
        double pWeight=1.0;
        double sigma=30;
        int radius=3;
        int maxHops=1;
        bool uniformUpdate=true;
        string metricName="NCC";
        string weightingName="uniform";
        (*as) >> parameter ("A",atlasSegmentationFileList , "list of atlas segmentations <id> <file>", true);
        (*as) >> parameter ("T", deformationFileList, " list of deformations", true);
        (*as) >> parameter ("i", imageFileList, " list of  images", true);
        (*as) >> parameter ("W", weightListFilename,"list of weights for deformations",false);
        (*as) >> parameter ("metric", metricName,"metric to be used for global or local weighting, valid: NONE,SAD,SSD,NCC,MI,NMI",false);
        (*as) >> parameter ("weighting", weightingName,"internal weighting scheme {uniform,local,global}. non-uniform will only work with metric != NONE",false);
        (*as) >> parameter ("s", sigma,"sigma",false);
        (*as) >> parameter ("O", outputDir,"outputdirectory (will be created + no overwrite checks!)",false);
        (*as) >> parameter ("radius", radius,"patch radius for NCC",false);
        (*as) >> parameter ("maxHops", maxHops,"maximum number of hops",false);
        (*as) >> parameter ("verbose", verbose,"get verbose output",false);
        (*as) >> help();
        as->defaultErrorHandling();
        string suffix;
        if (D==2)
            suffix=".png";
        else
            suffix=".nii";

        mkdir(outputDir.c_str(),0755);
        logSetStage("IO");
        logSetVerbosity(verbose);
        
        MetricType metric;
        if (metricName=="none")
            metric=NONE;
        else if (metricName=="SSD")
            metric=SSD;
        else if (metricName=="SAD")
            metric=SAD;
        else if (metricName=="NCC")
            metric=NCC;
        else if (metricName=="NMI")
            metric=NMI;
        else if (metricName=="MI")
            metric=MI;
        else{
            LOG<<"don't understand "<<metricName<<", defaulting to NONE"<<endl;
            metric=NONE;
        }
        WeightingType weighting;
        if (weightingName=="uniform"){
            weighting=UNIFORM;}
        else if (weightingName=="global")
            weighting=GLOBAL;
        else if (weightingName=="local"){
            weighting=LOCAL;
            if (metric==NMI || metric == NCC || metric == MI ){
                LOG<<VAR(metric)<<" incompatibel with local weighing, aborting"<<endl;
                exit(0);
            }
        }
        else{
            LOG<<"don't understand "<<VAR(weightingName)<<", defaulting to uniform."<<endl;
            weighting=UNIFORM;
        }

        LOGV(2)<<VAR(metric)<<endl;
        map<string,ImagePointerType> *inputImages,*inputAtlasSegmentations;
        typedef typename map<string, ImagePointerType>::iterator ImageListIteratorType;
        LOG<<"Reading atlas segmentations."<<endl;
        inputAtlasSegmentations = readImageList( atlasSegmentationFileList );
        int nAtlases = inputAtlasSegmentations->size();
        LOG<<"Reading input images."<<endl;
        inputImages = readImageList( imageFileList );
        int nImages = inputImages->size();
     

        LOG<<"Reading deformations."<<endl;
        map< string, map <string, DeformationFieldPointerType> > deformations;
        map<string, map<string, float> > globalWeights;
        {
            ifstream ifs(deformationFileList.c_str());
            while (!ifs.eof()){
                string id1,id2,defFileName;
                ifs >> id1;
                if (id1!=""){
                    ifs >> id2;
                    ifs >> defFileName;
                    if (inputImages->find(id1)==inputImages->end() || inputImages->find(id2)==inputImages->end() ){
                        LOG<<id1<<" or "<<id2<<" not in image database, skipping"<<endl;
                        //exit(0);
                    }else{
                        LOGV(3)<<"Reading deformation "<<defFileName<<" for deforming "<<id1<<" to "<<id2<<endl;
                        DeformationFieldPointerType def=ImageUtils<DeformationFieldType>::readImage(defFileName);
                        deformations[id1][id2]=def;
                        globalWeights[id1][id2]=1.0;
                    }
                }
            }
        }
        if (weightListFilename!=""){
            ifstream ifs(weightListFilename.c_str());
            while (!ifs.eof()){
                string id1,id2;
                ifs >> id1;
                ifs >> id2;
                if (inputImages->find(id1)==inputImages->end() || inputImages->find(id2)==inputImages->end() ){
                    LOG<<id1<<" or "<<id2<<" not in image database while reading weights, skipping"<<endl;
                }else{
                    ifs>> globalWeights[id1][id2];
                }
            }
        }
    
        logSetStage("Zero Hop");
        LOG<<"Computing"<<std::endl;

        map<string,ProbabilisticVectorImagePointerType> probabilisticTargetSegmentations;
        for (ImageListIteratorType targetImageIterator=inputImages->begin();targetImageIterator!=inputImages->end();++targetImageIterator){                //iterate over targets
            string id= targetImageIterator->first;
            if (inputAtlasSegmentations->find(id)==inputAtlasSegmentations->end()){ //do not calculate segmentation for atlas images
                
                probabilisticTargetSegmentations[id]=createEmptyProbImageFromImage( targetImageIterator->second);
                for (ImageListIteratorType atlasIterator=inputAtlasSegmentations->begin();atlasIterator!=inputAtlasSegmentations->end();++atlasIterator){//iterate over atlases
                    string atlasID=atlasIterator->first;
                    DeformationFieldPointerType deformation= deformations[atlasID][id];
                    ImagePointerType deformedAtlasSegmentation=TransfUtils<ImageType>::warpSegmentationImage(atlasIterator->second,deformation);
                    ProbabilisticVectorImagePointerType probDeformedAtlasSegmentation=segmentationToProbabilisticVector(deformedAtlasSegmentation);
                    double weight=globalWeights[atlasID][id];
                   
                    //update
                    if (weighting==UNIFORM){
                        updateProbabilisticSegmentationUniform(probabilisticTargetSegmentations[id],probDeformedAtlasSegmentation,weight);

                    }else{
                        ImagePointerType targetImage= targetImageIterator->second;
                        ImagePointerType atlasImage=(*inputImages)[atlasID];
                        if (weighting==GLOBAL){
                            updateProbabilisticSegmentationGlobal(probabilisticTargetSegmentations[id],probDeformedAtlasSegmentation,weight,targetImage,atlasImage,deformation);

                        }else if (weighting==LOCAL){
                            ImagePointerType deformedAtlas=TransfUtils<ImageType>::warpImage(atlasImage,deformation);
                            updateProbabilisticSegmentationLocal(probabilisticTargetSegmentations[id],probDeformedAtlasSegmentation,weight,targetImage,atlasImage,deformation);

                        }

                    }
                }
            }
            else{
                probabilisticTargetSegmentations[id]=segmentationToProbabilisticVector((*inputAtlasSegmentations)[id]);
            }
        }//finished zero-hop segmentation
        LOG<<"done"<<endl;

        LOGV(1)<<"Storing zero-hop segmentations."<<endl;
        for (ImageListIteratorType targetImageIterator=inputImages->begin();targetImageIterator!=inputImages->end();++targetImageIterator){
            string id= targetImageIterator->first;
            ImagePointerType outputImage=probSegmentationToSegmentationLinear(probabilisticTargetSegmentations[id]);
            ostringstream tmpSegmentationFilename;
            tmpSegmentationFilename<<outputDir<<"/segmentation-"<<id<<"-hop0"<<suffix;
            ImageUtils<ImageType>::writeImage(tmpSegmentationFilename.str().c_str(),outputImage);
        }


        bool converged=false;
        logSetStage("Iterative improvement");
        for (int n=1;n<=maxHops && !converged;++n){
            LOG<<"hop "<<n<<endl;
            map<string,ProbabilisticVectorImagePointerType> newProbabilisticTargetSegmentations;
            for (ImageListIteratorType targetImageIterator=inputImages->begin();targetImageIterator!=inputImages->end();++targetImageIterator){
                string id1= targetImageIterator->first;
                if ( inputAtlasSegmentations->find(id1)==inputAtlasSegmentations->end()){
                    //create probabilistic segmentation
                    newProbabilisticTargetSegmentations[id1]=ImageUtils<ProbabilisticVectorImageType>::createEmpty((typename ProbabilisticVectorImageType::ConstPointer) probabilisticTargetSegmentations[id1]);
                    for (ImageListIteratorType intermediateImageIterator=inputImages->begin();intermediateImageIterator!=inputImages->end();++intermediateImageIterator){
                        string id2= intermediateImageIterator->first;
                        if (id1!=id2 ){
                            //get deformation
                            DeformationFieldPointerType deformation=deformations[id2][id1];
                            ProbabilisticVectorImagePointerType deformedProbSeg= warpProbImage(probabilisticTargetSegmentations[id2],deformation);
                            double weight=globalWeights[id2][id1];

                            //UPDATE
                            if (weighting==UNIFORM){
                                updateProbabilisticSegmentationUniform(newProbabilisticTargetSegmentations[id1],deformedProbSeg,weight);
                            }else{
                                ImagePointerType img1=targetImageIterator->second;
                                ImagePointerType img2=intermediateImageIterator->second;
                                if (weighting==GLOBAL){
                                    updateProbabilisticSegmentationGlobal(newProbabilisticTargetSegmentations[id1],deformedProbSeg,weight,img1,img2,deformation,metric);
                                    
                                }else if (weighting==LOCAL){
                                    ImagePointerType deformedImg2=TransfUtils<ImageType>::warpImage(img2,deformation);
                                    updateProbabilisticSegmentationLocal(newProbabilisticTargetSegmentations[id1],deformedProbSeg,weight,img1,img2,deformation,metric);

                                }
                            }
                        }
                    }
                }else{
                    newProbabilisticTargetSegmentations[id1]=probabilisticTargetSegmentations[id1];
                }
            }//for n1
            LOG<<"done"<<endl;
            LOG<<"Storing output. and checking convergence"<<endl;
            for (ImageListIteratorType targetImageIterator=inputImages->begin();targetImageIterator!=inputImages->end();++targetImageIterator){
                string id= targetImageIterator->first;
                ImagePointerType outputImage=probSegmentationToSegmentationLinear(newProbabilisticTargetSegmentations[id]);
                ostringstream tmpSegmentationFilename;
                tmpSegmentationFilename<<outputDir<<"/segmentation-"<<id<<"-hop"<<n<<suffix;
                ImageUtils<ImageType>::writeImage(tmpSegmentationFilename.str().c_str(),outputImage);
            }
            probabilisticTargetSegmentations=newProbabilisticTargetSegmentations;
        }// hops
        return 1;
    }//run


private:
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
    ProbabilisticVectorImagePointerType segmentationToProbabilisticVector(ImagePointerType img){
        ProbabilisticVectorImagePointerType result=createEmptyProbImageFromImage(img);
        ProbImageIteratorType probIt(result,result->GetLargestPossibleRegion());
        ImageIteratorType imgIt(img,img->GetLargestPossibleRegion());
        for (imgIt.GoToBegin(),probIt.GoToBegin();!imgIt.IsAtEnd();++imgIt,++probIt){
            ProbabilisticPixelType p;
            p.Fill(0.0);
            p[(imgIt.Get()>0)]=1;
            probIt.Set(p);
        }
        return result;
    }

    ImagePointerType probSegmentationToSegmentationLinear( ProbabilisticVectorImagePointerType img){
        ImagePointerType result=ImageType::New();
        result->SetOrigin(img->GetOrigin());
        result->SetSpacing(img->GetSpacing());
        result->SetDirection(img->GetDirection());
        result->SetRegions(img->GetLargestPossibleRegion());
        result->Allocate();
        ImageIteratorType imgIt(result,result->GetLargestPossibleRegion());
        ProbImageIteratorType probIt(img,img->GetLargestPossibleRegion());
        for (imgIt.GoToBegin(),probIt.GoToBegin();!imgIt.IsAtEnd();++imgIt,++probIt){
            float maxProb=-std::numeric_limits<float>::max();
            int maxLabel=-1;
            ProbabilisticPixelType p = probIt.Get();
            for (unsigned int s=0;s<nSegmentationLabels;++s){
                if (p[s]>maxProb){
                    maxLabel=s;
                    maxProb=p[s];
                }
            }
            if (D==2){
                imgIt.Set(1.0*std::numeric_limits<PixelType>::max()*maxLabel/(nSegmentationLabels-1));
            }else{
                imgIt.Set(maxLabel);
            }
        }
        return result;
    }

    void updateProbabilisticSegmentationUniform(ProbabilisticVectorImagePointerType accumulator, ProbabilisticVectorImagePointerType increment,double globalWeight){
        ProbImageIteratorType accIt(accumulator,accumulator->GetLargestPossibleRegion());
        ProbImageIteratorType incIt(increment,increment->GetLargestPossibleRegion());
        for (accIt.GoToBegin(),incIt.GoToBegin();!incIt.IsAtEnd();++incIt,++accIt){
            accIt.Set(accIt.Get()+incIt.Get()*globalWeight);
        }
    }

    void updateProbabilisticSegmentationGlobalMetric(ProbabilisticVectorImagePointerType accumulator, ProbabilisticVectorImagePointerType increment,double globalWeight, ImagePointerType targetImage, ImagePointerType movingImage,DeformationFieldPointerType deformation,MetricType metric ){
        ProbImageIteratorType accIt(accumulator,accumulator->GetLargestPossibleRegion());
        ProbImageIteratorType incIt(increment,increment->GetLargestPossibleRegion());
        for (accIt.GoToBegin(),incIt.GoToBegin();!incIt.IsAtEnd();++incIt,++accIt){
            accIt.Set(accIt.Get()+incIt.Get()*globalWeight);
        }
    }

    ProbabilisticVectorImagePointerType createEmptyProbImageFromImage(ImagePointerType input){
        ProbabilisticVectorImagePointerType output=ProbabilisticVectorImageType::New();
        output->SetOrigin(input->GetOrigin());
        output->SetSpacing(input->GetSpacing());
        output->SetDirection(input->GetDirection());
        output->SetRegions(input->GetLargestPossibleRegion());
        output->Allocate();
        ProbabilisticPixelType p;
        p.Fill(0.0);
        output->FillBuffer(p);
        return output;
        
    }

    ProbabilisticVectorImagePointerType warpProbImage(ProbabilisticVectorImagePointerType input, DeformationFieldPointerType deformation){
        ProbabilisticVectorImagePointerType output=ProbabilisticVectorImageType::New();
        output->SetOrigin(input->GetOrigin());
        output->SetSpacing(input->GetSpacing());
        output->SetDirection(input->GetDirection());
        output->SetRegions(input->GetLargestPossibleRegion());
        output->Allocate();
        ProbImageIteratorType outIt(output,output->GetLargestPossibleRegion());
        typedef itk::VectorLinearInterpolateNearestNeighborExtrapolateImageFunction<
           ProbabilisticVectorImageType ,double> DefaultFieldInterpolatorType;
        typename DefaultFieldInterpolatorType::Pointer interpolator=DefaultFieldInterpolatorType::New();
        interpolator->SetInputImage(input);

        DeformationIteratorType deformationIt(deformation,deformation->GetLargestPossibleRegion());
        for (outIt.GoToBegin(),deformationIt.GoToBegin();!outIt.IsAtEnd();++outIt,++deformationIt){
            IndexType index=deformationIt.GetIndex();
            typename DefaultFieldInterpolatorType::ContinuousIndexType idx(index);
            DisplacementType displacement=deformationIt.Get();
            PointType p;
            output->TransformIndexToPhysicalPoint(index,p);
            p+=displacement;
            input->TransformPhysicalPointToContinuousIndex(p,idx);
            outIt.Set(interpolator->EvaluateAtContinuousIndex(idx));
        }
        return output;
    }
    
};//class
