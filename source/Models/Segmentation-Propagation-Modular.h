
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

    
public:
    int run(int argc, char ** argv)
    {
        feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
        argstream * as=new argstream(argc,argv);
        string deformationFileList,imageFileList,atlasSegmentationFileList,supportSamplesListFileName="",outputDir=".",outputSuffix="";
        int verbose=0;
        double pWeight=1.0;
        double sigma=30;
        bool NCC=false;
        bool SSD=false;
        int radius=3;
        int maxHops=10;
        (*as) >> parameter ("A",atlasSegmentationFileList , "list of atlas segmentations <id> <file>", true);
        (*as) >> parameter ("T", deformationFileList, " list of deformations", true);
        (*as) >> parameter ("i", imageFileList, " list of  images", true);
        (*as) >> parameter ("s", sigma,"sigma",false);
        (*as) >> parameter ("O", outputDir,"outputdirectory",false);
        (*as) >> option ("NCC", NCC," use NCC as weighting function");
        (*as) >> option ("SSD", SSD," use SSD as weighing function");
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
        bool SAD=!NCC;
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
                    }
                }
            }
        }

    
        logSetStage("Zero Hop");
        LOG<<"Computing"<<std::endl;
        map<string,ProbabilisticVectorImagePointerType> probabilisticTargetSegmentations;
        for (ImageListIteratorType targetImageIterator=inputImages->begin();targetImageIterator!=inputImages->end();++targetImageIterator){
            string id= targetImageIterator->first;
            if (inputAtlasSegmentations->find(id)==inputAtlasSegmentations->end()){
                ImagePointerType targetImage= targetImageIterator->second;
                probabilisticTargetSegmentations[id]=createEmptyProbImageFromImage(targetImage);
                //do not calculate segmentation for atlas images
                for (ImageListIteratorType atlasIterator=inputAtlasSegmentations->begin();atlasIterator!=inputAtlasSegmentations->end();++atlasIterator){
                    string atlasID=atlasIterator->first;
                    ImagePointerType atlasImage=(*inputImages)[atlasID];
                    DeformationFieldPointerType deformation= deformations[atlasID][id];
                    ImagePointerType deformedAtlas=TransfUtils<ImageType>::warpImage(atlasImage,deformation);
                    ImagePointerType deformedAtlasSegmentation=TransfUtils<ImageType>::warpSegmentationImage(atlasIterator->second,deformation);
                    ProbabilisticVectorImagePointerType probDeformedAtlasSegmentation=segmentationToProbabilisticVector(deformedAtlasSegmentation);
                    double weight=1.0;
                    updateProbabilisticSegmentationUniform(probabilisticTargetSegmentations[id],probDeformedAtlasSegmentation,weight,targetImage,atlasImage);
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
        for (int n=1;n<maxHops && !converged;++n){
            LOG<<"hop "<<n<<endl;
            for (ImageListIteratorType targetImageIterator=inputImages->begin();targetImageIterator!=inputImages->end();++targetImageIterator){
                string id1= targetImageIterator->first;
                if ( inputAtlasSegmentations->find(id1)==inputAtlasSegmentations->end()){
                    ImagePointerType img1=inputImages[id1].img;
                    //create probabilistic segmentation
                    for (ImageListIteratorType intermediateImageIterator=inputImages->begin();intermediateImageIterator!=inputImages->end();++intermediateImageIterator){
                        string id2= intermediateImageIterator->first;
                        if (id1!=id2 ){
                            DeformationFieldPointerType deformation=deformations[id2][id1];
                            ImagePointerType img2=inputImages[id2].img;
                            
                        }
                    }
                }else{

                }
            }//for n1
            LOG<<"done"<<endl;
            LOG<<"Storing output. and checking convergence"<<endl;
            for (unsigned int n1=0;n1<nImages;++n1){
             
            }
            segmentations=newSegmentations;
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

    void updateProbabilisticSegmentationUniform(ProbabilisticVectorImagePointerType accumulator, ProbabilisticVectorImagePointerType increment,double globalWeight, ImagePointerType img1, ImagePointerType img2){
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
    
};//class
