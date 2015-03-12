
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
#include "bgraph.h"
#include <sstream>
#include "ArgumentParser.h"
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include "itkConstNeighborhoodIterator.h"

using namespace std;

template <class ImageType>
class SegmentationPropagationDeformationWeighting{
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


    struct ImageInformation{
        ImagePointerType img;
        int imageSize;
    };
private:
    void NCCHelper(ImagePointerType i1, ImagePointerType i2, double & smmR , double & sffR , double   & sfmR ){
        ImageIteratorType it1(i1,i1->GetLargestPossibleRegion().GetSize());
        ImageIteratorType it2(i2,i2->GetLargestPossibleRegion().GetSize());
        int c=0;
        it1.GoToBegin();
        it2.GoToBegin();
        double sff=0.0,smm=0.0,sfm=0.0,sf=0.0,sm=0.0;

        for (;!it1.IsAtEnd();++it1,++it2,++c){
            double f=it1.Get();
            double m=it2.Get();
            sff+=f*f;
            smm+=m*m;
            sfm+=f*m;
        }
        sffR =sff- ( sf * sf / c );
        smmR = smm -( sm * sm / c );
        sfmR = sfm - ( sf * sm / c );
        //sfmR = ( sf * sm / c );
    
    }
    double NCCFuncLoc(ImageNeighborhoodIteratorPointerType i1, ImageNeighborhoodIteratorPointerType i2, double smm, double sff,double sfmR){
        double result=0;
        double count=0, totalCount=0;
        double sfm=0.0,sf=0.0,sm=0.0;
        unsigned int i=0;
        for (;i<i1->Size();++i){
            bool inBounds;
            double f=i1->GetPixel(i,inBounds);

            if (inBounds){
                double m=i2->GetPixel(i);;
                if (m!=0){

                    sfm+=f*m;
                    sf+=f;
                    sm+=m;
                    count+=1;
                }
            }
        }
        //sfm -= sfmR;
        //sfm = sfmR;
        sfm-=( sf * sm / count );
    
        if (smm*sff>0){
            result=((1+1.0*sfm/sqrt(sff*smm))/2);
            //cout<<VAR(result)<<" "<<VAR(sfm)<<" "<<VAR(sff)<<" "<<VAR(smm)<<endl;
            result=result;
        }else{
            result=1;
        }
        return result;
    }

    double NCCFunc(ImageNeighborhoodIteratorPointerType i1, ImageNeighborhoodIteratorPointerType i2, double sigma=-1){
        double result=0;
        double count=0, totalCount=0;
        double sff=0.0,smm=0.0,sfm=0.0,sf=0.0,sm=0.0;
        unsigned int i=0;
        IndexType centerIndex=i1->GetIndex(i1->Size()/2);

        for (;i<i1->Size();++i){
            bool inBounds;
            double f=i1->GetPixel(i,inBounds);
            if (inBounds){
                double m=i2->GetPixel(i);;
                IndexType idx=i1->GetIndex(i);
              
                sff+=f*f;
                smm+=m*m;
                sfm+=f*m;
                sf+=f;
                sm+=m;
                count+=1;
            }
        }
        smm -= ( sm * sm / count );
        sff -= ( sf * sf / count );
        sfm -= ( sf * sm / count );
    
        if (smm*sff>0){
            result=((1+1.0*sfm/sqrt(sff*smm))/2);
            //    cout<<VAR(result)<<" "<<VAR(sfm)<<" "<<VAR(sff)<<" "<<VAR(smm)<<endl;
            result=result;
        }else{
            result=0;
        }
        return result;
    }
    
    template<class bidiiter>
    bidiiter random_unique(bidiiter begin, bidiiter end, size_t num_random, string ID) {
        size_t left = std::distance(begin, end);
        srand ( time(NULL) );

        while (num_random--) {
            bidiiter r = begin;
            std::advance(r, rand()%left);
            if ((*r)!=ID){
                std::swap(*begin, *r);
                ++begin;
                --left;
            }else{
                num_random++;
            }
        }
        return begin;
    }
    
public:
    int run(int argc, char ** argv)
    {
        feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);


    

  
        ArgumentParser * as=new ArgumentParser(argc,argv);
        string atlasSegmentationFilename,deformationFileList,imageFileList,atlasID="",supportSamplesListFileName="",outputDir=".",outputSuffix="";
        int verbose=0;
        int nImages=-1;
        double pWeight=1.0;
        double sigma=30;
        double segPairwiseWeight=0.0;
        bool NCC=false;
        bool SSD=false;
        bool DEF=false;
        int radius=3;
        double edgeThreshold=0.0;
        double edgeCountPenaltyWeight=1.0;
        bool evalAtlas=false;
        int nRandomSupportSamples=0;        
        int maxHops=10;

        as->parameter ("sa", atlasSegmentationFilename, "atlas segmentation image (file name)", true);
        as->parameter ("T", deformationFileList, " list of deformations", true);
        as->parameter ("i", imageFileList, " list of  images, first image is assumed to be atlas image", true);
        as->parameter ("N", nImages,"number of target images", false);
        as->parameter ("a", atlasID,"atlas ID. if not set, first image in imageFileList is assumed to be the atlas",false);
        as->parameter ("w", pWeight,"inter-image pairwise potential weight",false);
        as->parameter ("sp", segPairwiseWeight,"intra-image pairwise potential weight",false);
        as->parameter ("s", sigma,"sigma",false);
        as->parameter ("O", outputDir,"outputdirectory",false);
        as->option ("NCC", NCC," use NCC as weighting function");
        as->option ("SSD", SSD," use SSD as weighing function");
        as->option ("DEF", DEF," use DEF as weighing function");
        as->parameter ("radius", radius,"patch radius for NCC",false);
        as->parameter ("thresh", edgeThreshold,"threshold for edge pruning (0=off)",false);
        as->parameter ("edgeCountPenaltyWeight", edgeCountPenaltyWeight,"penalize foreground label of pixels having less outgoing edges (0 to disable)",false);
        as->option ("evalAtlas", evalAtlas,"also segment the atlas within the network");
        as->parameter ("maxHops", maxHops,"maximum number of hops",false);

        as->parameter ("supportSamples",supportSamplesListFileName,"filename with a list of support sample IDs. if not set, all images will be used.",false);
        as->parameter ("nRandomSupportSamples",nRandomSupportSamples,"draw random target images as support samples.",false);
        as->parameter ("verbose", verbose,"get verbose output",false);
        as->help();
        as->parse();
        string suffix;
        if (D==2)
            suffix=".png";
        else
            suffix=".nii";

        mkdir(outputDir.c_str(),0755);
        logSetStage("IO");
        logSetVerbosity(verbose);
        bool SAD=!NCC && ! DEF;
        FloatImagePointerType atlasSegmentationImage=FilterUtils<ImageType,FloatImageType>::cast(FilterUtils<ImageType>::binaryThresholdingLow(ImageUtils<ImageType>::readImage(atlasSegmentationFilename),1));
        map<string,ImageInformation> inputImages;
        LOG<<"Reading images."<<endl;
        unsigned int totalNumberOfPixels=0;
        std::vector<string> imageIDs;
        int nTotalEdges=0;

        {
            ifstream ifs(imageFileList.c_str());
            while( ! ifs.eof() ) 
                {
                    string imageID;
                    ifs >> imageID;                
                    if (imageID!=""){
                        imageIDs.push_back(imageID);
                        ImageInformation img;
                        string imageFileName ;
                        ifs >> imageFileName;
                        LOGV(3)<<"Reading image "<<imageFileName<< "with ID "<<imageID<<endl;
                        img.img=ImageUtils<ImageType>::readImage(imageFileName);
                        img.imageSize=1;
                        SizeType imgSize=img.img->GetLargestPossibleRegion().GetSize();
                        for (unsigned int d=0;d<D;++d){
                            img.imageSize*=imgSize[d];
                        }
                        if (evalAtlas || imageID!=atlasID){
                            totalNumberOfPixels+=img.imageSize;
                            //add edges between pixels if weight is >0
                            if (segPairwiseWeight>0){
                                int interImgEdges=imgSize[0]*(imgSize[1]-1);
                                interImgEdges+=imgSize[1]*(imgSize[0]-1);
                                if (D==3){
                                    interImgEdges=interImgEdges*imgSize[2];
                                    interImgEdges+=(imgSize[2]-1)*imgSize[0]*imgSize[1];
                                }
                                LOGV(3)<<VAR(interImgEdges)<<endl;
                                nTotalEdges+=interImgEdges;
                            }
                        }
                        if (inputImages.find(imageID)==inputImages.end())
                            inputImages[imageID]=img;
                        else{
                            LOG<<"duplicate ID "<<imageID<<", aborting"<<endl;
                            exit(0);
                        }
                    }
                }
        }
        if (nImages>0)
            nImages=min(nImages,(int)(inputImages.size()));
        else
            nImages=inputImages.size();

        bool useSupportSamples=false;
        map<string,bool> supportSampleList;
        int nSupportSamples=0;
        if (supportSamplesListFileName!=""){
            useSupportSamples=true;
            ifstream ifs(supportSamplesListFileName.c_str());
            while( ! ifs.eof() ) {
                string supID;
                ifs>>supID;
                if (supID !=""){
                    supportSampleList[supID]=true;
                    nSupportSamples++;
                }
            }
        }else if (nRandomSupportSamples){
            useSupportSamples=true;
            nSupportSamples=nRandomSupportSamples;
            std::vector<string> tmpList = imageIDs;
            random_unique(tmpList.begin(),tmpList.end(),nRandomSupportSamples,atlasID);
            for (int i=0;i<nRandomSupportSamples;++i){
                supportSampleList[tmpList[i]]=true;
            }
        }
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
                    if ( id1 == atlasID|| (evalAtlas && id2 == atlasID)  || (! useSupportSamples || supportSampleList.find(id1)!=supportSampleList.end() ||  ( (supportSampleList.find(id2)!=supportSampleList.end()) && (id1!=atlasID))) ){
                         if (inputImages.find(id1)==inputImages.end() || inputImages.find(id2)==inputImages.end() ){
                            LOG<<id1<<" or "<<id2<<" not in image database, skipping"<<endl;
                            //exit(0);
                        }else{
                            LOGV(3)<<"Reading deformation "<<defFileName<<" for deforming "<<id1<<" to "<<id2<<endl;
                            DeformationFieldPointerType def=ImageUtils<DeformationFieldType>::readImage(defFileName);
                            deformations[id1][id2]=def;
                        
                            if ( evalAtlas || (id1!= atlasID && id2 != atlasID)){
                                int nDefs=1;
                                for (unsigned int d=0;d<D;++d){
                                    nDefs*=def->GetLargestPossibleRegion().GetSize()[d];
                                }
                                nTotalEdges+=nDefs;
                      
                            }
                        }

                    }
                }
            }
        }
        LOG<<"Computing deformation statistics"<<endl;
        map< string,  DeformationFieldPointerType > avgDeformations, deformationVariances;
        for (unsigned int n1=0;n1<nImages;++n1){
                string id1=imageIDs[n1];
                if ( id1!=atlasID){
                    ImagePointerType img1=inputImages[id1].img;
                    avgDeformations[id1]=TransfUtils<ImageType>::createEmpty(img1);
                    deformationVariances[id1]=TransfUtils<ImageType>::createEmpty(img1);  
                    DeformationIteratorType avgDefIt(avgDeformations[id1],avgDeformations[id1]->GetLargestPossibleRegion());
                    DeformationIteratorType varDefIt(deformationVariances[id1],deformationVariances[id1]->GetLargestPossibleRegion());
                    int count=0;
                    for (unsigned int n2=0;n2<nImages;++n2){
                        string id2=imageIDs[n2];
                        if (n1!=n2 ){
                            DeformationFieldPointerType def;
                            if (id2==atlasID){
                                def=deformations[atlasID][id1];
                            }else{
                                DeformationFieldPointerType def1=deformations[id2][id1];
                                DeformationFieldPointerType def2=deformations[atlasID][id2];
                                def=TransfUtils<ImageType>::composeDeformations(def1,def2);
                            }
                            DeformationIteratorType defIt(def,def->GetLargestPossibleRegion());
                          
                            defIt.GoToBegin();
                            avgDefIt.GoToBegin();
                            varDefIt.GoToBegin();
                            for (;!defIt.IsAtEnd();++defIt,++avgDefIt,++varDefIt){
                                DisplacementType t=defIt.Get();
                                avgDefIt.Set(avgDefIt.Get()+t);
                                for (unsigned int d=0;d<D;++d){
                                    t[d]=t[d]*t[d];
                                }
                                varDefIt.Set(varDefIt.Get()+t);
                            }
                            count++;

                        }
                    }
                    avgDefIt.GoToBegin();
                    varDefIt.GoToBegin();
                    for (;!avgDefIt.IsAtEnd();++avgDefIt,++varDefIt){
                        DisplacementType mean=avgDefIt.Get();
                        DisplacementType var=varDefIt.Get();
                        for (unsigned int d=0;d<D;++d){
                            mean[d]/=count;
                            var[d]=var[d]/count - mean[d]*mean[d];
                            LOGV(28)<<VAR(mean[d])<<" "<<VAR(var[d])<<endl;
                        }
                        avgDefIt.Set(mean);
                        varDefIt.Set(var);
                    }
                    ostringstream tmpSegmentationFilename;
                    tmpSegmentationFilename<<outputDir<<"/avgDeformedAtlasSegmentation-"<<id1<<suffix;
                    FloatImagePointerType deformedAtlasSeg=TransfUtils<FloatImageType>::warpSegmentationImage(atlasSegmentationImage,avgDeformations[id1]);
                    ImageUtils<FloatImageType>::multiplyImage(deformedAtlasSeg,65535);
                    ImageUtils<ImageType>::writeImage(tmpSegmentationFilename.str().c_str(),FilterUtils<FloatImageType,ImageType>::cast(deformedAtlasSeg));
                }
        }
        LOG<<"Done"<<endl;
        ImagePointerType atlasImage=inputImages[atlasID].img;
        map <string, DeformationFieldPointerType> atlasToTargetDeformations=deformations[atlasID];
        logSetStage("Zero Hop");
        LOG<<"Computing"<<std::endl;
        RadiusType rNCC;
        for (unsigned int i = 0; i < ImageType::ImageDimension; ++i) rNCC[i] = radius;
        std::vector<FloatImagePointerType> segmentations(nImages,NULL);
        for (unsigned int n1=0;n1<nImages;++n1){
            string id=imageIDs[n1];
            if (id!=atlasID){
                ImagePointerType img=inputImages[id].img;
                SizeType size=img->GetLargestPossibleRegion().GetSize();
                DeformationFieldPointerType deformation=atlasToTargetDeformations[id];
                ImagePointerType deformedAtlas=TransfUtils<ImageType>::warpImage(atlasImage,deformation);
                FloatImagePointerType deformedAtlasSegmentation=TransfUtils<FloatImageType>::warpSegmentationImage(atlasSegmentationImage,deformation);
                FloatImageIteratorType defSegIt(deformedAtlasSegmentation,deformedAtlasSegmentation->GetLargestPossibleRegion());
               
                FloatImagePointerType probSegmentation=FilterUtils<ImageType,FloatImageType>::createEmptyFrom(img);
                probSegmentation->FillBuffer(0.0);
                FloatImagePointerType normalization=FilterUtils<ImageType,FloatImageType>::createEmptyFrom(img);
                normalization->FillBuffer(0.0);
                FloatImageIteratorType normIt(normalization,normalization->GetLargestPossibleRegion());
                FloatImageIteratorType img2It(probSegmentation,probSegmentation->GetLargestPossibleRegion());

                if (SAD){
                    ImageIteratorType imgIt(img,img->GetLargestPossibleRegion());
                    ImageIteratorType defAtlasIt(deformedAtlas,deformedAtlas->GetLargestPossibleRegion());
                    for (normIt.GoToBegin(),img2It.GoToBegin(),defAtlasIt.GoToBegin(),defSegIt.GoToBegin(),imgIt.GoToBegin();!imgIt.IsAtEnd();++imgIt,++defAtlasIt,++defSegIt,++img2It,++normIt)
                        {
                            float segmentationLabel=defSegIt.Get();
                            double imageIntensity=imgIt.Get();
                            double deformedAtlasIntensity=defAtlasIt.Get();
                            double diff=imageIntensity-deformedAtlasIntensity;
                            double weight=1;
                            if (sigma>0){
                                if (!SSD)
                                    weight=exp(-0.5*fabs(diff)/sigma);
                                else
                                    weight=exp(-0.5*(diff*diff)/(sigma*sigma));
                            }

                            //LOGV(51)<<VAR(imageIntensity)<<" "<<VAR(deformedAtlasIntensity)<<" "<<VAR(weight)<<endl;
                            
                            img2It.Set(img2It.Get()+(weight)*segmentationLabel);
                            normIt.Set(normIt.Get()+weight);
                        }
                }else if (NCC){
                    double sff,smm,sfmR;
                    //NCCHelper(img,deformedAtlas,smm,sff,sfmR);
                    ImageNeighborhoodIteratorPointerType tIt=new ImageNeighborhoodIteratorType(rNCC,img,img->GetLargestPossibleRegion());
                    ImageNeighborhoodIteratorPointerType aIt=new ImageNeighborhoodIteratorType(rNCC,deformedAtlas,img->GetLargestPossibleRegion());
                    for (img2It.GoToBegin(),aIt->GoToBegin(),defSegIt.GoToBegin(),tIt->GoToBegin();!tIt->IsAtEnd();++(*tIt),++(*aIt),++defSegIt,++img2It)  {
                        PixelType segmentationLabel=defSegIt.Get()>0;
                        double weight=1;
                        if (radius>0)
                            NCCFunc(tIt,aIt,sigma);
                        //weight=NCCFuncLoc(tIt,aIt,smm,sff,sfmR);
                        img2It.Set( img2It.Get()+weight*segmentationLabel);
                    }
                    delete tIt;
                    delete aIt;
                }else if(DEF){
                    DeformationIteratorType avgDefIt(avgDeformations[id],avgDeformations[id]->GetLargestPossibleRegion());
                    DeformationIteratorType varDefIt(deformationVariances[id],avgDeformations[id]->GetLargestPossibleRegion());
                    DeformationIteratorType defIt(deformation,deformation->GetLargestPossibleRegion());
                    avgDefIt.GoToBegin();
                    defIt.GoToBegin();
                    varDefIt.GoToBegin();
                    defSegIt.GoToBegin();
                    for (defSegIt.GoToBegin();!defSegIt.IsAtEnd();++defSegIt,++avgDefIt,++defIt,++varDefIt,++img2It){
                        double weight=0;
                        PixelType segmentationLabel=defSegIt.Get()>0;
                        DisplacementType t,avg,variance;
                        t=defIt.Get();
                        avg=avgDefIt.Get();
                        variance=varDefIt.Get();
                        for (unsigned int d=0;d<D;++d){
                            double diff=t[d]-avg[d];
                            LOGV(20)<<VAR(diff)<<" "<<VAR(weight)<<" "<<VAR(variance[d])<<endl;
                            diff*=diff;
                            //                            if (variance[d])
                                //diff/=variance[d];

                            weight+=diff;
                        }
                        weight=exp(-0.5*(weight/(sigma*sigma)));
                        LOGV(19)<<VAR(weight)<<" "<<VAR(weight*segmentationLabel)<<endl;
                        img2It.Set(weight*segmentationLabel);

                    }
                }
#if 0
                //normalizing only makes sense with multiple atlas segmentations
                img2It.GoToBegin();
                normIt.GoToBegin();

                for (;!img2It.IsAtEnd();++img2It,++normIt){
                    float norm=normIt.Get();
                    if (norm!=0.0)
                        img2It.Set(img2It.Get()/norm);
                    else{
                        img2It.Set(0.0);
                    }
                }
#endif
                segmentations[n1]=probSegmentation;
            }
            else{
                segmentations[n1]=(atlasSegmentationImage);
            }
        }//finished zero-hop segmentation
        LOG<<"done"<<endl;
        LOGV(1)<<"Storing zero-hop segmentations."<<endl;
        for (unsigned int n1=0;n1<nImages;++n1){
            string id1=imageIDs[n1];
            if (evalAtlas || id1!=atlasID){
                ImagePointerType outputImage=FilterUtils<FloatImageType,ImageType>::createEmptyFrom(segmentations[n1]);
                ImageIteratorType outIt(outputImage,outputImage->GetLargestPossibleRegion());
                FloatImageIteratorType imgIt(segmentations[n1],segmentations[n1]->GetLargestPossibleRegion());
                outIt.GoToBegin();
                imgIt.GoToBegin();
                for (;!imgIt.IsAtEnd();++imgIt,++outIt){
                    outIt.Set(65535*imgIt.Get());
                    LOGV(17)<<VAR(outIt.Get())<<endl;
                }
                ostringstream tmpSegmentationFilename;
                tmpSegmentationFilename<<outputDir<<"/segmentation-"<<id1<<"-hop0"<<suffix;
                ImageUtils<ImageType>::writeImage(tmpSegmentationFilename.str().c_str(),outputImage);
            }
        }
        bool converged=false;
        logSetStage("Iterative improvement");
        for (int n=1;n<maxHops && !converged;++n){
            LOG<<"hop "<<n<<endl;
            std::vector<FloatImagePointerType> newSegmentations(nImages,NULL);
            for (unsigned int n1=0;n1<nImages;++n1){
                string id1=imageIDs[n1];
                if ( id1!=atlasID){
                    ImagePointerType img1=inputImages[id1].img;
                    //create probabilistic segmentation
                    FloatImagePointerType probSeg=FilterUtils<ImageType,FloatImageType>::createEmptyFrom(img1);
                    probSeg->FillBuffer(0.0);
                    FloatImagePointerType normalization=FilterUtils<ImageType,FloatImageType>::createEmptyFrom(img1);
                    normalization->FillBuffer(0.0);
                    FloatImageIteratorType probIt(probSeg,probSeg->GetLargestPossibleRegion());
                    FloatImageIteratorType normIt(normalization,probSeg->GetLargestPossibleRegion());

                    unsigned long int runningIndex2=0;
                    for (unsigned int n2=0;n2<nImages;++n2){
                        string id2=imageIDs[n2];
                        if (n1!=n2 ){
                            if (! useSupportSamples || supportSampleList.find(id2)!=supportSampleList.end() || supportSampleList.find(id1)!=supportSampleList.end() ){
                                
                         
                                DeformationFieldPointerType deformation=deformations[id2][id1];
                                ImagePointerType img2=inputImages[id2].img;
                                ImagePointerType deformedI2=TransfUtils<ImageType>::warpImage(img2,deformation);
                                FloatImagePointerType deformedSegmentation=TransfUtils<FloatImageType>::warpImage(segmentations[n2],deformation);

                                ImageIteratorType img1It(img1,img1->GetLargestPossibleRegion());
                                ImageIteratorType img2It(deformedI2,deformedI2->GetLargestPossibleRegion());
                                FloatImageIteratorType deformedSegmentationIt(deformedSegmentation,deformedSegmentation->GetLargestPossibleRegion());
                                
                                DeformationFieldPointerType composedDef;
                                if (id2!=atlasID){
                                    DeformationFieldPointerType def1=deformations[id2][id1];
                                    DeformationFieldPointerType def2=deformations[atlasID][id2];
                                    composedDef=TransfUtils<ImageType>::composeDeformations(def1,def2);
                                }else{
                                    composedDef=deformations[id2][id1];
                                }
                                DeformationIteratorType avgDefIt(avgDeformations[id1],avgDeformations[id1]->GetLargestPossibleRegion());
                                DeformationIteratorType varDefIt(deformationVariances[id1],avgDeformations[id1]->GetLargestPossibleRegion());
                                DeformationIteratorType defIt(composedDef,composedDef->GetLargestPossibleRegion());

                                
                                ImageNeighborhoodIteratorPointerType tIt=new ImageNeighborhoodIteratorType(rNCC,img1,img1->GetLargestPossibleRegion());
                                ImageNeighborhoodIteratorPointerType aIt=new ImageNeighborhoodIteratorType(rNCC,img2,img2->GetLargestPossibleRegion());
             
                                deformedSegmentationIt.GoToBegin();
                                img1It.GoToBegin();
                                img2It.GoToBegin();
                                tIt->GoToBegin();
                                aIt->GoToBegin();
                                probIt.GoToBegin();
                                normIt.GoToBegin();
                                avgDefIt.GoToBegin();
                                varDefIt.GoToBegin();
                                defIt.GoToBegin();
                                SizeType size2=img2->GetLargestPossibleRegion().GetSize();
                                SizeType size1=img1->GetLargestPossibleRegion().GetSize();
                                for (;!img1It.IsAtEnd();++img1It,++deformedSegmentationIt,++img2It,++(*tIt),++(*aIt),++probIt,++normIt,++defIt,++varDefIt,++avgDefIt){
                                
                                    float weight=1;
                                    
                                    if (sigma>0 && SAD) {
                                        if (SSD){
                                             weight=exp(-0.5*(img1It.Get()-img2It.Get())*(img1It.Get()-img2It.Get())/(sigma*sigma));
                                        }else{
                                             weight=exp(-0.5*fabs(img1It.Get()-img2It.Get())/sigma);
                                        }
                                    }
                                    else if (NCC) {
                                        if (radius>0)weight=NCCFunc(tIt,aIt);
                                    }else if (DEF){
                                        weight=0.0;
                                        DisplacementType t,avg,variance;
                                        t=defIt.Get();
                                        avg=avgDefIt.Get();
                                        variance=varDefIt.Get();
                                        for (unsigned int d=0;d<D;++d){
                                            double diff=t[d]-avg[d];
                                            LOGV(20)<<VAR(diff)<<" "<<VAR(weight)<<" "<<VAR(variance[d])<<endl;
                                            diff*=diff;
                                            //  if (variance[d])
                                            //  diff/=variance[d];
                                            
                                            weight+=diff;
                                        }
                                        weight=exp(-0.5*(weight/(sigma*sigma)));
                                        //weight=exp(-0.5*weight);
                                        
                                    }
                                    probIt.Set(probIt.Get()+weight*(deformedSegmentationIt.Get()));
                                    normIt.Set(normIt.Get()+weight);
                                    
                                }
                                delete tIt,aIt;
                            }
                            
                        }
                    }//for n2
                    //normalize the weighted sum
                    probIt.GoToBegin();
                    normIt.GoToBegin();
                    for (;!probIt.IsAtEnd();++probIt,++normIt){
                        float norm=normIt.Get();
                        if (norm!=0.0)
                            probIt.Set((probIt.Get()/norm));
                        else
                            probIt.Set(0.0);
                    }
                    newSegmentations[n1]=probSeg;
                }else{
                    
                    newSegmentations[n1]=segmentations[n1];
                }
            }//for n1
            LOG<<"done"<<endl;
            LOG<<"Storing output. and checking convergence"<<endl;
            for (unsigned int n1=0;n1<nImages;++n1){
                string id1=imageIDs[n1];
                if (evalAtlas || id1!=atlasID){
                    ImagePointerType outputImage=FilterUtils<FloatImageType,ImageType>::createEmptyFrom(newSegmentations[n1]);
                    ImageIteratorType outIt(outputImage,outputImage->GetLargestPossibleRegion());
                    FloatImageIteratorType imgIt(newSegmentations[n1],newSegmentations[n1]->GetLargestPossibleRegion());
                    outIt.GoToBegin();
                    imgIt.GoToBegin();
                    for (;!imgIt.IsAtEnd();++imgIt,++outIt){
                        outIt.Set(65535*(imgIt.Get()>0.5));
                    }
                    ostringstream tmpSegmentationFilename;
                    tmpSegmentationFilename<<outputDir<<"/segmentation-"<<id1<<"-hop"<<n<<suffix;
                    ImageUtils<ImageType>::writeImage(tmpSegmentationFilename.str().c_str(),outputImage);
                }
            }
            segmentations=newSegmentations;
            
        }// hops
        return 1;
    }//run
};//class
