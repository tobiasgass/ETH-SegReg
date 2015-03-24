/**
 * @file   SegmentationNetwork.h
 * @author Tobias Gass <tobiasgass@gmail.com>
 * @date   Thu Mar 12 14:56:36 2015
 * 
 * @brief  
 * 
 * 
 */

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

#ifndef WITH_GCO
#error "GCO library required"
#else
#include "graph.h"
#endif

#include <sstream>
#include "ArgumentParser.h"
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include "itkConstNeighborhoodIterator.h"
#include "itkLabelOverlapMeasuresImageFilter.h"

namespace SSSP{

  /**
   * \brief method which solves for segmentation labels in multiple images by constructing a graph based on pairwise registrations
   * 
   */
template <class ImageType>
class SegmentationNetwork{
public:
    typedef typename ImageType::PixelType PixelType;
    static const unsigned int D=ImageType::ImageDimension;
    typedef typename  ImageType::Pointer ImagePointerType;
    typedef typename  ImageType::IndexType IndexType;
    typedef typename  ImageType::PointType PointType;
    typedef typename  ImageType::OffsetType OffsetType;
    typedef typename  ImageType::SizeType SizeType;

    typedef typename  ImageType::ConstPointer ImageConstPointerType;
    typedef typename  TransfUtils<ImageType>::DisplacementType DisplacementType;
    typedef typename  TransfUtils<ImageType>::DeformationFieldType DeformationFieldType;
    typedef typename  DeformationFieldType::Pointer DeformationFieldPointerType;
    typedef typename  itk::ImageRegionIterator<ImageType> ImageIteratorType;
    typedef typename  itk::ImageRegionIterator<DeformationFieldType> DeformationIteratorType;
    typedef typename  ImageUtils<ImageType>::FloatImageType FloatImageType;
    typedef typename  FloatImageType::Pointer FloatImagePointerType;
    typedef typename  itk::ConstNeighborhoodIterator<ImageType> ImageNeighborhoodIteratorType;
    typedef   ImageNeighborhoodIteratorType * ImageNeighborhoodIteratorPointerType;
    typedef typename  ImageNeighborhoodIteratorType::RadiusType RadiusType;


    struct ImageInformation{
        ImagePointerType img;
        int imageSize;
    };
private:
    double getMaxVectorLength(ImagePointerType img, RadiusType r){
        double maxLength=0;
        ImageNeighborhoodIteratorType tIt=ImageNeighborhoodIteratorType(r,img,img->GetLargestPossibleRegion());
        int count=0;
        for (tIt.GoToBegin();!tIt.IsAtEnd();++tIt){
            double sf=0.0,sff=0.0;
            int c=0;
            for (int i=0 ;i<tIt.Size();++i){
                bool inBounds;
                double f=tIt.GetPixel(i,inBounds);
                if (inBounds){
                    ++c;
                    sf+=f;
                    sff+=f*f;
                }
            }
            sff -= ( sf * sf / c );
            //if (fabs(sff)>fabs(maxLength))
            maxLength+=sff;
            ++count;
        }
        return maxLength/count;
    }

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
    double NCCFuncLoc(ImageNeighborhoodIteratorPointerType i1, ImageNeighborhoodIteratorPointerType i2, double smm, double sff){
        double result=0;
        double count=0, totalCount=0;
        double sfm=0.0,sf=0.0,sm=0.0;
        unsigned int i=0;
        for (;i<i1->Size();++i){
            bool inBounds;
            double f=i1->GetPixel(i,inBounds);
            if (inBounds){
                double m=i2->GetPixel(i);;
                sfm+=f*m;
                sf+=f;
                sm+=m;
                count+=1;
            }
        }
      
        sfm-=( sf * sm / count );
    
        if (smm*sff>0){
            result=((1+1.0*sfm/(sqrt(sff)*sqrt(smm)))/2);
            LOGV(40)<<VAR(result)<<" "<<VAR(sfm)<<" "<<VAR(sff)<<" "<<VAR(smm)<<endl;
            //cout<<VAR(result)<<" "<<VAR(sfm)<<" "<<VAR(sff)<<" "<<VAR(smm)<<endl;
            result=result>1?1:result;
            result=result<0?0:result;
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
            //result=((1+1.0*sfm/sqrt(sff*smm))/2);
            result=((sfm));
            LOGV(41)<<VAR(result)<<" "<<VAR(sfm)<<" "<<VAR(sff)<<" "<<VAR(smm)<<endl;
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
        feraiseexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);

        ArgumentParser * as=new ArgumentParser(argc,argv);
        string groundTruthList="",atlasSegmentationFilename,deformationFileList,imageFileList,atlasID="",supportSamplesListFileName="",outputDir=".",outputSuffix="";
        int verbose=0;
        int nImages=-1;
        double pWeight=1.0;
        double sigma=30;
        double segPairwiseWeight=0.0;
        bool NCC=false;
        bool SSD=false;
        int radius=3;
        double edgeThreshold=0.0;
        double edgeCountPenaltyWeight=0.0;
        bool evalAtlas=false;
        int nRandomSupportSamples=0;
        double confidenceThreshold=2.0;
        as->parameter ("sa", atlasSegmentationFilename, "atlas segmentation image (file name)", true);
        as->parameter ("T", deformationFileList, " list of deformations", true);
        as->parameter ("i", imageFileList, " list of  images, first image is assumed to be atlas image", true);
        as->parameter ("gt", groundTruthList, " list of  ground-truth segmentations", false);
        as->parameter ("N", nImages,"number of target images", false);
        as->parameter ("a", atlasID,"atlas ID. if not set, first image in imageFileList is assumed to be the atlas",false);
        as->parameter ("w", pWeight,"inter-image pairwise potential weight",false);
        as->parameter ("sp", segPairwiseWeight,"intra-image pairwise potential weight",false);
        as->parameter ("s", sigma,"sigma",false);
        as->parameter ("conf", confidenceThreshold,"confidence threshold (0..1]. Only pixels with a 1hop confidence <ct are included in the MRF optimization. Will speedup things, but likely loose accuracy.",false);
        as->parameter ("O", outputDir,"outputdirectory",false);
        as->option ("NCC", NCC," use NCC as weighting function");
        as->option ("SSD", SSD," use SSD as weighing function");
        as->parameter ("radius", radius,"patch radius for NCC",false);
        as->parameter ("thresh", edgeThreshold,"threshold for edge pruning (0=off)",false);
        as->parameter ("edgeCountPenaltyWeight", edgeCountPenaltyWeight,"penalize foreground label of pixels having less outgoing edges (0 to disable)",false);
        as->option ("evalAtlas", evalAtlas,"also segment the atlas within the network");

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
        RadiusType rNCC;
        for (unsigned int i = 0; i < ImageType::ImageDimension; ++i) rNCC[i] = radius;
        mkdir(outputDir.c_str(),0755);
        logSetStage("IO");
        logSetVerbosity(verbose);
        bool SAD=!NCC;
        ImagePointerType atlasSegmentationImage=ImageUtils<ImageType>::readImage(atlasSegmentationFilename);
        map<string,ImageInformation> inputImages;
        map<string,ImagePointerType>  m_groundTruthSegmentations;

        LOG<<"Reading images."<<endl;
        unsigned int totalNumberOfPixels=0;
        std::vector<string> imageIDs;
        int nTotalEdges=0;
        map<string,double> maxNormFactor;
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
                        if (NCC){
                            maxNormFactor[imageID]=getMaxVectorLength(img.img,rNCC); 
                        }
                    }
                }
        }
        if (nImages>0)
            nImages=min(nImages,(int)(inputImages.size()));
        else
            nImages=inputImages.size();

        if (groundTruthList!=""){
            ifstream ifs(groundTruthList.c_str());
            while( ! ifs.eof() ) 
                {
                    string imageID;
                    ifs >> imageID;                
                    if (imageID!=""){
                        ImageInformation img;
                        string imageFileName ;
                        ifs >> imageFileName;
                        LOGV(3)<<"Reading image "<<imageFileName<< "with ID "<<imageID<<endl;
                        m_groundTruthSegmentations[imageID]=ImageUtils<ImageType>::readImage(imageFileName);
                    }
                }
        }

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

        if (confidenceThreshold<=1.0){
            //compute 1hop confidences and network sizes


        }
    
  
        logSetStage("Init");
  
      

        unsigned int nNodes=totalNumberOfPixels;
 
	//construct MRF
	typedef Graph<float,float,double> MRFType;
        typedef MRFType::node_id NodeType;
        MRFType* optimizer;
    
        LOG<<"Allocating MRF with "<<nNodes<<" nodes and "<<nTotalEdges<<" edges."<<endl;
        optimizer = new MRFType(nNodes,nTotalEdges);
        optimizer->add_node(nNodes);
        unsigned long  int i=0;
        map <string, DeformationFieldPointerType> atlasToTargetDeformations=deformations[atlasID];
        std::vector<int> edgeCount(nNodes,0);
        int totalEdgeCount=0;
        ImagePointerType atlasImage=inputImages[atlasID].img;
        unsigned long int runningIndex=0;
        LOG<<"Setting up pairwise potentials"<<endl;
        for (unsigned int n1=0;n1<nImages;++n1){
            string id1=imageIDs[n1];
            if (evalAtlas || id1!=atlasID){
                unsigned long int runningIndex2=0;
                for (unsigned int n2=0;n2<nImages;++n2){
                    string id2=imageIDs[n2];
                    if (evalAtlas || id2!=atlasID){
                        if (n1!=n2){
                            if (! useSupportSamples || supportSampleList.find(id2)!=supportSampleList.end() || supportSampleList.find(id1)!=supportSampleList.end() ){

                                //calculate edges from all pixel of image 1 to their corresponding locations in img2
                                ImagePointerType img1=inputImages[id1].img;
                                //inverse deformation.
                                DeformationFieldPointerType deformation=deformations[id2][id1];
                                ImagePointerType img2=inputImages[id2].img;
                                ImagePointerType deformedI2=TransfUtils<ImageType>::warpImage(img2,deformation);
                                ImageIteratorType img1It(img1,img1->GetLargestPossibleRegion());
                                ImageIteratorType img2It(deformedI2,deformedI2->GetLargestPossibleRegion());
                                ImageNeighborhoodIteratorPointerType tIt=new ImageNeighborhoodIteratorType(rNCC,img1,img1->GetLargestPossibleRegion());
                                ImageNeighborhoodIteratorPointerType aIt=new ImageNeighborhoodIteratorType(rNCC,img2,img2->GetLargestPossibleRegion());
             
                                DeformationIteratorType defIt(deformation,deformation->GetLargestPossibleRegion());
                                long unsigned int i=runningIndex;
                                defIt.GoToBegin();
                                img1It.GoToBegin();
                                img2It.GoToBegin();
                                tIt->GoToBegin();
                                aIt->GoToBegin();
                                SizeType size2=img2->GetLargestPossibleRegion().GetSize();
                                SizeType size1=img1->GetLargestPossibleRegion().GetSize();
                                for (;!img1It.IsAtEnd();++img1It,++defIt,++img2It,++i,++(*tIt),++(*aIt)){
                                    IndexType idx=img1It.GetIndex();
                                    //compute index in first image to get edge endpoint
                                    PointType pt;
                                    img1->TransformIndexToPhysicalPoint(idx,pt);
                                    pt+=defIt.Get();
                                    IndexType idx2;
                                    img2->TransformPhysicalPointToIndex(pt,idx2);
                            
                                    //check if index is within image bounds
                                    bool inside=true;
                                    int withinImageIndex=ImageUtils<ImageType>::ImageIndexToLinearIndex(idx2,size2,inside);
                                    int linearIndex=runningIndex2+withinImageIndex;
                                    //LOGV(150)<<inside<<" "<<VAR(id1)<<" "<<VAR(id2)<<" "<<VAR(i)<<" "<<VAR(idx)<<" "<<VAR(linearIndex)<<" "<<VAR(idx2)<<endl;
                                    if (inside){
                                        //compute linear edge index

                                        //compute edge weight
                                        float weight=1;
                                        if (SAD) {
                                            if (sigma>0)
                                                weight=exp(-0.5*fabs(img1It.Get()-img2It.Get())/sigma);
                                        }
                                        else if (NCC) {
                                            if (radius>0)
                                                weight=NCCFuncLoc(tIt,aIt,maxNormFactor[id1],maxNormFactor[id2]);//NCCFunc(tIt,aIt);
                                        }
                                        //add bidirectional edge
                                        if (weight>edgeThreshold){
                                            weight*=pWeight;
                                            optimizer -> add_edge(i,linearIndex,weight,weight);
                                            edgeCount[i]++;
                                            //edgeCount[linearIndex]++;
                                            totalEdgeCount++;
                                        }

                                    }
                                }
                                delete tIt,aIt;
                            }
                        }
                        runningIndex2+=inputImages[id2].imageSize;
                    }

                }
                runningIndex+=inputImages[id1].imageSize;

            }


        }
        LOG<<"done"<<endl;
        LOGV(1)<<VAR(totalEdgeCount)<<endl;
        double edgeToUnaryRatio=1.0*totalEdgeCount/nNodes;
        LOGV(1)<<VAR(edgeToUnaryRatio)<<endl;
        LOG<<"Setting up unary potentials"<<std::endl;
        runningIndex=0;
        i=0;
        for (unsigned int n1=0;n1<nImages;++n1){
            string id=imageIDs[n1];
            if (id!=atlasID){
                ImagePointerType img=inputImages[id].img;
                SizeType size=img->GetLargestPossibleRegion().GetSize();
                DeformationFieldPointerType deformation= atlasToTargetDeformations[id];
                ImagePointerType deformedAtlas=TransfUtils<ImageType>::warpImage(atlasImage,deformation);
                ImagePointerType deformedAtlasSegmentation=TransfUtils<ImageType>::warpImage(atlasSegmentationImage,deformation,true);
                ImageIteratorType defSegIt(deformedAtlasSegmentation,deformedAtlasSegmentation->GetLargestPossibleRegion());
                ImagePointerType weightImage=FilterUtils<ImageType>::createEmpty(img);
                ImageIteratorType img2It(weightImage,img->GetLargestPossibleRegion());

                if (SAD){
                    ImageIteratorType imgIt(img,img->GetLargestPossibleRegion());
                    ImageIteratorType defAtlasIt(deformedAtlas,deformedAtlas->GetLargestPossibleRegion());
                    for (img2It.GoToBegin(),defAtlasIt.GoToBegin(),defSegIt.GoToBegin(),imgIt.GoToBegin();!imgIt.IsAtEnd();++imgIt,++i,++defAtlasIt,++defSegIt,++img2It)
                        {
                            PixelType segmentationLabel=defSegIt.Get();
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
                            
                            img2It.Set(int(65535.0*weight));
                            //one node fore each pixel in each non-atlas image
                            int nLocalEdges=edgeCount[i];
                            //cost for labelling node foreground
                            double e1=(segmentationLabel==0)?1:0;
                            //cost for labelling node background
                            double e0=(segmentationLabel>0)?1:0;
                        
                            if (edgeCountPenaltyWeight>0.0){
                                int nMaxPossibleEdges=nImages-2;
                                if ( useSupportSamples && supportSampleList.find(id)==supportSampleList.end()  ) {
                                    nMaxPossibleEdges=nSupportSamples;
                                }
                            
                                if (nLocalEdges)
                                    {
                                        e1+=1.0*(nMaxPossibleEdges)/nLocalEdges -1 ;
                                        LOGV(30)<<VAR(1.0*(nMaxPossibleEdges)/nLocalEdges -1)<<" "<<VAR(nMaxPossibleEdges)<<" "<<VAR(nLocalEdges)<<endl;
                                    }
                                else{
                                    e1=100000;
                                    e0=0;
                                }
                            }
                        
                            //e0*=1.0*nLocalEdges/(2*(nImages-2));

                            optimizer->add_tweights(i,edgeToUnaryRatio*weight*e1,edgeToUnaryRatio*weight*e0);

                            if (segPairwiseWeight>0){
                                IndexType idx=imgIt.GetIndex();
                                for (unsigned  int d=0;d<D;++d){
                                    OffsetType off;
                                    off.Fill(0);
                                    off[d]+=1;
                                    IndexType neighborIndex=idx+off;
                                    bool inside2;
                                    int withinImageIndex2=ImageUtils<ImageType>::ImageIndexToLinearIndex(neighborIndex,size,inside2);
                                    if (inside2){
                                        double imageIntensity2=img->GetPixel(neighborIndex);
                                        double diff=imageIntensity-imageIntensity2;
                                        double weight2;
                                        if (!SSD)
                                            weight2=exp(-0.5*fabs(diff)/sigma);
                                        else
                                            weight2=exp(-0.5*(diff*diff)/(sigma*sigma));
                                        

                                        unsigned int linearIndex2=runningIndex+withinImageIndex2;
                                        optimizer -> add_edge(i,linearIndex2,segPairwiseWeight*weight2,segPairwiseWeight*weight2);
                                        nLocalEdges++;
                                    }
                                }
                            }

                        

                        }
                }else if (NCC){
                    double sff,smm,sfmR;
                    //NCCHelper(img,deformedAtlas,smm,sff,sfmR);
                    ImageNeighborhoodIteratorPointerType tIt=new ImageNeighborhoodIteratorType(rNCC,img,img->GetLargestPossibleRegion());
                    ImageNeighborhoodIteratorPointerType aIt=new ImageNeighborhoodIteratorType(rNCC,deformedAtlas,img->GetLargestPossibleRegion());
                    for (img2It.GoToBegin(),aIt->GoToBegin(),defSegIt.GoToBegin(),tIt->GoToBegin();!tIt->IsAtEnd();++(*tIt),++i,++(*aIt),++defSegIt,++img2It)  {
                        PixelType segmentationLabel=defSegIt.Get()>0;
                        double weight;//=NCCFunc(tIt,aIt,sigma);
                        if (radius>0)
                            weight=NCCFuncLoc(tIt,aIt,maxNormFactor[id],maxNormFactor[atlasID]);
                        else 
                            weight=1;
                        img2It.Set(65535.0*weight);
                        LOGV(40)<<weight<<endl;
                        double e1=(segmentationLabel==0)?1:0;
                        double e0=(segmentationLabel)?1:0;
                        int nLocalEdges=edgeCount[i];

                        if (edgeCountPenaltyWeight>0.0){
                            int nMaxPossibleEdges=nImages-2;
                            if ( useSupportSamples && supportSampleList.find(id)==supportSampleList.end()  ) {
                                nMaxPossibleEdges=nSupportSamples;
                            }
                        
                            if (nLocalEdges)
                                {
                                    e1+=1.0*(nMaxPossibleEdges)/nLocalEdges -1 ;
                                    LOGV(20)<<VAR(1.0*(nMaxPossibleEdges)/nLocalEdges -1)<<" "<<VAR(nMaxPossibleEdges)<<" "<<VAR(nLocalEdges)<<endl;
                                }
                            else{
                                e1=100000;
                                e0=0;
                            }
                        }
                        optimizer->add_tweights(i,weight*e1,weight*e0);
                        if (segPairwiseWeight>0){
                            IndexType idx=img2It.GetIndex();

                            for (unsigned  int d=0;d<D;++d){
                                OffsetType off;
                                off.Fill(0);
                                off[d]+=1;
                                IndexType neighborIndex=idx+off;
                                bool inside2;
                                int withinImageIndex2=ImageUtils<ImageType>::ImageIndexToLinearIndex(neighborIndex,size,inside2);
                                if (inside2){
                                    unsigned int linearIndex2=runningIndex+withinImageIndex2;
                                    optimizer -> add_edge(i,linearIndex2,segPairwiseWeight,segPairwiseWeight);
                                }
                            }
                        }
                    }
                    delete tIt;
                    delete aIt;
                }
                runningIndex=i;
                if (verbose>10){
                    ostringstream tmpSegmentationFilename;
                    tmpSegmentationFilename<<outputDir<<"/weightedSeg-"<<id<<suffix;
                    ImageUtils<ImageType>::writeImage(tmpSegmentationFilename.str().c_str(),weightImage);
                    ostringstream tmpSegmentationFilename2;
                    tmpSegmentationFilename2<<outputDir<<"/deformedAtlasSeg-"<<id<<suffix;
                    ImageUtils<ImageType>::writeImage(tmpSegmentationFilename2.str().c_str(),deformedAtlas);
                }
            }else if (evalAtlas){//id!=atlasID{
                ImagePointerType img=inputImages[id].img;
                SizeType size=img->GetLargestPossibleRegion().GetSize();

                ImageIteratorType atlasIt(atlasSegmentationImage,atlasSegmentationImage->GetLargestPossibleRegion());
                for (atlasIt.GoToBegin();!atlasIt.IsAtEnd();++i,++atlasIt){
                    PixelType segmentationLabel=atlasIt.Get();
                    int nLocalEdges=edgeCount[i];
                    //cost for labelling node foreground
                    double e1=0;//(segmentationLabel==0)?1:0;
                    //cost for labelling node background
                    double e0=0;//(segmentationLabel>0)?1:0;
          
                
                    //e0*=1.0*nLocalEdges/(2*(nImages-2));
                
                    optimizer->add_tweights(i,edgeToUnaryRatio*e1,edgeToUnaryRatio*e0);
              

                }


            }

        }
        optimizer->add_tweights(nNodes,0,nNodes);

    
        LOGV(1)<<"done"<<endl;
 
        logSetStage("Optimization");
        LOG<<"starting optimization"<<endl;
        float flow = optimizer -> maxflow();
        LOG<<"Done, resulting energy is "<<flow<< std::endl;
        logSetStage("Finalizing and storing output.");
        i=0;

        for (unsigned int n1=0;n1<nImages;++n1){
            string id1=imageIDs[n1];
            if (evalAtlas || id1!=atlasID){
                ImageIteratorType imgIt(inputImages[id1].img,inputImages[id1].img->GetLargestPossibleRegion());
                imgIt.GoToBegin();
                for (;!imgIt.IsAtEnd();++imgIt,++i){
                    double mult=D==2?65535:1.0;
                    
                    imgIt.Set(mult*(optimizer->what_segment(i)== MRFType::SINK ));
                }
                ostringstream tmpSegmentationFilename;
                tmpSegmentationFilename<<outputDir<<"/segmentation-"<<id1<<"-MRF-nImages"<<nImages<<suffix;
                ImageUtils<ImageType>::writeImage(tmpSegmentationFilename.str().c_str(),inputImages[id1].img);

                if ((m_groundTruthSegmentations)[id1].IsNotNull()){
                    typedef typename itk::LabelOverlapMeasuresImageFilter<ImageType> OverlapMeasureFilterType;
                    typename OverlapMeasureFilterType::Pointer filter = OverlapMeasureFilterType::New();
                    filter->SetSourceImage((m_groundTruthSegmentations)[id1]);
                    filter->SetTargetImage(inputImages[id1].img);
                    filter->SetCoordinateTolerance(1e-4);
                    filter->Update();
                    double dice=filter->GetDiceCoefficient();
                    LOGV(1)<<VAR(atlasID)<<" "<<VAR(id1)<<" "<<VAR(dice)<<endl;
                }
            }
        }
    
        delete optimizer;
        return 1;
    }
};//class
}//namespace
