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

typedef unsigned short int PixelType;
const unsigned int D=2;
typedef itk::Image<PixelType,D> ImageType;
typedef ImageType::Pointer ImagePointerType;
typedef ImageType::IndexType IndexType;
typedef ImageType::PointType PointType;
typedef ImageType::OffsetType OffsetType;
typedef ImageType::SizeType SizeType;

typedef ImageType::ConstPointer ImageConstPointerType;
typedef TransfUtils<ImageType>::DisplacementType DisplacementType;
typedef TransfUtils<ImageType>::DeformationFieldType DeformationFieldType;
typedef DeformationFieldType::Pointer DeformationFieldPointerType;
typedef itk::ImageRegionIterator<ImageType> ImageIteratorType;
typedef itk::ImageRegionIterator<DeformationFieldType> DeformationIteratorType;
typedef ImageUtils<ImageType>::FloatImageType FloatImageType;
typedef FloatImageType::Pointer FloatImagePointerType;
typedef itk::ConstNeighborhoodIterator<ImageType> ImageNeighborhoodIteratorType;
typedef ImageNeighborhoodIteratorType * ImageNeighborhoodIteratorPointerType;
typedef ImageNeighborhoodIteratorType::RadiusType RadiusType;

using namespace std;

struct ImageInformation{
    ImagePointerType img;
    int imageSize;
};



int main(int argc, char ** argv)
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
    int radius=3;
    as->parameter ("sa", atlasSegmentationFilename, "atlas segmentation image (file name)", true);
    as->parameter ("T", deformationFileList, " list of deformations", true);
    as->parameter ("i", imageFileList, " list of  images, first image is assumed to be atlas image", true);
    as->parameter ("N", nImages,"number of target images", false);
    as->parameter ("a", atlasID,"atlas ID. if not set, first image in imageFileList is assumed to be the atlas",false);
    as->parameter ("w", pWeight,"inter-image pairwise potential weight",false);
    as->parameter ("sp", segPairwiseWeight,"intra-image pairwise potential weight (NOT YET IMPLEMENTED)",false);
    as->parameter ("s", sigma,"sigma",false);
    as->parameter ("O", outputDir,"outputdirectory",false);
    as->option ("NCC", NCC,"outputdirectory");
    as->parameter ("radius", radius,"patch radius for NCC",false);

    as->parameter ("supportSamples",supportSamplesListFileName,"filename with a list of support sample IDs. if not set, all images will be used.",false);
    as->parameter ("verbose", verbose,"get verbose output",false);
    as->parse();
    

    logSetStage("IO");
    logSetVerbosity(verbose);
    bool SAD=!NCC;
    ImagePointerType atlasSegmentationImage=ImageUtils<ImageType>::readImage(atlasSegmentationFilename);
    map<string,ImageInformation> inputImages;
    //    LOG<<"Reading images."<<endl;
    unsigned int totalNumberOfPixels=0;
    std::vector<string> imageIDs;
    int nEdges=0;

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
                    if (imageID!=atlasID){
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
                            nEdges+=interImgEdges;
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
    }
    //LOG<<"Reading deformations."<<endl;
    map< string, map <string, DeformationFieldPointerType> > deformations;
    {
        ifstream ifs(deformationFileList.c_str());
        while (!ifs.eof()){
            string id1,id2,defFileName;
            ifs >> id1;
            if (id1!=""){
                ifs >> id2;
                ifs >> defFileName;
                if (id1 == atlasID || (! useSupportSamples || supportSampleList.find(id1)!=supportSampleList.end() ||  supportSampleList.find(id2)!=supportSampleList.end())){
                    LOGV(3)<<"Reading deformation "<<defFileName<<" for deforming "<<id1<<" to "<<id2<<endl;
                    DeformationFieldPointerType def=ImageUtils<DeformationFieldType>::readImage(defFileName);
                    if (inputImages.find(id1)==inputImages.end() || inputImages.find(id2)==inputImages.end() ){
                        LOG<<id1<<" or "<<id2<<" not in image database, aborting"<<endl;
                        exit(0);
                    }else{
                        deformations[id1][id2]=def;
                    }
                    if (id1!= atlasID && id2 != atlasID){
                        int nDefs=1;
                        for (unsigned int d=0;d<D;++d){
                            nDefs*=def->GetLargestPossibleRegion().GetSize()[d];
                        }
                        nEdges+=nDefs;
                      
                    }

                }
            }
        }
    }

    map<int,int> cliques;
    int clique_max=0;
    LOGV(1)<<"done"<<endl;
    //LOG<<"Setting up pairwise potentials"<<endl;
    int runningIndex=0;
    for (unsigned int n1=0;n1<nImages;++n1){
        string id1=imageIDs[n1];
        if (id1!=atlasID){
            unsigned long int runningIndex2=0;
            for (unsigned int n2=0;n2<nImages;++n2){
                string id2=imageIDs[n2];
                if (id2!=atlasID){
                    if (n1!=n2){
                        if (! useSupportSamples || supportSampleList.find(id2)!=supportSampleList.end()  ){

                            //calculate edges from all pixel of image 1 to their corresponding locations in img2
                            ImagePointerType img1=inputImages[id1].img;
                            //inverse deformation.
                            DeformationFieldPointerType deformation=deformations[id2][id1];
                            ImagePointerType img2=inputImages[id2].img;
                            ImagePointerType deformedI2=TransfUtils<ImageType>::warpImage(img2,deformation);
                            ImageIteratorType img1It(img1,img1->GetLargestPossibleRegion());
                            ImageIteratorType img2It(deformedI2,deformedI2->GetLargestPossibleRegion());
                          
                            DeformationIteratorType defIt(deformation,deformation->GetLargestPossibleRegion());
                            long unsigned int i=runningIndex;
                            defIt.GoToBegin();
                            img1It.GoToBegin();
                            img2It.GoToBegin();
                            ImageType::SizeType size2=img2->GetLargestPossibleRegion().GetSize();
                            ImageType::SizeType size1=img1->GetLargestPossibleRegion().GetSize();
                            for (;!img1It.IsAtEnd();++img1It,++defIt,++img2It,++i){
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
                                LOGV(50)<<inside<<" "<<VAR(id1)<<" "<<VAR(id2)<<" "<<VAR(i)<<" "<<VAR(idx)<<" "<<VAR(linearIndex)<<" "<<VAR(idx2)<<endl;
                                if (inside){
                                    //add bidirectional edge
                                    if (true){
                                        if (cliques.find(i)!=cliques.end()){
                                            int c=cliques[i];
                                            if (cliques.find(linearIndex)!=cliques.end()){
                                                if (c!=cliques[linearIndex]){
                                                    
                                                    LOGV(5)<<VAR(i)<<" "<<VAR(cliques[i])<<" "<<VAR(linearIndex)<<" "<<VAR(cliques[linearIndex])<<endl;
                                                    LOGV(5)<<"relabelling..."<<endl;
                                                    int tmp=cliques[linearIndex];
                                                    LOGV(5)<<VAR(c)<<" -> "<<VAR(tmp)<<endl;
                                                    for (int i=0;i<cliques.size();++i){
                                                        if (cliques[i]==c){
                                                            cliques[i]=tmp;
                                                        }
                                                    }

                                                }
                                            }
                                            else{
                                                cliques[linearIndex]=c;
                                            }
                                        }else if (cliques.find(linearIndex)!=cliques.end()){
                                                int c=cliques[linearIndex];
                                                cliques[i]=c;
                                        }
                                        else{
                                            ++clique_max;
                                            cliques[i]=clique_max;
                                            cliques[linearIndex]=clique_max;
                                        }
                                            
                                      
                                    }
                                    //cout<<VAR(i)<<" "<<VAR(cliques[i])<<" "<<VAR(linearIndex)<<" "<<VAR(cliques[linearIndex])<<endl;
                                    LOGV(5)<<i<<" "<<linearIndex<<endl;
                                }
                            }
                        }
                    }
                    runningIndex2+=inputImages[id2].imageSize;
                }

            }
            runningIndex+=inputImages[id1].imageSize;

        }


    }
    for (int i=0;i<cliques.size();++i){
        cout<<"Clique :"<<cliques[i]<<endl;

    }
    return 1;
}
