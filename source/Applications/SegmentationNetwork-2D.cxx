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
	
typedef unsigned char PixelType;
const unsigned int D=2;
typedef itk::Image<PixelType,D> ImageType;
typedef ImageType::Pointer ImagePointerType;
typedef ImageType::IndexType IndexType;
typedef ImageType::PointType PointType;

typedef ImageType::ConstPointer ImageConstPointerType;
typedef TransfUtils<ImageType>::DisplacementType DisplacementType;
typedef TransfUtils<ImageType>::DeformationFieldType DeformationFieldType;
typedef DeformationFieldType::Pointer DeformationFieldPointerType;
typedef itk::ImageRegionIterator<ImageType> ImageIteratorType;
typedef itk::ImageRegionIterator<DeformationFieldType> DeformationIteratorType;

using namespace std;

struct ImageInformation{
    ImagePointerType img;
    int imageSize;
};
    

int main(int argc, char ** argv)
{
	feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);


    

  
    argstream * as=new argstream(argc,argv);
    string atlasSegmentationFilename,deformationFileList,imageFileList,atlasID="",supportSamplesListFileName="",outputDir=".",outputSuffix="";
    int verbose=0;
    unsigned int nImages=-1;
    double pWeight=1.0;

    (*as) >> parameter ("sa", atlasSegmentationFilename, "atlas segmentation image (file name)", true);
    (*as) >> parameter ("T", deformationFileList, " list of deformations", true);
    (*as) >> parameter ("i", imageFileList, " list of  images, first image is assumed to be atlas image", true);
    (*as) >> parameter ("N", nImages,"number of target images", false);
    (*as) >> parameter ("a", atlasID,"atlas ID. if not set, first image in imageFileList is assumed to be the atlas",false);
    (*as) >> parameter ("w", pWeight,"pairwise potential weight",false);
    (*as) >> parameter ("O", outputDir,"outputdirectory",false);

    (*as) >> parameter ("supportSamples",supportSamplesListFileName,"filename with a list of support sample IDs. if not set, all images will be used.",false);
    (*as) >> parameter ("verbose", verbose,"get verbose output",false);
    (*as) >> help();
    as->defaultErrorHandling();

    logSetStage("IO");
    logSetVerbosity(verbose);
    
    ImagePointerType atlasSegmentationImage=ImageUtils<ImageType>::readImage(atlasSegmentationFilename);
    map<string,ImageInformation> inputImages;
    LOG<<"Reading images."<<endl;
    unsigned int totalNumberOfPixels=0;
    std::vector<string> imageIDs;
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
                    for (unsigned int d=0;d<D;++d){
                        img.imageSize*=img.img->GetLargestPossibleRegion().GetSize()[d];
                    }
                    if (imageID!=atlasID)
                        totalNumberOfPixels+=img.imageSize;
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
        nImages=min(nImages,(unsigned int)(inputImages.size()));
    else
        nImages=inputImages.size();

    LOG<<"Reading deformations."<<endl;
    int nEdges=0;
    map< string, map <string, DeformationFieldPointerType> > deformations;
    {
        ifstream ifs(deformationFileList.c_str());
        while (!ifs.eof()){
            string id1,id2,defFileName;
            ifs >> id1;
            if (id1!=""){
                ifs >> id2;
                ifs >> defFileName;
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

    
  
    logSetStage("Init");
  
    unsigned int nNodes=totalNumberOfPixels;
   

    typedef BGraph<float,float,float> MRFType;
	typedef MRFType::node_id NodeType;
    MRFType* optimizer;
    LOG<<"Allocating MRF with "<<nNodes<<" nodes and "<<nEdges<<" edges."<<endl;
    optimizer = new MRFType(nNodes,nEdges);
    optimizer->add_node(nNodes);
    LOG<<"Setting up unary potentials"<<std::endl;
    unsigned long  int i=0;
    map <string, DeformationFieldPointerType> atlasToTargetDeformations=deformations[atlasID];
    ImagePointerType atlasImage=inputImages[atlasID].img;
    for (unsigned int n1=0;n1<nImages;++n1){
        string id=imageIDs[n1];
        if (id!=atlasID){
            ImagePointerType img=inputImages[id].img;
            DeformationFieldPointerType deformation= atlasToTargetDeformations[id];
            ImagePointerType deformedAtlas=TransfUtils<ImageType>::warpImage(atlasImage,deformation);
            ImagePointerType deformedAtlasSegmentation=TransfUtils<ImageType>::warpImage(atlasSegmentationImage,deformation,true);
            ImageIteratorType imgIt(img,img->GetLargestPossibleRegion());
            ImageIteratorType defAtlasIt(deformedAtlas,deformedAtlas->GetLargestPossibleRegion());
            ImageIteratorType defSegIt(deformedAtlasSegmentation,deformedAtlasSegmentation->GetLargestPossibleRegion());
            for (defAtlasIt.GoToBegin(),defSegIt.GoToBegin(),imgIt.GoToBegin();!imgIt.IsAtEnd();++imgIt,++i,++defAtlasIt,++defSegIt)
                {
                    PixelType segmentationLabel=defSegIt.Get();
                    PixelType imageIntensity=imgIt.Get();
                    PixelType deformedAtlasIntensity=defAtlasIt.Get();
                    double weight=exp(-0.5*fabs(imageIntensity-deformedAtlasIntensity)/30);
                    //one node fore each pixel in each non-atlas image
                    double e0=(segmentationLabel==0)?1:-1;
                    double e1=(segmentationLabel>0)?1:-1;

                    optimizer->add_tweights(i,weight*e0,weight*e1);
                    
                }
        }
    }
    LOGV(1)<<"done"<<endl;
    LOG<<"Setting up pairwise potentials"<<endl;
    unsigned long int runningIndex=0;
    for (unsigned int n1=0;n1<nImages;++n1){
        string id1=imageIDs[n1];
        if (id1!=atlasID){
            unsigned long int runningIndex2=0;
            for (unsigned int n2=0;n2<nImages;++n2){
                string id2=imageIDs[n2];
                if (id2!=atlasID){
                    if (n1!=n2){
                        //calculate edges from all pixel of image 1 to their corresponding locations in img2
                        ImagePointerType img1=inputImages[id1].img;
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
                        ImageType::SizeType size=img2->GetLargestPossibleRegion().GetSize();
                        for (;!img1It.IsAtEnd();++img1It,++defIt,++img2It,++i){
                            IndexType idx=img1It.GetIndex();
                            //compute index in first image to get edge endpoint
                            PointType pt;
                            img1->TransformIndexToPhysicalPoint(idx,pt);
                            pt+=defIt.Get();
                            IndexType idx2;
                            img2->TransformPhysicalPointToIndex(pt,idx2);
                            
                            //check if index is within image bounds
                            int withinImageIndex=0;
                            int dimensionMultiplier=1;
                            bool inside=true;
                            for ( int d=0;inside && d<D;++d){
                                if (idx2[d]>=0 && idx2[d]<size[d]){
                                    withinImageIndex+=dimensionMultiplier*idx2[d];
                                    dimensionMultiplier*= size[d];
                                }else inside=false;
                            }
                            int linearIndex=runningIndex2+withinImageIndex;
                            LOGV(50)<<inside<<" "<<VAR(id1)<<" "<<VAR(id2)<<" "<<VAR(i)<<" "<<VAR(idx)<<" "<<VAR(linearIndex)<<" "<<VAR(idx2)<<endl;
                            if (inside){
                                //compute linear edge index

                                //compute edge weight
                                float weight=exp(-0.5*fabs(img1It.Get()-img2It.Get())/30);
                                //divide weight by (rough) number of edges of the node
                                weight/=(nImages-2);
                                weight*=pWeight;
                                //add bidirectional edge
                                if (weight>0){
                                    optimizer -> add_edge(i,linearIndex,weight,weight);
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
    LOG<<"done"<<endl;
    logSetStage("Optimization");
    LOG<<"starting optimization"<<endl;
    float flow = optimizer -> maxflow();
    LOG<<"Done, resulting energy is "<<flow<< std::endl;
    logSetStage("Finalizing and storing output.");
    i=0;
    for (unsigned int n1=0;n1<nImages;++n1){
        string id1=imageIDs[n1];
        if (id1!=atlasID){
            ImageIteratorType imgIt(inputImages[id1].img,inputImages[id1].img->GetLargestPossibleRegion());
            imgIt.GoToBegin();
            for (;!imgIt.IsAtEnd();++imgIt,++i){
                imgIt.Set(255*(optimizer->what_segment(i)== MRFType::SINK ));
            }
            ostringstream tmpSegmentationFilename;
            tmpSegmentationFilename<<outputDir<<"/segmentation-"<<id1<<"-MRF-nImages"<<nImages<<".png";
            ImageUtils<ImageType>::writeImage(tmpSegmentationFilename.str().c_str(),inputImages[id1].img);
        }
    }
    
    delete optimizer;
    return 1;
}
