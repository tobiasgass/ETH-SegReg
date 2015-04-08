#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "ArgumentParser.h"
#include "ImageUtils.h"
#include <fstream>
#include "itkGaussianImage.h"
#include "Metrics.h"
using namespace std;
using namespace itk;


int main(int argc, char ** argv)
{

	//feraiseexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    typedef  short  PixelType; 
    typedef  int  LabelType;
    const unsigned int D=3;
    typedef Image<PixelType,D> ImageType;
    typedef Image<LabelType,D> LabelImageType;
    typedef  ImageType::IndexType IndexType;
    typedef  ImageType::PointType PointType;
    typedef  ImageType::DirectionType DirectionType;
    typedef  ImageType::SpacingType SpacingType;

    typedef ImageType::Pointer ImagePointerType;
    typedef LabelImageType::Pointer LabelImagePointerType;
    typedef ImageType::ConstPointer ImageConstPointerType;
    typedef Image<float,D> FloatImageType;
    typedef FloatImageType::Pointer FloatImagePointerType;
    typedef ImageType::IndexType IndexType;
    string inFile, outFile;
  
    if (argc<4){
        LOG<<"Usage: FuseSegmentationsMaxVote3D <outputFile> <targetImage> <deformedImage1> <seg1>  <deformedImage2> <input2> ..."<<endl;
        LOG<<argv<<endl;
        exit(0);
    }
    
    
    ImagePointerType targetImage = ImageUtils<ImageType>::readImage(argv[2]);
    ImagePointerType downSampledTargetImage=FilterUtils<ImageType>::LinearResample(targetImage,0.15,true);
    int numberOfPixels=targetImage->GetRequestedRegion().GetNumberOfPixels();

    LOG<<"Imagedownsampling actual :"<<1.0*downSampledTargetImage->GetRequestedRegion().GetNumberOfPixels()/numberOfPixels<<" nu,ber of pixels in downsampled image :"<<downSampledTargetImage->GetRequestedRegion().GetNumberOfPixels() <<endl;

  

    LabelImagePointerType fullResResultImage=LabelImageType::New(); //=FilterUtils<ImageType,LabelImageType>::cast(targetImage);
    fullResResultImage->SetRegions(targetImage->GetRequestedRegion());
    fullResResultImage->SetDirection(targetImage->GetDirection());
    fullResResultImage->SetSpacing(targetImage->GetSpacing());
    fullResResultImage->SetOrigin(targetImage->GetOrigin());
    
    int maxNumberOfPixels=0.1*224657408;
    
    float resamplingFactor=min(1.0,pow(1.0*maxNumberOfPixels/numberOfPixels,1.0/D));

    LabelImagePointerType resultImage=FilterUtils<LabelImageType>::EmptyResample(fullResResultImage,resamplingFactor);
    LOG<<"Theoretical :"<<VAR(resamplingFactor)<<"; practical: "<<1.0*resultImage->GetLargestPossibleRegion().GetNumberOfPixels()/numberOfPixels<<endl;

    targetImage=NULL;

    LOG<<"allocating counts structure"<<endl;
    typedef  std::vector< map<LabelType,float > * > VoteMapType;
    VoteMapType votes(numberOfPixels,NULL);
    std::vector< float > backGroundVotes(numberOfPixels,0);
    LOG<<"done allocating counts structure"<<endl;
    //for relative weighing
    map<LabelType,unsigned char > totalLabelCountPerImage;
    
    typedef  map<LabelType,unsigned char >::iterator CountMapIterator;
    
    //accumulate counts
    for (int i=3;i<argc;i+=2){
        ImagePointerType deformedAtlasImage=FilterUtils<ImageType>::LinearResample(ImageUtils<ImageType>::readImage(argv[i]),downSampledTargetImage,true);
        LOG<<"Reading img "<<argv[i]<<endl;
        LabelImagePointerType img = FilterUtils<LabelImageType>::NNResample(ImageUtils<LabelImageType>::readImage(argv[i+1]),resultImage,false);
        FloatImagePointerType lncc=Metrics<ImageType,FloatImageType>::efficientLNCC(downSampledTargetImage,deformedAtlasImage,10,10);

        ImageUtils<LabelImageType>::ImageIteratorType it2(img,img->GetRequestedRegion());

        typedef  itk::LinearInterpolateImageFunction<FloatImageType, double> ImageInterpolatorType;
        ImageInterpolatorType::Pointer interpolator=ImageInterpolatorType::New();
        interpolator->SetInputImage(lncc);
        map<LabelType,unsigned char > labelCountPerImage;

        it2.GoToBegin();      
        for (int c=0;!it2.IsAtEnd();++it2,++c){
            LabelType val2=it2.Get();
            PointType pt;
            resultImage->TransformIndexToPhysicalPoint(it2.GetIndex(),pt);
            float weight=interpolator->Evaluate(pt)+std::numeric_limits<float>::epsilon();
            if (val2==0){
                backGroundVotes[c]+=weight;
            }else{
                if (votes[c] == NULL){
                    votes[c]=new  map<LabelType,float >;
                }
                (*(votes[c]))[val2]+=weight;
            }
            if (labelCountPerImage[val2] == 0) labelCountPerImage[val2]=1;
        }

        CountMapIterator countMapIt=labelCountPerImage.begin();
        for (;countMapIt!=labelCountPerImage.end();++countMapIt){
            ++totalLabelCountPerImage[countMapIt->first];
            LOG<<VAR(int(countMapIt->first))<<" "<<VAR(int(totalLabelCountPerImage[countMapIt->first]))<<endl;
        }
    }

    int maxCount=1;
    CountMapIterator countMapIt=totalLabelCountPerImage.begin();
    for (;countMapIt!=totalLabelCountPerImage.end();++countMapIt){
        LOG<<VAR(int(countMapIt->second)) <<" "<<VAR(countMapIt->first)<<endl;
        maxCount=max(maxCount,int(countMapIt->second));
    }
    
    downSampledTargetImage=NULL;
    resultImage->Allocate();
    
    ImageUtils<LabelImageType>::ImageIteratorType it2(resultImage,resultImage->GetRequestedRegion());
    it2.GoToBegin();
    for (int c=0;!it2.IsAtEnd();++it2,++c){
        float maxVote=1.0*backGroundVotes[c]/maxCount; int maxLabel=0;
        if (votes[c]){
            map<LabelType,float >::iterator mapIt=(*(votes[c])).begin();
            for (;mapIt!=votes[c]->end();++mapIt){
                float vote=1.0*mapIt->second/(totalLabelCountPerImage[mapIt->first]);
                if (vote>maxVote){
                    maxVote=vote;
                    maxLabel=mapIt->first;
                }
            }
            delete votes[c];
        }
        it2.Set(maxLabel);
    }

    ImageUtils<LabelImageType>::writeImage(argv[1],FilterUtils<LabelImageType>::NNResample(resultImage,fullResResultImage,false));

	return 1;
}
