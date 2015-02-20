#include <iostream>
#include "ImageUtils.h"
#include "FilterUtils.h"
#include "itkCastImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkApproximateSignedDistanceMapImageFilter.h"
#include "itkFastMarchingImageFilter.h"
#include <map>
#include "itkConnectedComponentImageFilter.h"
#include "itkMultiplyImageFilter.h"
#include "itkLabelShapeKeepNObjectsImageFilter.h"
#include <itkMinimumMaximumImageCalculator.h>
#include <itkHausdorffDistanceImageFilter.h>
#include "ChamferDistanceTransform.h"
#include <map>
#include "ArgumentParser.h"
#include <limits>
#define computeDistances  1


using namespace std;

const unsigned int D=3;
typedef unsigned char Label;
typedef float PixelType;
typedef itk::Image< Label, D >  LabelImage;
typedef itk::Image< float, D > TRealImage;


typedef LabelImage::OffsetType Offset;
typedef LabelImage::IndexType Index;
typedef LabelImage::PointType Point;


TRealImage::Pointer chamferDistance(LabelImage::Pointer image) {
    typedef ChamferDistanceTransform<LabelImage, TRealImage> CDT;
    CDT cdt;
    return cdt.compute(image, CDT::MANHATTEN, true);
}



// convert all non-zero values to 1
LabelImage::Pointer  normalizeImage(LabelImage::Pointer image) {

    itk::ImageRegionIterator<LabelImage> it(
                                            image, image->GetLargestPossibleRegion());

    Label ma=std::numeric_limits<Label>::min();
    Label mi=std::numeric_limits<Label>::max();

    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        Label tmp=it.Get();
        if (tmp>ma) ma=tmp;
        if (tmp<mi) mi=tmp;
    }
    //std::cout<<mi<<" " << ma << std::endl;
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        it.Set((it.Get()-mi));///(ma-mi));
    }
    return image;
}

// convert all non-zero values to 1
LabelImage::Pointer  convertToBinaryImage(LabelImage::Pointer image) {

    itk::ImageRegionIterator<LabelImage> it(
                                            image, image->GetLargestPossibleRegion());
    
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        if (it.Get() >0)
            //        if (it.Get() != 0 && it.Get() > 1000)
            //        if (it.Get() < 1022 || it.Get() > 1024)
            it.Set(1);
        else
            it.Set(0);
    }

    return image;
}

// convert all non-zero values to 1
LabelImage::Pointer  convertToBinaryImage(LabelImage::Pointer image, double threshold) {

    itk::ImageRegionIterator<LabelImage> it(
                                            image, image->GetLargestPossibleRegion());
    
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {

        if (it.Get() >threshold)
            //        if (it.Get() != 0 && it.Get() > 1000)
            //        if (it.Get() < 1022 || it.Get() > 1024)
            it.Set(1);
        else
            it.Set(0);
    }

    return image;
}

// convert all non-zero values to 1
LabelImage::Pointer  convertToBinaryImageFromBroken(LabelImage::Pointer image) {

    itk::ImageRegionIterator<LabelImage> it(
                                            image, image->GetLargestPossibleRegion());

    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        if (it.Get() >0 && it.Get()<numeric_limits<PixelType>::max())
            //        if (it.Get() != 0 && it.Get() > 1000)
            //        if (it.Get() < 1022 || it.Get() > 1024)
            it.Set(1);
        else
            it.Set(0);
    }

    return image;
}
// convert all non-1 values to 0
LabelImage::Pointer  convertToBinaryImageFromMultiLabel(LabelImage::Pointer image) {

    itk::ImageRegionIterator<LabelImage> it(
                                            image, image->GetLargestPossibleRegion());

    if (D==2){
        for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        
            if (it.Get() > numeric_limits<unsigned char>::max()/2  )
                //        if (it.Get() != 0 && it.Get() > 1000)
                //        if (it.Get() < 1022 || it.Get() > 1024)
                it.Set(1);
            else
                it.Set(0);
        
        }
    }else{
        for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
            if (it.Get() ==2){
                it.Set(1);
            }
            else{
                it.Set(0);
            }
        }
    }
    return image;
}
// convert all non-one values to zero
LabelImage::Pointer  convertSegmentedToBinaryImage(LabelImage::Pointer image) {

    itk::ImageRegionIterator<LabelImage> it(
                                            image, image->GetLargestPossibleRegion());
    
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        if (it.Get() == 1)
            //        if (it.Get() != 0 && it.Get() > 1000)
            //        if (it.Get() < 1022 || it.Get() > 1024)
            it.Set(1);
        else
            it.Set(0);
    }

    return image;
}



// change values 0 <-> 1
LabelImage::Pointer  invertBinaryImage(LabelImage::Pointer image) {
    itk::ImageRegionIterator<LabelImage> it(
                                            image, image->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        it.Set(  it.Get() == 0 ? 1 : 0);
    }
    return image;
}



TRealImage::Pointer labelToRealImageCast(LabelImage::Pointer image) {

	typedef itk::CastImageFilter <LabelImage,TRealImage> CastImageFilterType;
	CastImageFilterType::Pointer filter = CastImageFilterType::New();
	filter->SetInput(image);
	filter->Update();
	return filter->GetOutput();
}


Offset computeOffset(
                     LabelImage::Pointer baseImg, LabelImage::Pointer offsetImg
                     ) {

    Index baseStartIdx = baseImg->GetLargestPossibleRegion().GetIndex();
    Point baseStartPoint;
    baseImg->TransformIndexToPhysicalPoint(baseStartIdx, baseStartPoint);

    Index offsetStartIdx;
    offsetImg->TransformPhysicalPointToIndex(baseStartPoint, offsetStartIdx);

    return offsetStartIdx - baseStartIdx;

}


TRealImage::Pointer jointDistances(
                                   TRealImage::Pointer img1, TRealImage::Pointer img2
                                   ){

    itk::ImageRegionIteratorWithIndex<TRealImage> it(
                                                     img1, img1->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {

        if (it.Get()==0)
            it.Set(-img2->GetPixel(it.GetIndex()));

    }

    return img1;
}


struct Stats {

    unsigned falsePos ;
    unsigned falseNeg;
    unsigned truePos;
    unsigned trueNeg;

    Stats()
        : falsePos(0)
        , falseNeg(0)
        , truePos(0)
        , trueNeg(0)
    {}

    void add(bool truthCondiction, bool testResult) {

        if (testResult && truthCondiction)
            truePos++;

        else if (testResult && !truthCondiction)
            falsePos++;

        else if (!testResult && !truthCondiction)
            trueNeg++;

        else if (!testResult && truthCondiction)
            falseNeg++;
    }

    unsigned total() {
        return truePos + trueNeg + falsePos + falseNeg;
    }

};


int main(int argc, char * argv [])
{


    ArgumentParser as(argc, argv);
	string groundTruth,segmentationFilename,outputFilename="";
    bool hausdorff=false;
    double threshold=-9999999;
    bool convertFromClassified=false;
    bool broken=false;
    bool multilabel=false;
    bool connectedComponent=false;
	as.parameter ("g", groundTruth, "groundtruth image (file name)", true);
	as.parameter ("s", segmentationFilename, "segmentation image (file name)", true);
	as.parameter ("o", outputFilename, "output image (file name)", false);
	as.parameter ("h", hausdorff, "compute hausdorff distance(0,1)", false);
	as.parameter ("t", threshold, "threshold segmentedImage (threshold)", false);
	as.parameter ("c", convertFromClassified, "convert from classified segmentation (after normalization) (0,1)", false);
	as.option ("b", broken, "convert from broken segmentation");
	as.option ("m", multilabel, "convert from multilabel segmentation");
	as.option ("l", connectedComponent, "use largest connected component in segmentation");

	as.parse();
	

    typedef itk::HausdorffDistanceImageFilter<LabelImage, LabelImage> HausdorffDistanceFilterType;
    typedef HausdorffDistanceFilterType::Pointer HDPointerType;
    HDPointerType hdFilter=HausdorffDistanceFilterType::New();;
    LabelImage::Pointer groundTruthImg =
        ImageUtils<LabelImage>::readImage(groundTruth);
    LabelImage::Pointer segmentedImg =
        ImageUtils<LabelImage>::readImage(segmentationFilename);

    
 
    unsigned totalPixels = 0;

    //    groundTruthImg=normalizeImage(groundTruthImg);
    convertToBinaryImage(groundTruthImg);
    TRealImage::Pointer distancesOutsideTruthBone;
    TRealImage::Pointer distancesInsideTruthBone;
    TRealImage::Pointer distanceMap;
    float maxAbsDistance = 0;
    float maxDistance = -std::numeric_limits<float>::max();
    double minSum=0,maxSum=0;
    int minDistance = std::numeric_limits<int>::max();
    double minCount=0, maxCount=0;
    int sum = 0;
    unsigned totalEdges = 0;
    float mean=0;
 
    if (broken){
        segmentedImg=convertToBinaryImageFromBroken(segmentedImg);
    }
    else if (multilabel){
        segmentedImg=convertToBinaryImageFromMultiLabel(segmentedImg);
    }
    else if (convertFromClassified)
        segmentedImg=convertSegmentedToBinaryImage(segmentedImg);
    else{
        //segmentedImg= normalizeImage ( segmentedImg) ;
        if (threshold!=-9999999){
            segmentedImg= convertToBinaryImage (segmentedImg, threshold);         
        }else{
            //            segmentedImg=normalizeImage(segmentedImg);
            segmentedImg= convertToBinaryImage ( segmentedImg) ;
        }
    }    
    typedef LabelImage::ConstPointer ConstType;

    if (connectedComponent){  
        typedef itk::MinimumMaximumImageCalculator <LabelImage>
            ImageCalculatorFilterType;
        typedef itk::ConnectedComponentImageFilter<LabelImage,LabelImage>  ConnectedComponentImageFilterType;
        ConnectedComponentImageFilterType::Pointer filter =
            ConnectedComponentImageFilterType::New();
        filter->SetInput(segmentedImg);
        filter->Update();
    
        typedef itk::LabelShapeKeepNObjectsImageFilter< LabelImage > LabelShapeKeepNObjectsImageFilterType;
        LabelShapeKeepNObjectsImageFilterType::Pointer labelShapeKeepNObjectsImageFilter = LabelShapeKeepNObjectsImageFilterType::New();
        labelShapeKeepNObjectsImageFilter->SetInput( filter->GetOutput() );
        labelShapeKeepNObjectsImageFilter->SetBackgroundValue( 0 );
        labelShapeKeepNObjectsImageFilter->SetNumberOfObjects( 1);
        labelShapeKeepNObjectsImageFilter->SetAttribute( LabelShapeKeepNObjectsImageFilterType::LabelObjectType::NUMBER_OF_PIXELS);
        labelShapeKeepNObjectsImageFilter->Update();
      
        
      
        segmentedImg = convertToBinaryImage (labelShapeKeepNObjectsImageFilter->GetOutput(), 0.1);     //;//f->GetOutput();// FilterUtils<LabelImage>::binaryThresholding(labelShapeKeepNObjectsImageFilter->GetOutput(),1,10000);//filter->GetOutput(),1,1);
    }
    if (outputFilename!=""){
        if (D==2){
            typedef itk::MultiplyImageFilter<LabelImage> MultiplyImageFilterType;
            MultiplyImageFilterType::Pointer f=MultiplyImageFilterType::New();
            f->SetConstant(255);
            f->SetInput(segmentedImg);
            f->Update();
            ImageUtils<LabelImage>::writeImage(outputFilename,(ConstType)f->GetOutput());
        }else{
            ImageUtils<LabelImage>::writeImage(outputFilename,(ConstType)segmentedImg);
        }
    }
   
    if (hausdorff){
        hdFilter->SetInput1(groundTruthImg);
        hdFilter->SetInput2(segmentedImg);
        hdFilter->SetUseImageSpacing(true);
        hdFilter->Update();
        mean=hdFilter->GetAverageHausdorffDistance();
        maxAbsDistance=hdFilter->GetHausdorffDistance();
    }
    if (hausdorff){

        distancesOutsideTruthBone = chamferDistance(groundTruthImg);
        distancesInsideTruthBone = chamferDistance(invertBinaryImage(groundTruthImg));
        invertBinaryImage(groundTruthImg);
        distanceMap =
            jointDistances(distancesOutsideTruthBone, distancesInsideTruthBone);
    }
   
    //invertBinaryImage(groundTruthImg);


    // ImageUtils<TRealImage>::writeImage("/home/marcel/tmp/s3.nii", distanceMap);

    Stats glob, narrowBand;

    Offset offset = computeOffset(groundTruthImg, segmentedImg);
    //std::cerr << offset << " " << std::endl; 

    double globalAbsSum;
    double globalAbsSumSquares;
    map<int,unsigned> dists;
    float hausdorf=-1;
    itk::ImageRegionIteratorWithIndex<LabelImage> itTruth(
                                                          groundTruthImg, groundTruthImg->GetLargestPossibleRegion());
    for (itTruth.GoToBegin(); !itTruth.IsAtEnd();  ++itTruth) {

        bool truthBone = ( itTruth.Get() == 1);
        bool segmBone = ( segmentedImg->GetPixel(itTruth.GetIndex() + offset) == 1);
        glob.add(truthBone, segmBone);
        totalPixels++;

        if (hausdorff){
            float distFromBone = distanceMap->GetPixel(itTruth.GetIndex());

            if (abs(distFromBone) <= 3.01)
                narrowBand.add(truthBone, segmBone);


            // Tobi, as far as i remember, the haussdorf is computed here

            // if this pixel belongs to a bone
            if (segmBone) {

                // then i examine all neigbours
                // THIS ONLY WORKS FOR straight 90 degr. boundaries. If boundaries are curved or have fractional angles multiple neighbouring pixels can be tissue and therefore the result exaggerates the distances at this point.
                // It would be correct to look for the neighboring pixel normal to the bone surface.
                // the best approximation would be to take the maximal distance over all tissue neighbors, which is also not correct, though
                float localMax=-1;
                bool isBoundary=false;
                for (int dim=0; dim<D; ++dim)
                    for (int off=-1; off<=1; off+=2) {

                        Index idx = itTruth.GetIndex();
                        idx[dim]+=off;

                        // and i check whether the neighbourhooding pixel is a tissue or a bone
                        bool isNeighBone = (segmentedImg->GetPixel(idx+offset) == 1);
                        double localDist;
                        if (!isNeighBone) {
                            
                            // the neighbourhood is a tissue, which means i am on a bone boundary
                            isBoundary=true;
                            // let's take the distance from the neighbourhooding pixel to the true bone boundary
                            float distNeigh = distanceMap->GetPixel(idx);

                            // this IF is a special case when the segmentation in this place is perfect:
                            // in this case neighbourhood is outside (distance == +1) and the central
                            // pixel is insde (distance == -1) or vice versa so the sum is zero
                            if (abs(distFromBone+distNeigh) < 0.0001) {
                                //                          dists[0]++;
                                localDist=0;
                            } else{

                                // otherwise i am taking the minimum. This implementation here is crappy,
                                // because dists is a map from int to int. I wanted to replace it. If you
                                // are looking for the hausdorf only, take the code below
                                //dists[min(distFromBone,distNeigh)]++;
                                localDist=min(distFromBone,distNeigh);
                            }

                            if (abs(localDist)>localMax)
                                localMax=localDist;
                            //  Use this for precise Hausdorf

                            float distanceFromBoneBoundary = (abs(distFromBone+distNeigh) < 0.0001 ) ? 0 : abs(min(distFromBone,distNeigh));
                            hausdorf =distanceFromBoneBoundary>hausdorf?distanceFromBoneBoundary:hausdorf;
                        }
                    }
                if (isBoundary){
                    dists[localMax]++;
                    globalAbsSum+=abs(localMax);
                    globalAbsSumSquares+=(localMax)*(localMax);
                    // std::cout<<hausdorf<<" "<<localMax<<std::endl;
                    hausdorf=max(hausdorf,localMax);
                }
            }

        }

    }



    assert(totalPixels == glob.total());

    float precisionG = (glob.truePos+glob.trueNeg) / (glob.total() / 100.0);
   

    float precision   = float(glob.truePos)/(glob.truePos+glob.falsePos);
    float recall      = float(glob.truePos)/(glob.truePos+glob.falseNeg);
    float specificity = float(glob.trueNeg)/(glob.trueNeg+glob.falsePos);
    float accuracy    = float(glob.truePos+glob.trueNeg)/(glob.total());
    float f1          = 2*(precision*recall)/(precision+recall);
    std::cout<<" prec "<< precision;
    std::cout<<" recall "<< recall;
    std::cout<<" specificity "<< specificity;
    std::cout<<" accuracy "<< accuracy;
    std::cout<<" f1 "<< f1;
    std::cout<<" HD "<<-99;

    std::cout<<" TP "<< glob.truePos;//;
    std::cout<<" TN "<< glob.trueNeg;
    std::cout<<" FP "<< glob.falsePos;
    std::cout<<" FN "<< glob.falseNeg;
    std::cout<<" PM "<< precisionG ;
    float var;

    if (hausdorff){
#if 0
        float precisionNB = float(narrowBand.truePos) / (narrowBand.truePos + narrowBand.falsePos) ;
        float recallNB    = float(narrowBand.truePos) / (narrowBand.truePos + narrowBand.falseNeg) ;
        float f1NB= 2*(precisionNB*recallNB)/(precisionNB+recallNB);
    
        std::cout<<" NB-f1 "<<f1NB;

      
        map<unsigned, unsigned> distribFunc;

        map<int,unsigned>::const_iterator it;
        for ( it = dists.begin() ; it != dists.end(); it++ ) {

            int distance = it->first;
            int count = it->second;

            totalEdges+= count;
            sum += count * distance;

            // update the partial cumulative distribution function
            if (abs(distance) > 2.99) distribFunc[3]+=count;
            if (abs(distance)> 4.99) distribFunc[5]+=count;
            if (abs(distance) > 9.99) distribFunc[10]+=count;


            if (abs(distance) > abs(maxAbsDistance))
                maxAbsDistance = distance;
            if ((distance) > (maxDistance))
                maxDistance = distance;
            if ((distance) < (minDistance))
                minDistance = distance;
            if (distance<0){
                minSum+=distance*count;
                minCount+=count;

            }else{
                maxSum+=distance*count;
                maxCount+=count;
            }

            //fprintf(stderr, "%d %d\n", distance, count);
        }

        mean = sum/(float)totalEdges;

        // compute variance
        var = 0;
        for ( it = dists.begin() ; it != dists.end(); it++ ) {
            int distance = it->first;
            int count = it->second;

            var += count * pow(distance - mean,2);
        }
        var /= (totalEdges - 1);

     

#endif
        std::cout<<"  Mean "<< mean;
        std::cout<<" Std "<< sqrt(var);
        std::cout<<" MaxAbs "<< maxAbsDistance;
        std::cout<<" MarcelHD "<< hausdorf;
        //std::cout<<" Dist3 "<< distribFunc[3];
        //std::cout<<" Dist5 "<< distribFunc[5];
        //std::cout<<" Dist10 "<< distribFunc[10];
        std::cout<<" max "<< maxDistance;
        std::cout<<" min "<< minDistance;

        if (maxCount>0) maxSum/=maxCount;
        if (minCount>0) minSum/=minCount;
        std::cout<<" avgPos "<<maxSum;
        std::cout<<" avgNed "<<minSum;
        //        std::cout<<" edges "<< totalEdges <<" true bone count "<< glob.truePos+ glob.falseNeg;
        std::cout<<" l1Avg "<< globalAbsSum/totalEdges;
        std::cout<<" l2Norm "<< sqrt(globalAbsSumSquares/totalEdges);

    } //old hausdoirff

    std::cout<< std::endl;
    // std::cout<<"EvalG - % of bone segmented "<< float(glob.truePos) / ((glob.truePos + glob.falseNeg) / 100)<< std::endl;

    

  


	return EXIT_SUCCESS;
}
