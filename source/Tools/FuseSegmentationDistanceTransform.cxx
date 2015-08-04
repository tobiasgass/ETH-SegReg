#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "ArgumentParser.h"
#include "ImageUtils.h"
#include <fstream>
#include "itkGaussianImage.h"
#include "itkSTAPLEImageFilter.h"
#include <fstream>

using namespace std;
using namespace itk;

//feraiseexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
typedef  unsigned short int  PixelType;
const unsigned int D = 3;
typedef Image<PixelType, D> ImageType;
typedef  ImageType::IndexType IndexType;
typedef  ImageType::PointType PointType;
typedef  ImageType::DirectionType DirectionType;

typedef ImageType::Pointer ImagePointerType;
typedef ImageType::ConstPointer ImageConstPointerType;
typedef Image<float, D> FloatImageType;
typedef FloatImageType::Pointer FloatImagePointerType;
typedef ImageType::IndexType IndexType;

FloatImagePointerType computeMedianOfImageStack(std::vector<FloatImagePointerType> stack){
	std::vector<itk::ImageRegionConstIterator<FloatImageType> > iterators(stack.size());
	for (int i = 0; i < stack.size(); ++i){
		iterators[i] = itk::ImageRegionConstIterator<FloatImageType>(stack[i], stack[i]->GetLargestPossibleRegion());
		iterators[i].GoToBegin();
	}
	FloatImagePointerType resultImage = ImageUtils<FloatImageType>::createEmpty(stack[0]);
	itk::ImageRegionIterator<FloatImageType> resultIt(resultImage, resultImage->GetLargestPossibleRegion());
	resultIt.GoToBegin();
	while (!resultIt.IsAtEnd()){
		std::vector<float> pixelValueStack(stack.size());
		for (int i = 0; i < stack.size(); ++i){
			pixelValueStack[i] = iterators[i].Get();
			++iterators[i];
		}
		//hodges-lehman: median of all pairwise means
		std:vector<float> hlStack((stack.size()*(stack.size() + 1)) / 2);
		for (int count = 0,  i = 0; i < stack.size(); ++i){
			for (int j = i; j < stack.size(); ++j, ++count){
				hlStack[count] = 0.5* (pixelValueStack[i] + pixelValueStack[j]);
			}
		}
		std::sort(hlStack.begin(), hlStack.end());
		if (hlStack.size() % 2){
			resultIt.Set(-0.5*(hlStack[int(hlStack.size() / 2)] + hlStack[int(hlStack.size() / 2 + 1)]));
		}
		else{
			resultIt.Set(-hlStack[int(hlStack.size() / 2)]);
		}
		++resultIt;
	}
	return resultImage;

}

int main(int argc, char ** argv)
{

	
    string inFile, outFile;
  
    if (argc<4){
        LOG<<"Usage: FuseSegmentationsDistanceTransform <outputFile>  <input1>  <input2> ..."<<endl;
        exit(0);
    }
    
	std::vector<FloatImagePointerType> distanceTransforms(argc - 2);

    //accumulate counts
	double minValue = std::numeric_limits<double>::max();
	for (int i = 2; i < argc; ++i){
		ImagePointerType img;
		
		distanceTransforms[i-2] = FilterUtils<ImageType, FloatImageType>::distanceMapBySignedMaurer(ImageUtils<ImageType>::readImage(argv[i]), 1);
		//get minimum DT value, which is approximally the largest 'thickness' of the structure in question
		double minD = FilterUtils<FloatImageType>::getMin(distanceTransforms[i - 2]);
		if (minD < minValue){
			minValue = minD;
		}
	}

	//threshold DTs such that outside vbalues can not hopelessly overpower inside values 
	//or should this be done locally?
	for (int i = 0; i < argc - 2; ++i){
		distanceTransforms[i] = FilterUtils<FloatImageType>::thresholding(distanceTransforms[i], -std::numeric_limits<double>::max(), -minValue);
	}
	FloatImagePointerType result = computeMedianOfImageStack(distanceTransforms);

	std::string suffix = "distanceTransf.nii"; std::string fname = argv[1];
	ImageUtils<FloatImageType>::writeImage(fname+suffix, result);
	ImagePointerType resultBinary = FilterUtils<FloatImageType, ImageType>::binaryThresholdingLow(result, 0);
	
	//ImageUtils<ImageType>::writeImage(argv[1], result);
	
	ImageUtils<ImageType>::writeImage(argv[1], resultBinary);

	return 1;
}
