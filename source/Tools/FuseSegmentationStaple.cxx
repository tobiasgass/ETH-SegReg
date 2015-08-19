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


ImagePointerType readTxtFile(string fname){
	ImagePointerType result = ImageType::New();
	ImageType::SizeType size;
	size[0] = 100;
	size[1] = 100;
	ImageType::SpacingType space;
	space.Fill(1);
	ImageType::RegionType region;
	region.SetSize(size);
	result->SetRegions(region);
	result->SetSpacing(space);
	result->Allocate();
	ifstream is(fname);
	PixelType v;
	itk::ImageRegionIterator<ImageType> it(result, region);
	it.GoToBegin();
	int count = 0;
	while (count<(size[0]*size[1])){
		is >> v;
		it.Set(v);
		++v;
		++count;
		++it;
		//std::cout << it.GetIndex() << " " << v<< std::endl;
	}
	std::cout << "Read " << count << " pixels for input file " << fname << std::endl;
	is.close();
	return result;
}

void writeTxtFile(FloatImagePointerType img, string fname){

	ofstream of(fname);
	itk::ImageRegionIterator<FloatImageType> it(img, img->GetLargestPossibleRegion());
	it.GoToBegin();
	while (!it.IsAtEnd()){
		of << it.Get() << " " << std::endl;
		++it;
	}
	of.close();
}

int main(int argc, char ** argv)
{

	
    string inFile, outFile;
  
    if (argc<4){
        LOG<<"Usage: FuseSegmentationsStaple <outputFile>  <input1>  <input2> ..."<<endl;
        exit(0);
    }
    
	typedef itk::STAPLEImageFilter<ImageType,FloatImageType> STAPLEFilterType;
	STAPLEFilterType::Pointer stapleFilter = STAPLEFilterType::New();
	stapleFilter->SetForegroundValue(1);

    //accumulate counts
	for (int i = 2; i < argc; ++i){
		ImagePointerType img;
		//img = readTxtFile(argv[i]);
		std::cout << "adding image " << argv[i] << " to staple."<<std::endl;
		img = FilterUtils<ImageType, ImageType>::binaryThresholdingLow(ImageUtils<ImageType>::readImage(argv[i]), 1);
		stapleFilter->SetInput(i - 2,img);
	}
	std::cout << "running staple" << std::endl;
	stapleFilter->Update();
	std::cout << stapleFilter->GetElapsedIterations() << std::endl;
	FloatImagePointerType probImage = stapleFilter->GetOutput();
	//ImageUtils<FloatImageType>::writeImage(argv[1], probImage);
	ImagePointerType result = FilterUtils<FloatImageType, ImageType>::binaryThresholdingLow(probImage, 0.5);
	
	ImageUtils<ImageType>::writeImage(argv[1], result);
	//ImageUtils<FloatImageType>::writeImage(argv[1], probImage);
	//writeTxtFile(probImage, "w.txt");
	return 1;
}
