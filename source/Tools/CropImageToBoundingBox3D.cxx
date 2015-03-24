
// Software Guide : BeginCodeSnippet
#include "itkImageMaskSpatialObject.h"
#include "itkImage.h"
// Software Guide : EndCodeSnippet

#include "itkImageFileReader.h"
#include "ImageUtils.h"
#include "FilterUtils.hpp"
#include "itkCropImageFilter.h"

int main( int argc, char * argv [] )
{
    double marginInMM=50;
  if ( argc < 3 )
    {
    std::cerr << "Missing Parameters " << std::endl;
    std::cerr << "Usage: " << argv[0];
    std::cerr << " inputBinaryImageFile  ";
    std::cerr << " outputBinaryImageFile  ";
    std::cerr << " <margin in mm(optional)  ";
    return EXIT_FAILURE;
    }
  if (argc>3)
      marginInMM=atof(argv[3]);
  const unsigned int D=3;
  typedef int PixelType;
  typedef itk::ImageMaskSpatialObject<D>      ImageMaskSpatialObject;
  typedef itk::Image<PixelType,D> ImageType;
  typedef ImageMaskSpatialObject::ImageType   BinaryImageType;
  typedef ImageType::RegionType               RegionType;
  typedef ImageType::Pointer               ImagePointerType;
  typedef BinaryImageType::Pointer               BinaryImagePointerType;


  ImagePointerType img=ImageUtils<ImageType>::readImage(argv[1]);
  BinaryImagePointerType bImg=FilterUtils<ImageType,BinaryImageType>::binaryThresholdingLow(img,1);
  //bImg=FilterUtils<BinaryImageType>::dilation(bImg,100);
  
  ImageMaskSpatialObject::Pointer maskSO = ImageMaskSpatialObject::New();

  maskSO->SetImage ( bImg );

  RegionType boundingBoxRegion  = maskSO->GetAxisAlignedBoundingBoxRegion();

  std::cout << "Bounding Box Region: " << boundingBoxRegion << std::endl;
  typedef ImageType::SizeType SizeType;
  SizeType upperCropSize,lowerCropSize;
  for (int d=0;d<D;++d){
      upperCropSize[d]=img->GetLargestPossibleRegion().GetSize()[d]-((boundingBoxRegion.GetIndex()[d])+boundingBoxRegion.GetSize()[d]);
      lowerCropSize[d]=boundingBoxRegion.GetIndex()[d];
      //add some margin
     
      int marginInPix=marginInMM/img->GetSpacing()[d];
      std::cout<<upperCropSize[d]<<" "<<marginInPix<<std::endl;
      upperCropSize[d]=std::max(0,int(upperCropSize[d]-marginInPix));
      lowerCropSize[d]=std::max(0,int(lowerCropSize[d]-marginInPix));;
      
  }
  typedef itk::CropImageFilter <ImageType, ImageType>
      CropImageFilterType;
 
  CropImageFilterType::Pointer cropFilter
    = CropImageFilterType::New();
  cropFilter->SetInput(img);
 
  cropFilter->SetUpperBoundaryCropSize(upperCropSize);
  cropFilter->SetLowerBoundaryCropSize(lowerCropSize);
  cropFilter->Update();
  img=cropFilter->GetOutput();
  ImageType::IndexType zero;
  zero.Fill(0);
  ImageType::RegionType r = img->GetLargestPossibleRegion();
  r.SetIndex(zero);
  img->SetLargestPossibleRegion(r);
  //std::cout<<img->GetLargestPossibleRegion()<<endl;
  ImageUtils<ImageType>::writeImage(argv[2],img);
  return EXIT_SUCCESS;
}
