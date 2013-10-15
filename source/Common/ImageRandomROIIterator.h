#pragma once

#include <vector>
#include <itkImageRegionConstIterator.h>
#include <itkImageRandomConstIterator.h>

template<class ImageType>
class ImageRandomROIIterator{
    typedef typename ImageType::Pointer ImagePointerType;
    typedef typename ImageType::PixelType PixelType;
    typedef typename itk::ImageRegionConstIteratorWithIndex< ImageType > ConstImageIteratorType;


public:
    
    ImageRandomROIIterator(ImagePointerType inputImage, ImagePointerType ROI, int nSamples){
        
        
        


    }
    
    PixelType Get(){
        
    }

    void operator++(){

    }

    void GoToBegin(){

    }
    void IsAtEnd(){

    }

};
