#pragma once


#include "ImageUtils.h"
#include "FilterUtils.hpp"
#include <vector>
#include <itkImageAlgorithm.h>

using namespace std;

template <class ImageType>
class TemporalMedianImageFilter{

    typedef typename ImageType::Pointer ImagePointerType;
    typedef typename ImageType::PixelType PixelType;
    static const int D=ImageType::ImageDimension;

    //typedef itk::Vector<PixelType,SequenceLength> MedianFilterPixelType;
    typedef itk::VectorImage<float,D> MedianFilterImageType;
    typedef typename MedianFilterImageType::PixelType  MedianFilterPixelType;

    typedef typename MedianFilterImageType::Pointer MedianFilterImagePointerType;
    
    typedef typename ImageUtils<MedianFilterImageType>::ImageIteratorType MedianFilterImageIteratorType;
    typedef typename ImageUtils<ImageType>::ImageIteratorType ImageIteratorType;
    
    
private:
    
    MedianFilterImagePointerType m_medianFilterImage;
    int m_sequenceLength,m_sequenceCount;
    MedianFilterImageIteratorType m_medianIterator;
    ImagePointerType  m_medianImage;
    
public:
    TemporalMedianImageFilter(){}
    TemporalMedianImageFilter(ImagePointerType img, int sequenceLength){
        m_sequenceLength=sequenceLength;
      
        m_medianFilterImage=MedianFilterImageType::New();
        m_medianFilterImage->SetRegions(img->GetLargestPossibleRegion());
        m_medianFilterImage->SetOrigin(img->GetOrigin());
        m_medianFilterImage->SetDirection(img->GetDirection());
        m_medianFilterImage->SetSpacing(img->GetSpacing());
        m_medianFilterImage->SetVectorLength(m_sequenceLength);
        m_medianFilterImage->Allocate();
      

        m_sequenceCount=0;


        m_medianIterator=MedianFilterImageIteratorType(m_medianFilterImage,m_medianFilterImage->GetLargestPossibleRegion());

        for (m_medianIterator.GoToBegin();!m_medianIterator.IsAtEnd();++m_medianIterator){
            
            MedianFilterPixelType  newVec=MedianFilterPixelType(m_sequenceLength);
            newVec.Fill(-1);
            m_medianIterator.Set(newVec);
         
        }
        m_medianImage=ImageUtils<ImageType>::createEmpty(img);
    }

    ~TemporalMedianImageFilter(){
//         for (m_medianIterator.GoToBegin();!m_medianIterator.IsAtEnd();++m_medianIterator){
//             delete m_medianIterator.Get();
//         }
//         delete m_medianIterator;
//
     }
    

    void insertImage(ImagePointerType img){
        if (  m_sequenceCount>=m_sequenceLength){
            LOG<<"too many images inserted! "<<endl;
            exit(0);
        }
        LOG<<VAR(m_medianFilterImage->GetLargestPossibleRegion().GetSize())<<endl;
        ImageIteratorType imgIt(img,img->GetLargestPossibleRegion());
        imgIt.GoToBegin();
        
        for (m_medianIterator.GoToBegin();!m_medianIterator.IsAtEnd();++m_medianIterator,++imgIt){

            MedianFilterPixelType  px=m_medianIterator.Get();
            insertValue(imgIt.Get(),px);
            m_medianIterator.Set(px);
            
        }
        ++m_sequenceCount;
    }

    ImagePointerType getMedian(){
        ImageIteratorType imgIt(m_medianImage,m_medianImage->GetLargestPossibleRegion());
        imgIt.GoToBegin();
        for (m_medianIterator.GoToBegin();!m_medianIterator.IsAtEnd();++m_medianIterator,++imgIt){
            MedianFilterPixelType m=m_medianIterator.Get();

            PixelType localMedian=m[ int((0.51+m_sequenceCount)/2)];
            imgIt.Set(localMedian);
        }
        return m_medianImage;
    }

    void insertValue(PixelType p,MedianFilterPixelType  & medianVector){
        if (p<0){
            LOG<<"ERROR, median filter is setup to only handle positive values!" <<endl;
        }
        //insertion sort :/
        int i=0;
        for (;i<m_sequenceLength;++i){
            if (p>(medianVector)[i]){
                break;
            }
        }
        PixelType tmp=(medianVector)[i];
        (medianVector)[i]=p;
        for (int j=i+1;j<m_sequenceLength-1;++j){
            PixelType tmp2=(medianVector)[j];
            (medianVector)[j]=tmp;
            tmp=tmp2;
        }
    }


};
