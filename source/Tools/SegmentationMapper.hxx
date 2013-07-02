#pragma once

#include <boost/bimap.hpp>
#include <itkImage.h>
#include "ImageUtils.h"
#include "Log.h"
template<class ImageType>
class SegmentationMapper{
public:
    
    typedef typename ImageType::Pointer ImagePointerType;
    typedef  boost::bimap<int,int> MapType;
    typedef MapType::value_type MapValueType;
    typedef typename ImageUtils<ImageType>::ImageIteratorType ImageIteratorType;
private:
    MapType m_map;
    int m_nLabels;
public:
    ImagePointerType FindMapAndApplyMap(ImagePointerType input){
 
        ImagePointerType result=ImageUtils<ImageType>::duplicate(input);

        ImageIteratorType inputIt(input,input->GetLargestPossibleRegion());
        ImageIteratorType resultIt(result,input->GetLargestPossibleRegion());

        m_map=MapType();
        m_nLabels=0;
        LOG<<"Mapping segmentation labels to a continuous discrete range... "<<std::endl;
        //build mmap
        for (inputIt.GoToBegin();!inputIt.IsAtEnd();++ inputIt){
            int l=inputIt.Get();
            if (m_map.left.find(l)==m_map.left.end()){
                m_map.insert(MapValueType(l,m_nLabels));
                LOG<<"Mapping label "<<l<<" to "<<m_nLabels<<endl;
                ++m_nLabels;
            }
        }
        for (resultIt.GoToBegin();!resultIt.IsAtEnd();++ resultIt){
            int l=resultIt.Get();
            if (m_map.left.find(l)==m_map.left.end()){
                LOG<<"could not find map for label "<<l<<endl;
            }
            resultIt.Set(m_map.left.find(l)->second);
        }
        LOG<<"Found "<<m_nLabels<<" segmentation labels."<<std::endl;
        return result;
    }

    ImagePointerType ApplyMap(ImagePointerType input){
        ImagePointerType result=ImageUtils<ImageType>::duplicate(input);
        ImageIteratorType resultIt(result,input->GetLargestPossibleRegion());
        for (resultIt.GoToBegin();!resultIt.IsAtEnd();++ resultIt){
            int l=resultIt.Get();
            if (m_map.left.find(l)==m_map.left.end()){
                LOG<<"could not find map for label "<<l<<endl;
            }
            resultIt.Set(m_map.left.find(l)->second);
        }
        return result;
    }

    int GetInverseMappedLabel(int l){
         if (m_map.right.find(l)==m_map.right.end()){
                LOG<<"could not find map for label "<<l<<endl;
         }
        return m_map.right.find(l)->second;
    }
    
    ImagePointerType MapInverse(ImagePointerType input){
        ImagePointerType result=ImageUtils<ImageType>::duplicate(input);
        ImageIteratorType resultIt(result,input->GetLargestPossibleRegion());
        for (resultIt.GoToBegin();!resultIt.IsAtEnd();++ resultIt){
            int l=resultIt.Get();
            if (m_map.right.find(l)==m_map.right.end()){
                LOG<<"could not find map for label "<<l<<endl;
            }
            resultIt.Set(m_map.right.find(l)->second);
        }
           return result;

    }


};//class SegmentationMapper
