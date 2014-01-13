#pragma once
#include <algorithm>    // std::sort
#include <boost/bimap.hpp>
#include <itkImage.h>
#include "ImageUtils.h"
#include "Log.h"
template<class ImageType>
class SegmentationMapper{
public:
    
    typedef typename ImageType::Pointer ImagePointerType;
    typedef boost::bimap<int,int> MapType;
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
        std::vector<int> labelList(0);
        LOG<<"Mapping segmentation labels to a continuous discrete range... "<<std::endl;
        //build mmap
        for (inputIt.GoToBegin();!inputIt.IsAtEnd();++ inputIt){
            int l=inputIt.Get();
            if (m_map.left.find(l)==m_map.left.end()){
                m_map.insert(MapValueType(l,m_nLabels));
                labelList.push_back(l);
                ++m_nLabels;
            }
        }
        //order preserving mapping!
        std::sort(labelList.begin(),labelList.end());
        m_map=MapType();
        for (int n=0;n<m_nLabels;++n){
            m_map.insert(MapValueType(labelList[n],n));
            LOGV(1)<<"Mapping label "<<labelList[n]<<" to "<<n<<endl;


        }

        for (resultIt.GoToBegin();!resultIt.IsAtEnd();++ resultIt){
            int l=resultIt.Get();
            if (m_map.left.find(l)==m_map.left.end()){
                LOGV(4)<<"could not find map for label "<<l<<endl;
                resultIt.Set(0);
            }else{
                resultIt.Set(m_map.left.find(l)->second);
            }
        }
        LOG<<"Found "<<m_nLabels<<" segmentation labels."<<std::endl;
        return result;
    }

    ImagePointerType ApplyMap(ImagePointerType input){
        if (m_map.size() ==0){
            return FindMapAndApplyMap(input);
        }
        ImagePointerType result=ImageUtils<ImageType>::duplicate(input);
        ImageIteratorType resultIt(result,input->GetLargestPossibleRegion());
        for (resultIt.GoToBegin();!resultIt.IsAtEnd();++ resultIt){
            int l=resultIt.Get();
             if (m_map.left.find(l)==m_map.left.end()){
                LOGV(4)<<"could not find map for label "<<l<<endl;
                resultIt.Set(0);
            }else{
                resultIt.Set(m_map.left.find(l)->second);
            }        }
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
                LOGV(3)<<"could not find map for label "<<l<<endl;
                resultIt.Set(0);
            }else{
                resultIt.Set(m_map.right.find(l)->second);
            }
        }
           return result;

    }
    int getNumberOfLabels(){return m_nLabels;}

};//class SegmentationMapper
