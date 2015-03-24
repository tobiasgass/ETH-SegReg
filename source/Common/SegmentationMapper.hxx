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
    typedef typename ImageType::ConstPointer ConstImagePointerType;
    typedef boost::bimap<int,int> MapType;
    typedef MapType::value_type MapValueType;
    typedef typename ImageUtils<ImageType>::ImageIteratorType ImageIteratorType;
    typedef typename ImageUtils<ImageType>::ConstImageIteratorType ConstImageIteratorType;
private:
    MapType m_map;
    int m_nLabels;
public:
    ImagePointerType FindMapAndApplyMap(ImagePointerType input){
        return FindMapAndApplyMap((ConstImagePointerType)input);
    }
    ImagePointerType FindMapAndApplyMap(ConstImagePointerType input){
 
        ImagePointerType result=ImageUtils<ImageType>::duplicateConst(input);

        ConstImageIteratorType inputIt(input,input->GetLargestPossibleRegion());
        ImageIteratorType resultIt(result,input->GetLargestPossibleRegion());

        m_map=MapType();
        m_nLabels=0;
        std::vector<int> labelList(0);
        LOGV(2)<<"Mapping segmentation labels to a continuous discrete range... "<<std::endl;
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
            LOGV(5)<<"Mapping label "<<labelList[n]<<" to "<<n<<std::endl;


        }

        for (resultIt.GoToBegin();!resultIt.IsAtEnd();++ resultIt){
            int l=resultIt.Get();
            if (m_map.left.find(l)==m_map.left.end()){
                LOGV(4)<<"could not find map for label "<<l<<std::endl;
                resultIt.Set(0);
            }else{
                resultIt.Set(m_map.left.find(l)->second);
            }
        }
        LOGV(2)<<"Found "<<m_nLabels<<" segmentation labels."<<std::endl;
        return result;
    }
    void FindMap(ImagePointerType input){
 
        ImagePointerType result=ImageUtils<ImageType>::duplicate(input);

        ConstImageIteratorType inputIt(input,input->GetLargestPossibleRegion());

        m_map=MapType();
        m_nLabels=0;
        std::vector<int> labelList(0);
        LOGV(2)<<"Mapping segmentation labels to a continuous discrete range... "<<std::endl;
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
            LOGV(5)<<"Mapping label "<<labelList[n]<<" to "<<n<<std::endl;


        }

        LOGV(2)<<"Found "<<m_nLabels<<" segmentation labels."<<std::endl;
    }
    ImagePointerType ApplyMap(ImagePointerType input){
        return ApplyMap((ConstImagePointerType)input);
    }
    ImagePointerType ApplyMap(ConstImagePointerType input){
        if (m_map.size() ==0){
            return FindMapAndApplyMap(input);
        }
        ImagePointerType result=ImageUtils<ImageType>::duplicateConst(input);
        ImageIteratorType resultIt(result,input->GetLargestPossibleRegion());
        for (resultIt.GoToBegin();!resultIt.IsAtEnd();++ resultIt){
            int l=resultIt.Get();
             if (m_map.left.find(l)==m_map.left.end()){
                LOG<<"could not find map for label "<<l<<std::endl;
                resultIt.Set(0);
            }else{
                resultIt.Set(m_map.left.find(l)->second);
            }        }
        return result;
    }

    int GetInverseMappedLabel(int l){
         if (m_map.right.find(l)==m_map.right.end()){
                LOG<<"could not find map for label "<<l<<std::endl;
         }
        return m_map.right.find(l)->second;
    }
    
    ImagePointerType MapInverse(ImagePointerType input){
        ImagePointerType result=ImageUtils<ImageType>::duplicate(input);
        ImageIteratorType resultIt(result,input->GetLargestPossibleRegion());
        for (resultIt.GoToBegin();!resultIt.IsAtEnd();++ resultIt){
            int l=resultIt.Get();
            if (m_map.right.find(l)==m_map.right.end()){
                LOGV(3)<<"could not find map for label "<<l<<std::endl;
                resultIt.Set(0);
            }else{
                resultIt.Set(m_map.right.find(l)->second);
            }
        }
           return result;

    }
    int getNumberOfLabels(){return m_nLabels;}

};//class SegmentationMapper
