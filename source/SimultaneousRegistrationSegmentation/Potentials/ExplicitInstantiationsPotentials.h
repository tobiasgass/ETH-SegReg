#pragma once

#ifdef WITH_CUGMIX
#include "GMMClassifier.h"
#endif
#ifdef WITH_RF
#include "RandomForestClassifier.h"
#endif

namespace SRS{


template<class ImageType>
  class PotentialInstantiations{
  
 private:
#ifdef WITH_CUGMIX
  typedef SegmentationGMMClassifier<ImageType> segClassGMMType;
  typedef MultilabelSegmentationGMMClassifier<ImageType> segClassMultilabelGMMType;
#endif
#ifdef WITH_RF 
  typedef SegmentationRandomForestClassifier<ImageType> segClassRFType;
#endif
 };



}//namespace
