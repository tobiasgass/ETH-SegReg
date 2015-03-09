#pragma once

#ifdef WITH_CUGMIX
#include "Classifier-Segmentation-Unary-GMM.h"
#endif
#ifdef WITH_RF
#include "Classifier-Segmentation-Unary-RandomForest.h"
#endif

namespace SRS{


template<class ImageType>
class PotentialInstantiations{

private:
#ifdef WITH_CUGMIX
	typedef ClassifierSegmentationUnaryGMM<ImageType> segClassGMMType;
	typedef ClassifierSegmentationUnaryGMMMultilabel<ImageType> segClassMultilabelGMMType;
#endif
#ifdef WITH_RF 
	typedef ClassifierSegmentationUnaryRandomForest<ImageType> segClassRFType;
#endif
};



}//namespace
 
