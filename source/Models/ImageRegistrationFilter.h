/*
 * ImageRegistrationFilter.h
 *
 *  Created on: Nov 8, 2010
 *      Author: gasst
 */

#ifndef IMAGEREGISTRATIONFILTER_H_
#define IMAGEREGISTRATIONFILTER_H_

template<class ImageType>
class ImageRegistrationFilter{

public:
	typedef typename ImageType::Pointer PImage;
	ImageRegistrationFilter(PImage fixedImage, PImage movingImage, int maxDisplacement=5, int displacementSampling=5);

private:
	PImage m_fixedImage,m_movingImage;


};

#endif /* IMAGEREGISTRATIONFILTER_H_ */
