/*=========================================================================

  Program: Automatic Segmentation Of Bones In 3D-CT Images

  Author:  Marcel Krcah <marcel.krcah@gmail.com>
  Computer Vision Laboratory
  ETH Zurich
  Switzerland

  Date:    2010-09-01

  Version: 1.0

  =========================================================================*/


#pragma once


#include "itkImage.h"
#include "ImageUtils.h"
#include "FilterUtils.hpp"
#include "SheetnessFilter.h"
#include "boost/tuple/tuple.hpp"
#include "Log.h"
#include <vector>
#include "itkConnectedComponentImageFilter.h"
using namespace boost;


template<class InputImage>
class Preprocessing {
public:
    typedef typename InputImage::Pointer  InputImagePointer;
    typedef typename InputImage::ConstPointer  ConstInputImagePointer;
    
    typedef typename InputImage::PixelType InputImagePixelType;
    typedef typename InputImage::IndexType InputImageIndex;
    typedef typename InputImage::RegionType InputImageRegion;
    
    typedef typename ImageUtils<InputImage>::FloatImageType FloatImage;
    typedef typename FloatImage::Pointer FloatImagePointer;

    typedef typename ImageUtils<InputImage>::UCharImageType UCharImage;
    typedef typename UCharImage::Pointer UCharImagePointer;
    typedef typename ImageUtils<InputImage>::UIntImageType UIntImage;
    typedef typename UIntImage::Pointer UIntImagePointer;

public:
    // compute multiscale sheetness measure
    // if roi is specified, than compute the measure only for pixels within ROI
    // (i.e. pixels where roi(pixel) >= 1)
    static FloatImagePointer multiscaleSheetness(
                                                 FloatImagePointer img, std::vector<float> scales
                                                 ) {

        assert(scales.size() >= 1);

        FloatImagePointer multiscaleSheetness;

        for (unsigned i = 0; i < scales.size(); ++i) {

            LOG<<"Computing single-scale sheetness, sigma="<< scales[i]<<std::endl;

            MemoryEfficientObjectnessFilter *sheetnessFilter =
                new MemoryEfficientObjectnessFilter();
            sheetnessFilter->SetImage((img));
            sheetnessFilter->SetAlpha(0.5);
            sheetnessFilter->SetBeta(0.5);
            sheetnessFilter->SetSigma(scales[i]);
            sheetnessFilter->SetObjectDimension(2);
            sheetnessFilter->SetBrightObject(true);
            //sheetnessFilter->SetBrightObject(false);
            sheetnessFilter->ScaleObjectnessMeasureOff();
            sheetnessFilter->Update();
            sheetnessFilter->SetROIImage(NULL);

            FloatImagePointer singleScaleSheetness = sheetnessFilter->GetOutput();

            if (i==0) {
                multiscaleSheetness = singleScaleSheetness;
                continue;
            }

            // update the multiscale sheetness
            // take the value which is larger in absolute value
            itk::ImageRegionIterator<FloatImage>
                itMulti(singleScaleSheetness,singleScaleSheetness->GetLargestPossibleRegion());
            itk::ImageRegionIterator<FloatImage>
                itSingle(multiscaleSheetness,multiscaleSheetness->GetLargestPossibleRegion());
            for (
                 itMulti.GoToBegin(),itSingle.GoToBegin();
                 !itMulti.IsAtEnd();
                 ++itMulti, ++itSingle
                 ) {
                float multiVal = itMulti.Get();
                float singleVal = itSingle.Get();

                // higher absolute value is better
                if (fabs(singleVal) > fabs(multiVal)) {
                    itMulti.Set(singleVal);
                }
            }
            delete sheetnessFilter;
        } // iteration trough scales

        return multiscaleSheetness;
    }

    static InputImagePointer computeSheetness(InputImagePointer img){
        std::vector<float> sigmasLargeScale;
        sigmasLargeScale.push_back(0.6);
        sigmasLargeScale.push_back(0.8);
        LOG<<"Unsharp masking"<<std::endl;
        FloatImagePointer imgUnsharpMasked =
            FilterUtils<FloatImage>::add(
                                         FilterUtils<InputImage,FloatImage>::cast(img),
                                         FilterUtils<FloatImage>::linearTransform(
                                                                                  FilterUtils<FloatImage>::substract(
                                                                                                                     FilterUtils<InputImage,FloatImage>::cast(img),
                                                                                                                     FilterUtils<InputImage,FloatImage>::gaussian((ConstInputImagePointer)img, 1.0)),
                                                                                  10.0, 0.0)
                                         );
        LOGI(15,ImageUtils<FloatImage>::writeImage("unsharpMasked.nii",imgUnsharpMasked););
        LOG<<"Computing multiscale sheetness measure at "<< sigmasLargeScale.size()<<" scales."<<std::endl;
        if ((float)((typename InputImage::PixelType(-1.0))) >=0){
            LOG<<"ERROR : PixelType does not support negative values! Sheetness image will be truncated . Min value is:" << numeric_limits<typename InputImage::PixelType>::min() << endl;
        }
        InputImagePointer sheetness = FilterUtils<FloatImage,InputImage>::linearTransform(multiscaleSheetness(imgUnsharpMasked, sigmasLargeScale),100,0);
        //sheetness = FilterUtils<InputImage,InputImage>::linearTransform(sheetness,255.0/200,100);
        return sheetness;

    }
    /*
      Input: Normalized CT image, scales for the sheetness measure
      Output: (ROI, MultiScaleSheetness, SoftTissueEstimation)
    */
    static InputImagePointer
    computeSoftTissueEstimate( InputImagePointer img) {
        float sigmaSmallScale = 1.5;
        UCharImagePointer roi;
        InputImagePointer softTissueEstimation;
        FloatImagePointer sheetness;

        {
            LOG<<"Thresholding input image"<<std::endl;
            InputImagePointer thresholdedInputCT =
                FilterUtils<InputImage>::thresholding(
                                                      ImageUtils<InputImage>::duplicate(img),
                                                      25, 600
                                                      );

            std::vector<float> scales; 
            scales.push_back(sigmaSmallScale);
            FloatImagePointer tmpPointer=FilterUtils<InputImage,FloatImage>::cast(thresholdedInputCT);
            FloatImagePointer smallScaleSheetnessImage =
                multiscaleSheetness(tmpPointer, scales);



            UCharImagePointer temporary = FilterUtils<FloatImage,UCharImage>::binaryThresholding(smallScaleSheetnessImage, -0.05, +0.05);
                
            typedef typename itk::ConnectedComponentImageFilter<UCharImage,UIntImage>  ConnectedComponentImageFilterType;
            typename ConnectedComponentImageFilterType::Pointer filter =
                ConnectedComponentImageFilterType::New();
            filter->SetInput(temporary);
            filter->Update();

            LOG<<"Estimating soft-tissue voxels"<<std::endl;
            softTissueEstimation =  FilterUtils<UIntImage,InputImage>::binaryThresholding(FilterUtils<UIntImage>::relabelComponents(filter->GetOutput()),1,1);
        }
     
        return softTissueEstimation;
    }

}; // class Preprocessing
