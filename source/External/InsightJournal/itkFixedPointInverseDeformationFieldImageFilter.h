/*=========================================================================

 Program:   Insight Segmentation & Registration Toolkit
 Module:    $RCSfile: itkFixedPointInverseDeformationFieldImageFilter.h,v $
 Language:  C++

 Copyright: University of Basel, All rights reserved

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 =========================================================================*/

#ifndef __itkFixedPointInverseDeformationFieldImageFilter_h
#define __itkFixedPointInverseDeformationFieldImageFilter_h


#include "itkImageToImageFilter.h"

#include "itkWarpVectorImageFilter.h"
#include "itkVectorLinearInterpolateImageFunction.h"
#include "itkImageRegionIterator.h"
#include "itkTimeProbe.h"

namespace itk
{

/** \class FixedPointInverseDeformationFieldImageFilter
 * \brief Computes the inverse of a deformation field using a fixed point iteration scheme.
 *
 * FixedPointInverseDeformationFieldImageFilter takes a deformation field as input and
 * computes the deformation field that is its inverse. If the input deformation
 * field was mapping coordinates from a space A into a space B, the output of
 * this filter will map coordinates from the space B into the space A.
 *
 * To compute the inverse of the given deformation field, the fixed point algorithm by
 * Mingli Chen, Weiguo Lu, Quan Chen, Knneth J. Ruchala and Gusavo H. Olivera
 * described in the paper
 * "A simple fixed-point approach to invert a deformation field",
 * Medical Physics, vol. 35, issue 1, p. 81,
 * is applied.
 *
 * \author Marcel Lüthi, Computer Science Department, University of Basel
 */

template < class TInputImage, class TOutputImage >
class ITK_EXPORT FixedPointInverseDeformationFieldImageFilter :
    public ImageToImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef FixedPointInverseDeformationFieldImageFilter   Self;
  typedef ImageToImageFilter<TInputImage,TOutputImage>  Superclass;
  typedef SmartPointer<Self>                            Pointer;
  typedef SmartPointer<const Self>                      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(FixedPointInverseDeformationFieldImageFilter, ImageToImageFilter);


  /** Some typedefs. */
  typedef TInputImage                              InputImageType;
  typedef typename InputImageType::ConstPointer    InputImageConstPointer;
  typedef typename InputImageType::Pointer         InputImagePointer;
  typedef typename InputImageType::PointType       InputImagePointType;
  typedef typename InputImageType::RegionType      InputImageRegionType;
  typedef typename InputImageType::SpacingType     InputImageSpacingType;
  typedef typename InputImageType::IndexType      InputImageIndexType;


  typedef TOutputImage                             OutputImageType;
  typedef typename OutputImageType::Pointer        OutputImagePointer;
  typedef typename OutputImageType::PixelType      OutputImagePixelType;
  typedef typename OutputImageType::PointType      OutputImagePointType;
  typedef typename OutputImageType::IndexType      OutputImageIndexType;
  typedef typename OutputImagePixelType::ValueType OutputImageValueType;
  typedef typename OutputImageType::SizeType 	    OutputImageSizeType;
  typedef typename OutputImageType::SpacingType     OutputImageSpacingType;
  typedef typename TOutputImage::PointType   		OutputImageOriginPointType;
  typedef TimeProbe TimeType;


  /** Number of dimensions. */
  itkStaticConstMacro(ImageDimension, unsigned int,
                      TOutputImage::ImageDimension);



  typedef ImageRegionConstIterator<InputImageType> InputConstIterator;
  typedef ImageRegionIterator<InputImageType>      InputIterator;
  typedef ImageRegionConstIterator<OutputImageType> OutputConstIterator;
  typedef ImageRegionIterator<OutputImageType>     OutputIterator;

  typedef WarpVectorImageFilter<TOutputImage,TInputImage,TOutputImage> VectorWarperType;

  typedef VectorLinearInterpolateImageFunction<TInputImage,double> FieldInterpolatorType;
  typedef typename FieldInterpolatorType::Pointer                  FieldInterpolatorPointer;
  typedef typename FieldInterpolatorType::OutputType               FieldInterpolatorOutputType;


  itkSetMacro(NumberOfIterations, unsigned int);
  itkGetConstMacro(NumberOfIterations, unsigned int);


  /** Set the size of the output image. */
  itkSetMacro( Size, OutputImageSizeType );
  /** Get the size of the output image. */
  itkGetConstReferenceMacro( Size, OutputImageSizeType );

  /** Set the output image spacing. */
  itkSetMacro(OutputSpacing, OutputImageSpacingType);
  virtual void SetOutputSpacing(const double* values);

  /** Get the output image spacing. */
  itkGetConstReferenceMacro( OutputSpacing, OutputImageSpacingType );

  /** Set the output image origin. */
  itkSetMacro(OutputOrigin, OutputImageOriginPointType);
  virtual void SetOutputOrigin( const double* values);



#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(OutputHasNumericTraitsCheck,
                  (Concept::HasNumericTraits<OutputImageValueType>));
  /** End concept checking */
#endif





protected:
  FixedPointInverseDeformationFieldImageFilter();
  ~FixedPointInverseDeformationFieldImageFilter() {}

  void PrintSelf(std::ostream& os, Indent indent) const;

  void GenerateData( );
  void GenerateOutputInformation();
  void GenerateInputRequestedRegion();
  unsigned int m_NumberOfIterations;


private:
  FixedPointInverseDeformationFieldImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  OutputImageSizeType                      m_Size;              // Size of the output image
  OutputImageSpacingType                   m_OutputSpacing;     // output image spacing
  OutputImageOriginPointType               m_OutputOrigin;      // output image origin


};

} // end namespace itk

#include "itkFixedPointInverseDeformationFieldImageFilter.txx"

#endif
