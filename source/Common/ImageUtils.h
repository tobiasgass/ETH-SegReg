#ifndef IMAGE_UTILS
#define IMAGE_UTILS
#pragma once


#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageFileWriter.h"
#include "itkImageDuplicator.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkImageSeriesWriter.h"
#include "itkGDCMImageIO.h"
#include "itkNumericSeriesFileNames.h" 
#include "itkImageRegion.h"

template<class ImageType>
class ImageUtils {

    
    typedef typename ImageType::Pointer  ImagePointerType;
    typedef typename ImageType::PixelType PixelType;
  


    typedef itk::Image< PixelType , 2 > ImageType2D;

    typedef itk::ImageFileReader< ImageType >  ReaderType;
    typedef typename ReaderType::Pointer  ReaderTypePointer;
    typedef itk::ImageFileWriter< ImageType >  WriterType;
    typedef typename WriterType::Pointer  WriterTypePointer;
    typedef itk::ImageSeriesWriter< ImageType, ImageType2D>   SeriesWriterType;
    typedef typename SeriesWriterType::Pointer SeriesWriterTypePointer;

    typedef itk::NumericSeriesFileNames             NamesGeneratorType; 

    typedef std::vector< std::string > FileNamesContainer;
    typedef itk::GDCMImageIO           ImageIOType;

    typedef itk::ImageDuplicator< ImageType > DuplicatorType;
    typedef typename DuplicatorType::Pointer DuplicatorPointerType;
	typedef itk::RegionOfInterestImageFilter<ImageType,ImageType> ROIFilterType;
    typedef typename ROIFilterType::Pointer ROIFilterPointerType;


    typedef typename ImageType::IndexType IndexType;
    typedef typename ImageType::SizeType RegionSizeType;
    typedef typename ImageType::RegionType RegionType;
    typedef typename ImageType::OffsetType OffsetType;
    typedef typename ImageType::SpacingType SpacingType;


private:

    static void ITKImageToVTKImage(ImagePointerType image) {

        // fix origin
        itk::Point<double,ImageType::ImageDimension> origin = image->GetOrigin();
        origin[0] *= -1;
        origin[1]*=-1;
        image->SetOrigin(origin);

        // fix spacing
        SpacingType spacing = image->GetSpacing();
        spacing[0]*=-1;
        spacing[1]*=-1;
        image->SetSpacing(spacing);
    }

    static void VTKImageToITKImage(ImagePointerType image) {
        ITKImageToVTKImage(image);
    }




public:


    /* Read an itk image from a file */
    static ImagePointerType readImage(std::string fileName) {
        ReaderTypePointer reader = ReaderType::New();
        reader->SetFileName( fileName  );
        reader->Update();
        return reader->GetOutput();
    }

    /* Read an itk image from a file */
    static ImagePointerType readImage(std::string fileName, RegionType roi) {
        ReaderTypePointer reader = ReaderType::New();
        reader->SetFileName( fileName  );

        ROIFilterPointerType roiFilter = ROIFilterType::New();
        roiFilter->SetInput(reader->GetOutput());
        roiFilter->SetRegionOfInterest(roi);

        roiFilter->Update();

        return roiFilter->GetOutput();
    }

    /* Write an itk image to a file */
    static void writeImage(std::string fileName, ImagePointerType image) {
        WriterTypePointer writer = WriterType::New();
        writer->SetFileName( fileName );
        writer->SetInput( image );
        writer->Update();
    }

    
    /* Write an itk image as dicom slices to a directory */
    static void writeImageSeries(std::string directoryname, ImagePointerType image) {
        
        itksys::SystemTools::MakeDirectory( directoryname.c_str() ); 
        SeriesWriterTypePointer seriesWriter = SeriesWriterType::New();
        seriesWriter->SetInput( image );
        ImageIOType::Pointer dicomIO = ImageIOType::New();
        seriesWriter->SetImageIO( dicomIO );
        

        RegionType region= image->GetLargestPossibleRegion();

        std::cout << image->GetSliceThickess() << std::endl;
        
        IndexType start = region.GetIndex();
        RegionSizeType  size  = region.GetSize();
        
        std::string format = directoryname;
        format += "/image%03d.dcm";

        NamesGeneratorType::Pointer namesGenerator = NamesGeneratorType::New(); 
        namesGenerator->SetSeriesFormat( format.c_str() );
        namesGenerator->SetStartIndex( start[2] );
        namesGenerator->SetEndIndex( start[2] + size[2] - 1 );
        namesGenerator->SetIncrementIndex( 1 );

        seriesWriter->SetFileNames( namesGenerator->GetFileNames() ); 
        
        try
            {
                seriesWriter->Update();
            }
        catch( itk::ExceptionObject & excp )
            {
                std::cerr << "Exception thrown while writing the series " << std::endl;
                std::cerr << excp << std::endl;
            }
        
        
    }
    
    static ImagePointerType duplicate(ImagePointerType img) {
        DuplicatorPointerType duplicator = DuplicatorType::New();
        duplicator->SetInputImage( img );
        duplicator->Update();
        return duplicator->GetOutput();
    }
    
    
    static ImagePointerType createEmpty(const RegionSizeType & size) {
        
       IndexType startIndex;
       startIndex.Fill(0);

       RegionType region;
       region.SetSize(size);
       region.SetIndex(startIndex);

       ImagePointerType img = ImageType::New();
       img->SetRegions( region );
       img->Allocate();

       return img;
    };


    /*
    Get the region representing the bounding box which is in @offset pixels
    distance from @region boundary.

    For region with start [i1,i2] and size [s1,s2], return region
    with start [i1+offset, i2+offset] and size [s1 - 2xoffset,s2 - 2xoffset]

    */
    static RegionType getInnerBoundingBox(
        const RegionType & region, unsigned int offset
    ) {

        IndexType newStart = region.GetIndex();
        RegionSizeType newSize = region.GetSize();

        for (unsigned int dim = 0; dim < region.GetImageDimension(); ++dim) {
            newStart[dim] += offset;
            newSize[dim] -= 2 * offset;
        }

        RegionType innerRegion;
        innerRegion.SetIndex(newStart);
        innerRegion.SetSize(newSize);

        return innerRegion;


    }


    static IndexType getIndexFromVTKPoint(
        ImagePointerType itkImage,
        const std::vector<float> & vtkSeed
    ) {

        VTKImageToITKImage(itkImage);

        itk::Point<double,ImageType::ImageDimension> itkPoint;
        for (unsigned int i=0; i<ImageType::ImageDimension; ++i ) {
            itkPoint[i] = vtkSeed[i];
        }

        IndexType itkIndex;
        itkImage->TransformPhysicalPointToIndex(itkPoint,itkIndex);

        ITKImageToVTKImage(itkImage);
        return itkIndex;

    }

    static bool isInside(ImagePointerType itkImage, const IndexType &idx) {

        RegionSizeType size = itkImage->GetLargestPossibleRegion().GetSize();
        for (unsigned int i=0; i<ImageType::ImageDimension; ++i ) {
            if ((unsigned)idx[i] < 0 || (unsigned)idx[i] >= size[i])
                return false;
        }

        return true;
    }

};
#endif // IMAGE_UTILS
