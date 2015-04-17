#ifndef IMAGE_UTILS
#define IMAGE_UTILS
#pragma once

#include <limits.h>
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
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionConstIterator.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include <boost/random.hpp>
#include "time.h"
#include "itkImageAdaptor.h"
#include "itkContinuousIndex.h"
#include <ctime>
#include <itkResampleImageFilter.h>
//#include "FilterUtils.hpp"
template<typename ImageType, class FloatPrecision=float>
class ImageUtils {

public:
	typedef typename ImageType::Pointer  ImagePointerType;
	typedef typename ImageType::ConstPointer  ConstImagePointerType;
	typedef typename ImageType::PixelType PixelType;
    typedef typename ImageType::SizeType SizeType;
    

    typedef typename itk::Image<FloatPrecision,ImageType::ImageDimension> FloatImageType;
    typedef typename FloatImageType::Pointer FloatImagePointerType;
    typedef typename itk::Image<unsigned int,ImageType::ImageDimension> UIntImageType;
    typedef typename UIntImageType::Pointer UIntImagePointerType;
    typedef typename itk::Image<int,ImageType::ImageDimension> IntImageType;
    typedef typename IntImageType::Pointer IntImagePointerType;
    typedef typename itk::Image<char,ImageType::ImageDimension> CharImageType;
    typedef typename CharImageType::Pointer CharImagePointerType;
    typedef typename itk::Image<unsigned char,ImageType::ImageDimension> UCharImageType;
    typedef typename UCharImageType::Pointer UCharImagePointerType;
    typedef itk::ImageRegionIterator<ImageType> ImageIteratorType;
    typedef itk::ImageRegionConstIterator<ImageType> ConstImageIteratorType;
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
	typedef typename ImageType::PointType PointType;
	typedef typename ImageType::DirectionType DirectionType;
	typedef typename ImageType::SpacingType SpacingType;

    static const unsigned int D=ImageType::ImageDimension;

    typedef  itk::ContinuousIndex<double,D> ContinuousIndexType;


    typedef itk::Vector<FloatPrecision,D> FloatVectorType;
    typedef itk::Image<FloatVectorType,D> FloatVectorImageType;
    typedef typename FloatVectorImageType::Pointer FloatVectorImagePointerType;

static const __int64 DELTA_EPOCH_IN_MICROSECS= 11644473600000000;

struct timezone2 
{
  __int32  tz_minuteswest; /* minutes W of Greenwich */
  bool  tz_dsttime;     /* type of dst correction */
};


static int gettimeofday(struct timeval *tv/*in*/, struct timezone2 *tz/*in*/)
{
  FILETIME ft;
  __int64 tmpres = 0;
  TIME_ZONE_INFORMATION tz_winapi;
  int rez=0;

   ZeroMemory(&ft,sizeof(ft));
   ZeroMemory(&tz_winapi,sizeof(tz_winapi));

    GetSystemTimeAsFileTime(&ft);

    tmpres = ft.dwHighDateTime;
    tmpres <<= 32;
    tmpres |= ft.dwLowDateTime;

    /*converting file time to unix epoch*/
    tmpres /= 10;  /*convert into microseconds*/
    tmpres -= DELTA_EPOCH_IN_MICROSECS; 
    tv->tv_sec = (__int32)(tmpres*0.000001);
    tv->tv_usec =(tmpres%1000000);


    //_tzset(),don't work properly, so we use GetTimeZoneInformation
    rez=GetTimeZoneInformation(&tz_winapi);
    tz->tz_dsttime=(rez==2)?true:false;
    tz->tz_minuteswest = tz_winapi.Bias + ((rez==2)?tz_winapi.DaylightBias:0);

  return 0;
}



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
		try{
			reader->Update();
		}
		catch( itk::ExceptionObject & err )
		{
			std::cerr << "ExceptionObject caught !" << std::endl;
            std::cerr << "Could not read image from "<<fileName<<" "<<std::endl;
			std::cerr << err << std::endl;
		}
            return reader->GetOutput();
       
	}

	/* Read an itk image from a file */
	static ImagePointerType readImage(std::string fileName, RegionType roi) {
		ReaderTypePointer reader = ReaderType::New();
		reader->SetFileName( fileName  );

		ROIFilterPointerType roiFilter = ROIFilterType::New();
		roiFilter->SetInput(reader->GetOutput());
		roiFilter->SetRegionOfInterest(roi);
		try{
			roiFilter->Update();
		}
		catch( itk::ExceptionObject & err )
		{
			std::cerr << "ExceptionObject caught !" << std::endl;
			std::cerr << err << std::endl;

		}
		return roiFilter->GetOutput();
	}

	/* Write an itk image to a file */
	static void writeImage(std::string fileName, ImagePointerType image) {
		WriterTypePointer writer = WriterType::New();
		writer->SetFileName( fileName );
		writer->SetInput( image );
		writer->Update();
	}
	static void writeImage(std::string fileName, ConstImagePointerType image) {
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
  
    
	static ImagePointerType duplicateConst(ConstImagePointerType img) {
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

	static ImagePointerType createEmpty(ConstImagePointerType refImg) {
		ImagePointerType img=ImageType::New();
		img->SetRegions(refImg->GetLargestPossibleRegion());
		img->SetOrigin(refImg->GetOrigin());
		img->SetSpacing(refImg->GetSpacing());
		img->SetDirection(refImg->GetDirection());
		img->Allocate();
		return img;
	};

    static ImagePointerType createEmpty(RegionType region, PointType origin, SpacingType spacing, DirectionType dir) {
		ImagePointerType img=ImageType::New();
		img->SetRegions(region);
        img->SetOrigin(origin);
		img->SetSpacing(spacing);
		img->SetDirection(dir);
		img->Allocate();
		return img;
	};

	static ImagePointerType createEmpty(ImagePointerType refImg) {
		return createEmpty((ConstImagePointerType)refImg);
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
    
    static double sumAbsDist(ConstImagePointerType im1,ConstImagePointerType im2){
        int c=0;
        double result=0.0;
        typedef itk::ImageRegionConstIterator<ImageType> IteratorType;
        IteratorType it1(im1,im1->GetLargestPossibleRegion());
        IteratorType it2(im2,im2->GetLargestPossibleRegion());
        for (it1.GoToBegin(),it2.GoToBegin();!it1.IsAtEnd();++it1,++it2,++c){
            result+=fabs(it1.Get()-it2.Get());
        }
        return result/c;
    }

    static  void multiplyImage(ImagePointerType img, double scalar){
        typedef itk::ImageRegionIterator<ImageType> IteratorType;
        IteratorType it1(img,img->GetLargestPossibleRegion());
        for (it1.GoToBegin();!it1.IsAtEnd();++it1){
            it1.Set(it1.Get()*scalar);
        }

    }
    static  void expNormImage(ImagePointerType img, double scalar){
        typedef itk::ImageRegionIterator<ImageType> IteratorType;
        IteratorType it1(img,img->GetLargestPossibleRegion());
        
        if (scalar==0.0){
            double mean=0.0;
            int c=0;
            for (it1.GoToBegin();!it1.IsAtEnd();++it1){
                mean+=it1.Get();
                ++c;
            }
            scalar=2.0*mean/c;
            
        }
        for (it1.GoToBegin();!it1.IsAtEnd();++it1){
            it1.Set(exp(-it1.Get()/scalar));
        }

    }
    static  void add(ImagePointerType img, PixelType scalar){
        typedef itk::ImageRegionIterator<ImageType> IteratorType;
        IteratorType it1(img,img->GetLargestPossibleRegion());
        for (it1.GoToBegin();!it1.IsAtEnd();++it1){
            it1.Set(it1.Get()+scalar);
        }

    }
    static  ImagePointerType addOutOfPlace(ImagePointerType img, PixelType scalar){
        ImagePointerType result=duplicate(img);
        typedef itk::ImageRegionIterator<ImageType> IteratorType;
        IteratorType it1(img,img->GetLargestPossibleRegion());
        IteratorType itRes(result,img->GetLargestPossibleRegion());
        itRes.GoToBegin();
        for (it1.GoToBegin();!it1.IsAtEnd();++it1,++itRes){
            itRes.Set(it1.Get()+scalar);
        }
        return result;
    }
    static  void sqrtImage(ImagePointerType img){
        typedef itk::ImageRegionIterator<ImageType> IteratorType;
        IteratorType it1(img,img->GetLargestPossibleRegion());
        for (it1.GoToBegin();!it1.IsAtEnd();++it1){
            it1.Set(sqrt(it1.Get()));
        }

    }
    static  ImagePointerType multiplyImageOutOfPlace(ImagePointerType img, double scalar){
        typedef itk::ImageRegionIterator<ImageType> IteratorType;
        ImagePointerType result=createEmpty(ConstImagePointerType(img));
        IteratorType it1(img,img->GetLargestPossibleRegion());
        IteratorType it2(result,img->GetLargestPossibleRegion());
        for (it2.GoToBegin(),it1.GoToBegin();!it1.IsAtEnd();++it1,++it2){
            it2.Set(it1.Get()*scalar);
            //std::cout<<it2.Get()<<" "<<scalar<<" "<<it1.Get()<<std::endl;

        }
        return result;
    }


    static  ImagePointerType multiplyImageOutOfPlace(ImagePointerType img, ImagePointerType img2,bool verbose=false){
        typedef itk::ImageRegionIterator<ImageType> IteratorType;
        ImagePointerType result=createEmpty(ConstImagePointerType(img));
        IteratorType it1(img,img->GetLargestPossibleRegion());
        IteratorType it2(img2,img2->GetLargestPossibleRegion());
        IteratorType itRes(result,img->GetLargestPossibleRegion());
        itRes.GoToBegin();
        for (it1.GoToBegin(),it2.GoToBegin();!it1.IsAtEnd();++it1,++it2,++itRes){
            itRes.Set(it1.Get()*it2.Get());
            //std::cout<<it2.Get()<<" "<<scalar<<" "<<it1.Get()<<std::endl;
            if (verbose) std::cout<<itRes.Get()<<" "<<it1.Get()<<" "<<it2.Get()<<std::endl;

        }
        return result;
    }
     static  void multiplyImage(ImagePointerType img, ImagePointerType img2,bool verbose=false){
            typedef itk::ImageRegionIterator<ImageType> IteratorType;

        IteratorType it1(img,img->GetLargestPossibleRegion());
        IteratorType it2(img2,img2->GetLargestPossibleRegion());
        for (it1.GoToBegin(),it2.GoToBegin();!it1.IsAtEnd();++it1,++it2){
            PixelType val=it1.Get()*it2.Get();
            it1.Set(val);
            //std::cout<<it2.Get()<<" "<<scalar<<" "<<it1.Get()<<std::endl;
            if (verbose) std::cout<<" "<<it1.Get()<<" "<<it2.Get()<<std::endl;

        }
    }
     static  ImagePointerType localSquare(ImagePointerType img){
        typedef itk::ImageRegionIterator<ImageType> IteratorType;
        ImagePointerType result=createEmpty(ConstImagePointerType(img));
        IteratorType it1(img,img->GetLargestPossibleRegion());
        IteratorType itRes(result,img->GetLargestPossibleRegion());
        itRes.GoToBegin();
        for (it1.GoToBegin();!it1.IsAtEnd();++it1,++itRes){
            PixelType p=it1.Get();
            itRes.Set(p*p);
            //std::cout<<it2.Get()<<" "<<scalar<<" "<<it1.Get()<<std::endl;

        }
        return result;
    }

    static  ImagePointerType divideImageOutOfPlace(ImagePointerType img, ImagePointerType img2,bool verbose=false){
        typedef itk::ImageRegionIterator<ImageType> IteratorType;
        ImagePointerType result=createEmpty(ConstImagePointerType(img));
        IteratorType it1(img,img->GetLargestPossibleRegion());
        IteratorType it3(img2,img2->GetLargestPossibleRegion());
        IteratorType it2(result,img->GetLargestPossibleRegion());
        it3.GoToBegin();
        for (it2.GoToBegin(),it1.GoToBegin();!it1.IsAtEnd();++it1,++it2,++it3){
            it2.Set(it1.Get()/it3.Get());

        }
        return result;
    }
    static ImagePointerType deformSegmentationImage(ConstImagePointerType segmentationImage, FloatVectorImagePointerType deformation){
            //assert(segmentationImage->GetLargestPossibleRegion().GetSize()==deformation->GetLargestPossibleRegion().GetSize());
            typedef  typename itk::ImageRegionIterator<FloatVectorImageType> LabelIterator;
            typedef   typename itk::ImageRegionIterator<ImageType> ImageIterator;
            LabelIterator deformationIt(deformation,deformation->GetLargestPossibleRegion());
        
            typedef typename  itk::NearestNeighborInterpolateImageFunction<ImageType, double> ImageInterpolatorType;
            typename ImageInterpolatorType::Pointer interpolator=ImageInterpolatorType::New();

            interpolator->SetInputImage(segmentationImage);
            ImagePointerType deformed=ImageUtils<ImageType>::createEmpty(segmentationImage);
            deformed->SetRegions(deformation->GetLargestPossibleRegion());
            deformed->SetOrigin(deformation->GetOrigin());
            deformed->SetSpacing(deformation->GetSpacing());
            deformed->SetDirection(deformation->GetDirection());
            deformed->Allocate();
            ImageIterator imageIt(deformed,deformed->GetLargestPossibleRegion());        
            
            for (imageIt.GoToBegin(),deformationIt.GoToBegin();!imageIt.IsAtEnd();++imageIt,++deformationIt){
                IndexType index=deformationIt.GetIndex();
                typename ImageInterpolatorType::ContinuousIndexType idx(index);
                FloatVectorType displacement=deformationIt.Get();
                idx+=(displacement);
                if (interpolator->IsInsideBuffer(idx)){
                    imageIt.Set(int(interpolator->EvaluateAtContinuousIndex(idx)));
                }else{
                    imageIt.Set(0);
                }
            }
            return deformed;
        }
    static     ImagePointerType deformImage(ConstImagePointerType image, FloatVectorImagePointerType deformation){
            //assert(segmentationImage->GetLargestPossibleRegion().GetSize()==deformation->GetLargestPossibleRegion().GetSize());
            typedef  typename itk::ImageRegionIterator<FloatVectorImageType> LabelIterator;
            typedef  itk::ImageRegionIterator<ImageType> ImageIterator;
            LabelIterator deformationIt(deformation,deformation->GetLargestPossibleRegion());

            typedef typename itk::LinearInterpolateImageFunction<ImageType, double> ImageInterpolatorType;
            typename ImageInterpolatorType::Pointer interpolator=ImageInterpolatorType::New();

            interpolator->SetInputImage(image);
            ImagePointerType deformed=ImageType::New();//ImageUtils<ImageType>::createEmpty(image);
      
            deformed->SetRegions(deformation->GetLargestPossibleRegion());
            deformed->SetOrigin(deformation->GetOrigin());
            deformed->SetSpacing(deformation->GetSpacing());
            deformed->SetDirection(deformation->GetDirection());
            deformed->Allocate();
            ImageIterator imageIt(deformed,deformed->GetLargestPossibleRegion());        
            for (imageIt.GoToBegin(),deformationIt.GoToBegin();!imageIt.IsAtEnd();++imageIt,++deformationIt){
                IndexType index=deformationIt.GetIndex();
                typename ImageInterpolatorType::ContinuousIndexType idx(index);
                FloatVectorType displacement=deformationIt.Get();
                idx+=(displacement);
                if (interpolator->IsInsideBuffer(idx)){
                    imageIt.Set(interpolator->EvaluateAtContinuousIndex(idx));
                    //deformed->SetPixel(imageIt.GetIndex(),interpolator->EvaluateAtContinuousIndex(idx));

                }else{
                    imageIt.Set(0);
                    //                deformed->SetPixel(imageIt.GetIndex(),0);
                }
            }
            return deformed;
        }
    
    static inline  int ImageIndexToLinearIndex(IndexType &idx, SizeType &size, bool & inside){
        inside=true;
        unsigned int dimensionMultiplier=1;
        int withinImageIndex=0;
        for ( int d=0; d<D;++d){
            if (idx[d]>=0 && idx[d]<size[d]){
                withinImageIndex+=dimensionMultiplier*idx[d];
                dimensionMultiplier*= size[d];
            }else inside=false;
        }
        return withinImageIndex;
    }

    static inline ImagePointerType cutOffIntensities(ImagePointerType img){
        typedef itk::ImageRegionIterator<ImageType> IteratorType;
        IteratorType it1(img,img->GetLargestPossibleRegion());
        ImagePointerType res=createEmpty(img);
        IteratorType it2(res,img->GetLargestPossibleRegion());
        for (it1.GoToBegin(),it2.GoToBegin();!it1.IsAtEnd();++it1,++it2){
            PixelType maxCutOff=max(it1.Get(),std::numeric_limits<PixelType>::min());
            PixelType minMaxCutOff=min(maxCutOff,std::numeric_limits<PixelType>::max());
                                    
            it2.Set(minMaxCutOff);
        }
        return res;
    }


    static inline void setRegion(ImagePointerType img, RegionType region, PixelType val){
        ImageIteratorType it(img,region);
        for (it.GoToBegin();!it.IsAtEnd();++it){
            it.Set(val);
        }

    }


    static std::map<std::string,ImagePointerType>  readImageList(std::string filename,std::vector<std::string> & imageIDs,ImagePointerType ROI=NULL,bool nnInterpol=false){
        std::map<std::string,ImagePointerType>  result;//=new  std::map<std::string,ImagePointerType>;

        std::ifstream ifs(filename.c_str());
        if (!ifs){
            std::cerr<<"could not read "<<filename<<std::endl;
            exit(0);
        }
        while( ! ifs.eof() ) 
            {
                std::string imageID;
                ifs >> imageID;                
                if (imageID!=""){
                    imageIDs.push_back(imageID);
                    ImagePointerType img;
                    std::string imageFileName ;
                    ifs >> imageFileName;
                    //LOGV(3)<<"Reading image with id "<<imageID<<" from file "<<imageFileName<<endl;
                    img=ImageUtils<ImageType>::readImage(imageFileName);
                    if (ROI.IsNotNull()){
                        typedef typename itk::ResampleImageFilter< ImageType,ImageType>	ResampleFilterType;
                        typename ResampleFilterType::Pointer resampler=ResampleFilterType::New();
                        typedef typename itk::NearestNeighborInterpolateImageFunction<ImageType, double> NNInterpolatorType;
                        typedef typename NNInterpolatorType::Pointer NNInterpolatorPointerType;
                        NNInterpolatorPointerType interpolNN=NNInterpolatorType::New();
                        if (nnInterpol){
                            resampler->SetInterpolator(interpolNN);
                        }
                        resampler->SetInput(img);
                        resampler->SetOutputOrigin(ROI->GetOrigin());
                        resampler->SetOutputSpacing ( ROI->GetSpacing() );
                        resampler->SetOutputDirection ( ROI->GetDirection() );
                        resampler->SetSize ( ROI->GetLargestPossibleRegion().GetSize() );
                        resampler->Update();
                        img=resampler->GetOutput();
                        //img=FilterUtils<ImageType>::linearResample(img,ROI,false);
                    }
                    if (result.find(imageID)==result.end())
                        result[imageID]=img;
                    else{
                        std::cerr<<"duplicate image ID "<<imageID<<", aborting"<<std::endl;
                        exit(0);
                    }
                }
            }
        return result;
    }        


    static ImagePointerType addNoise(ImagePointerType img, double var=0.01, double mean=0.0, double freq=0.1){

        struct timeval time; 
        static boost::minstd_rand randgen((time.tv_sec * 1000) + (time.tv_usec / 1000));
		static boost::normal_distribution<> dist(mean, var);
		static boost::variate_generator<boost::minstd_rand, boost::normal_distribution<> > r(randgen, dist);

        gettimeofday(&time,NULL);    
        
        srand((time.tv_sec * 1000) + (time.tv_usec / 1000));

        typedef itk::ImageRegionIterator<ImageType> IteratorType;
        ImagePointerType result=createEmpty(ConstImagePointerType(img));
        IteratorType it1(img,img->GetLargestPossibleRegion());
        IteratorType it2(result,img->GetLargestPossibleRegion());
        for (it2.GoToBegin(),it1.GoToBegin();!it1.IsAtEnd();++it1,++it2){
           
            double val=it1.Get();
            float rnd = (float)rand()/(float)RAND_MAX;
            if (rnd<freq){
                double randVal=r();
                val+=randVal;
                if (val<std::numeric_limits<PixelType>::min())
                    val=std::numeric_limits<PixelType>::min();
                else if (val>std::numeric_limits<PixelType>::max())
                    val=std::numeric_limits<PixelType>::max();
            }
            it2.Set(val);
            //std::cout<<it2.Get()<<" "<<randVal<<" "<<it1.Get()<<std::endl;
        }
        return result;

    }
};
#endif // IMAGE_UTILS
