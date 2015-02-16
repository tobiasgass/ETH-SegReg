/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: MultiResAffineRegistration3D.cxx,v $
  Language:  C++
  Date:      $Date: 2010/02/28 20:19:51 $
  Version:   $Revision: 1.1.1.1 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

  This software is distributed WITHOUT ANY WARRANTY; without even 
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
  PURPOSE.  See the above copyright notices for more information.

  =========================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

//  This example illustrates the use of more complex components of the
//  registration framework. In particular, it introduces the use of the
//  \doxygen{AffineTransform} and the importance of fine-tuning the scale
//  parameters of the optimizer.
//
// \index{itk::ImageRegistrationMethod!AffineTransform}
// \index{itk::ImageRegistrationMethod!Scaling parameter space}
// \index{itk::AffineTransform!Image Registration}
//
// The AffineTransform is a linear transformation that maps lines into
// lines. It can be used to represent translations, rotations, anisotropic
// scaling, shearing or any combination of them. Details about the affine
// transform can be seen in Section~\ref{sec:AffineTransform}.
//


#include "itkAffineTransform.h"
#include "itkTransformFileReader.h"
#include "itkTransformFileWriter.h"

#include "itkCenteredTransformInitializer.h"
#include "itkMultiResolutionImageRegistrationMethod.h"
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkNormalizedCorrelationImageToImageMetric.h"
#include "itkMattesMutualInformationImageToImageMetric.h"
#include "itkNormalizedMutualInformationHistogramImageToImageMetric.h"
#include "itkLinearInterpolateImageFunction.h"

#include "itkGradientDescentOptimizer.h"
#include "itkRegularStepGradientDescentOptimizer.h"
#include "itkSPSAOptimizer.h"
#include "itkOnePlusOneEvolutionaryOptimizer.h"
#include "itkNormalVariateGenerator.h"
#include "itkPowellOptimizer.h"

#include "itkRecursiveMultiResolutionPyramidImageFilter.h"
#include "itkImage.h"
#include "itkImageMaskSpatialObject.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkCheckerBoardImageFilter.h"
#include <limits>
// Default filenames
char *fixedName = NULL, *movingName = NULL, *dofinName = NULL, *dofoutName = NULL, *fixedmaskName = NULL;
char *movingTransfName = NULL, *checkBeforeName = NULL, *checkAfterName = NULL, *movingmaskName = NULL;
char metricName[256] = "NCC";
char optimizerName[256] = "RSGD";

bool MinimizeFlag = true;
double overallbestValue = std::numeric_limits<double>::max();
	


const    unsigned int    Dimension = 3;

typedef  float    InternalPixelType;
typedef itk::Image< InternalPixelType, Dimension > InternalImageType;
 
typedef itk::MultiResolutionImageRegistrationMethod< 
    InternalImageType, 
    InternalImageType    > RegistrationType;

typedef RegistrationType::ParametersType ParametersType;
ParametersType overallbestParameters;


//  The following section of code implements an observer
//  that will monitor the evolution of the registration process.
//
#include "itkCommand.h"
class CommandIterationUpdate : public itk::Command 
{
public:
    typedef  CommandIterationUpdate   Self;
    typedef  itk::Command             Superclass;
    typedef  itk::SmartPointer<Self>  Pointer;
    itkNewMacro( Self );
protected:
    CommandIterationUpdate(): m_CumulativeIterationIndex(0) {};
public:
    typedef itk::GradientDescentOptimizer                GDOptimizerType;
    typedef itk::RegularStepGradientDescentOptimizer     RSGDOptimizerType;
    typedef itk::SPSAOptimizer                           SPSAOptimizerType;
    typedef itk::OnePlusOneEvolutionaryOptimizer         EVOptimizerType;
    typedef itk::PowellOptimizer                         PowellOptimizerType;


    typedef const GDOptimizerType   *   GDOptimizerPointer;
    typedef const RSGDOptimizerType *   RSGDOptimizerPointer;
    typedef const SPSAOptimizerType *   SPSAOptimizerPointer;
    typedef const EVOptimizerType *     EVOptimizerPointer;
	typedef const PowellOptimizerType * PowellOptimizerPointer;

    void Execute(itk::Object *caller, const itk::EventObject & event)
    {
        Execute( (const itk::Object *)caller, event);
    }

    void Execute(const itk::Object * object, const itk::EventObject & event)
    {
        GDOptimizerPointer   GDoptimizer;
        SPSAOptimizerPointer SPSAoptimizer;
        RSGDOptimizerPointer RSGDoptimizer;
        EVOptimizerPointer   EVoptimizer;
        PowellOptimizerPointer   PowellOptimizer;

        double thisBestValue;
        bool betterValue = false;

        if (! strcmp(optimizerName, "GD"))
            {
                GDoptimizer = dynamic_cast< GDOptimizerPointer >( object );
            }
        else
            {
                if (! strcmp(optimizerName, "SPSA"))
                    {
                        SPSAoptimizer = dynamic_cast< SPSAOptimizerPointer >( object );
                    }
                else
                    {
                        if (! strcmp(optimizerName, "EV"))
                            {
                                EVoptimizer = dynamic_cast< EVOptimizerPointer >( object );
                            }
                        else
                            {
                                if (! strcmp(optimizerName, "Powell"))
                                    {
                                        PowellOptimizer = dynamic_cast< PowellOptimizerPointer >( object );
                                    }
                                else
                                    {
                                        RSGDoptimizer = dynamic_cast< RSGDOptimizerPointer >( object );
                                    }
                            }
                    }
            }

        if( !(itk::IterationEvent().CheckEvent( &event )) )
            {
                return;
            }


        if (! strcmp(optimizerName, "GD"))
            {
                thisBestValue = GDoptimizer->GetValue();
                std::cout << GDoptimizer->GetCurrentIteration() << "   ";
            }
        else
            {
                if (! strcmp(optimizerName, "SPSA"))
                    {
                        std::cout << SPSAoptimizer->GetCurrentIteration() << "   ";
                        thisBestValue = SPSAoptimizer->GetValue();
                    }
                else
                    {
                        if (! strcmp(optimizerName, "EV"))
                            {
                                std::cout << EVoptimizer->GetCurrentIteration() << "   ";
                                thisBestValue = EVoptimizer->GetValue();
                            }
                        else
                            {
                                if (! strcmp(optimizerName, "Powell"))
                                    {
                                        std::cout << PowellOptimizer->GetCurrentIteration() << "   ";
                                        thisBestValue = PowellOptimizer->GetValue();
                                    }
                                else
                                    {
                                        std::cout << RSGDoptimizer->GetCurrentIteration() << "   ";
                                        thisBestValue = RSGDoptimizer->GetValue();
                                    }
                            }
                    }
            }

        if (MinimizeFlag)
            {
                if (thisBestValue < overallbestValue)
                    {
                        betterValue = true;
                    }

            }
        else
            {
                if (thisBestValue > overallbestValue)
                    {
                        betterValue = true;
                    }
            }


        if (betterValue)
            {
                overallbestValue = thisBestValue; 
                if (! strcmp(optimizerName, "GD"))
                    {
                        overallbestParameters = GDoptimizer->GetCurrentPosition();
                    }
                else
                    {
                        if (! strcmp(optimizerName, "SPSA"))
                            {
                                overallbestParameters = SPSAoptimizer->GetCurrentPosition();
                            }
                        else
                            {
                                if (! strcmp(optimizerName, "EV"))
                                    {
                                        overallbestParameters = EVoptimizer->GetCurrentPosition();
                                    }
                                else
                                    {
                                        if (! strcmp(optimizerName, "Powell"))
                                            {
                                                overallbestParameters = PowellOptimizer->GetCurrentPosition();
                                            }
                                        else
                                            {
                                                overallbestParameters = RSGDoptimizer->GetCurrentPosition();
                                            }
                                    }
                            }
                    }
            }
		
        std::cout << thisBestValue << "   ";
        std::cout << overallbestValue << "   ";
        std::cout << overallbestParameters;

        vnl_matrix<double> p(2, 2);
        p[0][0] = (double) overallbestParameters[0];
        p[0][1] = (double) overallbestParameters[1];
        p[1][0] = (double) overallbestParameters[2];
        p[1][1] = (double) overallbestParameters[3];
 
        double detA = p[0][0]*p[1][1] - p[0][1]*p[1][0];
      
        std::cout << " det(A) = " << detA;
        std::cout <<  "  " << m_CumulativeIterationIndex++ << std::endl;
    }
private:
    unsigned int m_CumulativeIterationIndex;
};


//  The following section of code implements a Command observer
//  that will control the modification of optimizer parameters
//  at every change of resolution level.
//
template <typename TRegistration>
class RegistrationInterfaceCommand : public itk::Command 
{
public:
    typedef  RegistrationInterfaceCommand   Self;
    typedef  itk::Command                   Superclass;
    typedef  itk::SmartPointer<Self>        Pointer;
    itkNewMacro( Self );
protected:
    RegistrationInterfaceCommand() {};
public:
    typedef   TRegistration                              RegistrationType;
    typedef   RegistrationType *                         RegistrationPointer;

    typedef itk::GradientDescentOptimizer                GDOptimizerType;
    typedef itk::RegularStepGradientDescentOptimizer     RSGDOptimizerType;
    typedef itk::SPSAOptimizer                           SPSAOptimizerType;
    typedef itk::OnePlusOneEvolutionaryOptimizer         EVOptimizerType;
    typedef itk::PowellOptimizer                         PowellOptimizerType;

    typedef GDOptimizerType   *   GDOptimizerPointer;
    typedef RSGDOptimizerType *   RSGDOptimizerPointer;
    typedef SPSAOptimizerType *   SPSAOptimizerPointer;
    typedef EVOptimizerType *     EVOptimizerPointer;
	typedef PowellOptimizerType * PowellOptimizerPointer;

    void Execute(itk::Object * object, const itk::EventObject & event)
    {
        if( !(itk::IterationEvent().CheckEvent( &event )) )
            {
                return;
            }
        RegistrationPointer registration =
            dynamic_cast<RegistrationPointer>( object );

		GDOptimizerPointer     GDoptimizer;
		SPSAOptimizerPointer   SPSAoptimizer;
		RSGDOptimizerPointer   RSGDoptimizer;
		EVOptimizerPointer     EVoptimizer;
		PowellOptimizerPointer PowellOptimizer;

		if (! strcmp(optimizerName, "GD"))
			{
                GDoptimizer = dynamic_cast< GDOptimizerPointer >( registration->GetOptimizer() );
			}
		else
			{
				if (! strcmp(optimizerName, "SPSA"))
					{
						SPSAoptimizer = dynamic_cast< SPSAOptimizerPointer >( registration->GetOptimizer() );
					}
				else
					{
						if (! strcmp(optimizerName, "EV"))
							{
								EVoptimizer = dynamic_cast< EVOptimizerPointer >( registration->GetOptimizer() );
							}
						else
							{
								if (! strcmp(optimizerName, "Powell"))
									{
										PowellOptimizer = dynamic_cast< PowellOptimizerPointer >( registration->GetOptimizer() );
									}
								else
									{
										RSGDoptimizer = dynamic_cast< RSGDOptimizerPointer >( registration->GetOptimizer() );
									}
							}
					}
			}
		
        std::cout << "-------------------------------------" << std::endl;
        std::cout << "MultiResolution Level : "
                  << registration->GetCurrentLevel()  << std::endl;
        std::cout << std::endl;

		if (MinimizeFlag)
			overallbestValue = std::numeric_limits<double>::max();  // reset to get new values in different resolution	  
		else
			overallbestValue = 0;  // reset to get new values in different resolution

		if (! strcmp(optimizerName, "EV"))
			{
				if ( registration->GetCurrentLevel() > 0 )
					{
						double initRadius = EVoptimizer->GetInitialRadius(); 
						double grow = EVoptimizer->GetGrowthFactor();
						initRadius /= 1.5;
						grow /= 1.5;
						double shrink = pow(grow, -0.25) ;
						EVoptimizer->Initialize(initRadius, grow, shrink) ;
						std::cout << "EV Radius : " << initRadius << " grow " << grow << std::endl;
					}
			}

 		if (! strcmp(optimizerName, "Powell"))
 			{
				if ( registration->GetCurrentLevel() > 0 )
					{
						PowellOptimizer->SetStepLength( PowellOptimizer->GetStepLength() / 2.0 );
						PowellOptimizer->SetStepTolerance( PowellOptimizer->GetStepTolerance() / 2.0 );
					}
				std::cout << "Stepsize : "
                          << PowellOptimizer->GetStepLength() << std::endl;
 			}
	

		if (! strcmp(optimizerName, "RSGD"))
			{
                if ( registration->GetCurrentLevel() > 0 )
                    {
                        RSGDoptimizer->SetMaximumStepLength( RSGDoptimizer->GetMaximumStepLength() / 2.0 );
                        RSGDoptimizer->SetMinimumStepLength( RSGDoptimizer->GetMinimumStepLength() / 5.0 );
                        RSGDoptimizer->SetNumberOfIterations(  RSGDoptimizer->GetNumberOfIterations()/2.0  );
                    }
                std::cout << "Stepsize : "
                          << RSGDoptimizer->GetMinimumStepLength()  << " " << RSGDoptimizer->GetMaximumStepLength() << std::endl;
                std::cout<< "MaxIter : " << RSGDoptimizer->GetNumberOfIterations() << std::endl;
			}
    }
    void Execute(const itk::Object * , const itk::EventObject & )
    { return; }
};


int usage()
{
    std::cerr << "Missing Parameters " << std::endl;
    std::cerr << "Usage: MultiResAffineRegistration2D fixedImageFile  movingImageFile <options>" << std::endl;
    std::cerr << "where <options> are "<< std::endl;
    std::cerr << " -levels value           number of levels (default 3)" << std::endl;
    std::cerr << " -stepLength value       maximum step length (default 0.1)" << std::endl;
    std::cerr << " -maxSteps value         maximum number of iterations (default 300)" << std::endl;
    std::cerr << " -stepReduction value    divisor for reducing number of iterations per level" << std::endl;
    std::cerr << " -transScale value       scale of translation vs. rotation/shearing/scaling (default 1.0/1e7)" << std::endl;
    std::cerr << " -rotScale   value       scale of rotation/shearing vs. translation/scaling (default 1.0)" << std::endl;
    std::cerr << " -xScale   value       scale of rotation/shearing vs. translation/scaling (default 1.0)" << std::endl;
    std::cerr << " -yScale   value       scale of rotation/shearing vs. translation/scaling (default 1.0)" << std::endl;
    std::cerr << " -zScale   value       scale of rotation/shearing vs. translation/scaling (default 1.0)" << std::endl;
    std::cerr << " -metric value           image similarity metric (SSD, NCC, MI, NMI) (default NCC)" << std::endl;
    std::cerr << " -optimizer value        select optimizer (GD, RSGD, SPSA, EV, Powell) (default RSGD)" << std::endl;
    std::cerr << " -best                   use best result ever encountered by optimizer (GD, RSGD, SPSA)" << std::endl;
    std::cerr << " -useDerivatives         use explicit derivatives for MI (default Off)" << std::endl;
    std::cerr << " -nBins value            number of bins for MI & NMI (default 128)" << std::endl;
    std::cerr << " -nSamples value         number of samples for MI (default 50000)" << std::endl;
    std::cerr << " -dofin file             initial affine transformation" << std::endl;
    std::cerr << " -moments                initialize by using moments" << std::endl;
    std::cerr << " -dofout file            resulting affine transformation" << std::endl;
    std::cerr << " -fixedmask file         use only intensities within mask for similarity " << std::endl;
    std::cerr << " -movingmask file        use only intensities within mask for similarity " << std::endl;
    std::cerr << " -outputAfter file       output transformed source image after registration" << std::endl;
    std::cerr << " -outputCheckAfter file  output checkerboard image after registration" << std::endl;
    std::cerr << " -outputCheckBefore file output checkerboard image before registration" << std::endl;
    return EXIT_FAILURE;
}


int main( int argc, char *argv[] )
{
    // Check command line
    if (argc < 3){
        return usage();
    }

    // Parse input and output names
    fixedName = argv[1];
    argc--;
    argv++;
    movingName = argv[1];
    argc--;
    argv++;
  
    bool ok = true;
    unsigned int numberOfLevels = 3;
    double steplength = 0.1;
    unsigned int maxNumberOfIterations = 300;
    double translationScale = 1.0 / 1e7;
    double rotationScale = 1.0 ;
    double xScale=1.0,yScale=1.0,zScale=1.0;
    unsigned int numberOfHistogramBins = 128;
    unsigned int numberOfSpatialSamples = 50000;
    unsigned int useExplicitPDFDerivatives = 0;
    bool initWithMoments = false;
	bool useBestResult = false;
    double stepReduction=2.0;
    while (argc > 1){
        ok = false;
        if ((ok == false) && (strcmp(argv[1], "-levels") == 0)){
            argc--;
            argv++;
            numberOfLevels = atoi(argv[1]);
            argc--;
            argv++;
            ok = true;
        }
        if ((ok == false) && (strcmp(argv[1], "-stepLength") == 0)){
            argc--;
            argv++;
            steplength = atof(argv[1]);
            argc--;
            argv++;
            ok = true;
        }
   if ((ok == false) && (strcmp(argv[1], "-stepReduction") == 0)){
            argc--;
            argv++;
            stepReduction = atof(argv[1]);
            argc--;
            argv++;
            ok = true;
        }
        if ((ok == false) && (strcmp(argv[1], "-maxSteps") == 0)){
            argc--;
            argv++;
            maxNumberOfIterations = atoi(argv[1]);
            argc--;
            argv++;
            ok = true;
        }
        if ((ok == false) && (strcmp(argv[1], "-transScale") == 0)){
            argc--;
            argv++;
            translationScale = atof(argv[1]);
            argc--;
            argv++;
            ok = true;
        }
        if ((ok == false) && (strcmp(argv[1], "-rotScale") == 0)){
            argc--;
            argv++;
            rotationScale = atof(argv[1]);
            argc--;
            argv++;
            ok = true;
        } if ((ok == false) && (strcmp(argv[1], "-xScale") == 0)){
            argc--;
            argv++;
            xScale = atof(argv[1]);
            argc--;
            argv++;
            ok = true;
        } if ((ok == false) && (strcmp(argv[1], "-zScale") == 0)){
            argc--;
            argv++;
            zScale = atof(argv[1]);
            argc--;
            argv++;
            ok = true;
        } if ((ok == false) && (strcmp(argv[1], "-yScale") == 0)){
            argc--;
            argv++;
            yScale = atof(argv[1]);
            argc--;
            argv++;
            ok = true;
        }
        if ((ok == false) && (strcmp(argv[1], "-metric") == 0)){
            argc--;
            argv++;
			strcpy(metricName,argv[1]);
            argc--;
            argv++;
            ok = true;
        }
        if ((ok == false) && (strcmp(argv[1], "-optimizer") == 0)){
            argc--;
            argv++;
			strcpy(optimizerName,argv[1]);
            argc--;
            argv++;
            ok = true;
        }
        if ((ok == false) && (strcmp(argv[1], "-best") == 0)){
            argc--;
            argv++;
            useBestResult = true;
            ok = true;
        }
        if ((ok == false) && (strcmp(argv[1], "-useDerivatives") == 0)){
            argc--;
            argv++;
            useExplicitPDFDerivatives = 1;   // correct ?
            ok = true;
        }
        if ((ok == false) && (strcmp(argv[1], "-nBins") == 0)){
            argc--;
            argv++;
            numberOfHistogramBins = atoi(argv[1]);
            argc--;
            argv++;
            ok = true;
        }
        if ((ok == false) && (strcmp(argv[1], "-nSamples") == 0)){
            argc--;
            argv++;
            numberOfSpatialSamples = atoi(argv[1]);
            argc--;
            argv++;
            ok = true;
        }
        if ((ok == false) && (strcmp(argv[1], "-dofin") == 0)){
            argc--;
            argv++;
            dofinName = argv[1];
            argc--;
            argv++;
            ok = true;
        }
        if ((ok == false) && (strcmp(argv[1], "-dofout") == 0)){
            argc--;
            argv++;
            dofoutName = argv[1];
            argc--;
            argv++;
            ok = true;
        }
        if ((ok == false) && (strcmp(argv[1], "-moments") == 0)){
            argc--;
            argv++;
            initWithMoments = true;  
            ok = true;
        }
        if ((ok == false) && (strcmp(argv[1], "-fixedmask") == 0)){
            argc--;
            argv++;
            fixedmaskName = argv[1];
            argc--;
            argv++;
            ok = true;
        }
        if ((ok == false) && (strcmp(argv[1], "-movingmask") == 0)){
            argc--;
            argv++;
            movingmaskName = argv[1];
            argc--;
            argv++;
            ok = true;
        }
        if ((ok == false) && (strcmp(argv[1], "-outputAfter") == 0)){
            argc--;
            argv++;
            movingTransfName = argv[1];
            argc--;
            argv++;
            ok = true;
        }
        if ((ok == false) && (strcmp(argv[1], "-outputCheckAfter") == 0)){
            argc--;
            argv++;
            checkAfterName = argv[1];
            argc--;
            argv++;
            ok = true;
        }
        if ((ok == false) && (strcmp(argv[1], "-outputCheckBefore") == 0)){
            argc--;
            argv++;
            checkBeforeName = argv[1];
            argc--;
            argv++;
            ok = true;
        }
        if (ok == false) {
            std::cout << std::endl << "Option " << argv[1] << " is unknown" << std::endl << std::endl;
            return usage();
        }
    }

	typedef  float    PixelType;
	typedef itk::Image< PixelType, Dimension >  FixedImageType;
	typedef itk::Image< PixelType, Dimension >  MovingImageType;

    typedef itk::ImageMaskSpatialObject< Dimension >   MaskType;
    MaskType::Pointer  spatialObjectFixedMask = MaskType::New();
    MaskType::Pointer  spatialObjectMovingMask = MaskType::New();
    typedef itk::Image< unsigned char, Dimension >   ImageMaskType;
    typedef itk::ImageFileReader< ImageMaskType >    MaskReaderType;
    MaskReaderType::Pointer  maskReaderF = MaskReaderType::New();
    MaskReaderType::Pointer  maskReaderM = MaskReaderType::New();
  
    //  Software Guide : BeginLatex
    //  
    //  The configuration of the registration method in this example closely
    //  follows the procedure in the previous section. The main changes involve the
    //  construction and initialization of the transform. The instantiation of
    //  the transform type requires only the dimension of the space and the
    //  type used for representing space coordinates.
    //  
    //  \index{itk::AffineTransform!Instantiation}
    //
    //  Software Guide : EndLatex 
  
    typedef itk::AffineTransform< double, Dimension > TransformType;


    typedef itk::LinearInterpolateImageFunction< 
        InternalImageType,
        double             > InterpolatorType;
    typedef itk::MeanSquaresImageToImageMetric< 
        InternalImageType, 
        InternalImageType >    SSDMetricType;
    typedef itk::NormalizedCorrelationImageToImageMetric<
        InternalImageType,
        InternalImageType >    NCCMetricType;
    typedef itk::MattesMutualInformationImageToImageMetric< 
        InternalImageType, 
        InternalImageType >    MIMetricType;
    typedef itk::NormalizedMutualInformationHistogramImageToImageMetric< 
        InternalImageType, 
        InternalImageType >    NMIMetricType;
  
    typedef itk::RegularStepGradientDescentOptimizer       RSGDOptimizerType;
    typedef itk::GradientDescentOptimizer                  GDOptimizerType;
    typedef itk::SPSAOptimizer                             SPSAOptimizerType;
    typedef itk::OnePlusOneEvolutionaryOptimizer           EVOptimizerType;
	typedef itk::PowellOptimizer                           PowellOptimizerType;

    typedef RSGDOptimizerType::ScalesType     RSGDOptimizerScalesType;
    typedef GDOptimizerType::ScalesType       GDOptimizerScalesType;
    typedef SPSAOptimizerType::ScalesType     SPSAOptimizerScalesType;
    typedef EVOptimizerType::ScalesType       EVOptimizerScalesType;
	typedef PowellOptimizerType::ScalesType   PowellOptimizerScalesType;

	// for the OnePlusOneEvolutionaryOptimizer
	typedef itk::Statistics::NormalVariateGenerator GeneratorType;
	GeneratorType::Pointer generator = GeneratorType::New() ;

    typedef itk::RecursiveMultiResolutionPyramidImageFilter<
        InternalImageType,
        InternalImageType  >    FixedImagePyramidType;
    typedef itk::RecursiveMultiResolutionPyramidImageFilter<
        InternalImageType,
        InternalImageType  >   MovingImagePyramidType;

    RSGDOptimizerType::Pointer   RSGDoptimizer =   RSGDOptimizerType::New();
    GDOptimizerType::Pointer     GDoptimizer =     GDOptimizerType::New();
    SPSAOptimizerType::Pointer   SPSAoptimizer =   SPSAOptimizerType::New();
    EVOptimizerType::Pointer     EVoptimizer =     EVOptimizerType::New();
	PowellOptimizerType::Pointer PowellOptimizer = PowellOptimizerType::New();

    InterpolatorType::Pointer   interpolator  = InterpolatorType::New();
    RegistrationType::Pointer   registration  = RegistrationType::New();
  
    SSDMetricType::Pointer      SSDmetric     = SSDMetricType::New();
    NCCMetricType::Pointer      NCCmetric     = NCCMetricType::New();
    MIMetricType::Pointer       MImetric      = MIMetricType::New();
    NMIMetricType::Pointer      NMImetric     = NMIMetricType::New();

    NMIMetricType::HistogramType::SizeType histSize;
  
    // read fixed mask if input
    if (fixedmaskName != NULL)
        {
            maskReaderF->SetFileName( fixedmaskName );
            try 
                { 
                    maskReaderF->Update(); 
                } 
            catch( itk::ExceptionObject & err ) 
                { 
                    std::cerr << "ExceptionObject caught !" << std::endl; 
                    std::cerr << err << std::endl; 
                    return EXIT_FAILURE;
                } 
            spatialObjectFixedMask->SetImage( maskReaderF->GetOutput() );
            SSDmetric->SetFixedImageMask( spatialObjectFixedMask );
            NCCmetric->SetFixedImageMask( spatialObjectFixedMask );
            MImetric->SetFixedImageMask( spatialObjectFixedMask );
            NMImetric->SetFixedImageMask( spatialObjectFixedMask );
        }
    if (movingmaskName != NULL)
        {
            maskReaderM->SetFileName( movingmaskName );
            try 
                { 
                    maskReaderM->Update(); 
                } 
            catch( itk::ExceptionObject & err ) 
                { 
                    std::cerr << "ExceptionObject caught !" << std::endl; 
                    std::cerr << err << std::endl; 
                    return EXIT_FAILURE;
                } 
            spatialObjectMovingMask->SetImage( maskReaderM->GetOutput() );
            SSDmetric->SetMovingImageMask( spatialObjectMovingMask );
            NCCmetric->SetMovingImageMask( spatialObjectMovingMask );
            MImetric->SetMovingImageMask( spatialObjectMovingMask );
            NMImetric->SetMovingImageMask( spatialObjectMovingMask );
        }
  
	// default
	GDoptimizer->MinimizeOn();
	RSGDoptimizer->MinimizeOn();
	SPSAoptimizer->MinimizeOn();
	EVoptimizer->MaximizeOff();
	PowellOptimizer->SetMaximize(false);

	if (! strcmp(metricName, "SSD"))
		{
			registration->SetMetric( SSDmetric );
			std::cout << "Using SSD image similarity" << std::endl;  
		}
	else
		{
			if (! strcmp(metricName, "MI"))
				{
					registration->SetMetric( MImetric );
					std::cout << "Using Matts MI image similarity" << std::endl;  
					MImetric->ReinitializeSeed( 76926294 );
					MImetric->SetNumberOfHistogramBins( numberOfHistogramBins );
					MImetric->SetNumberOfSpatialSamples( numberOfSpatialSamples );
					if (useExplicitPDFDerivatives > 0)
						MImetric->SetUseExplicitPDFDerivatives( useExplicitPDFDerivatives );
				}
			else
				{
					if (! strcmp(metricName, "NMI"))
						{
							registration->SetMetric( NMImetric );
							std::cout << "Using NMI image similarity" << std::endl;  
							histSize[0] = numberOfHistogramBins;
							histSize[1] = numberOfHistogramBins;
							NMImetric->SetHistogramSize(histSize);                                  
							GDoptimizer->MaximizeOn();
							RSGDoptimizer->MaximizeOn();
							SPSAoptimizer->MaximizeOn();
							EVoptimizer->MaximizeOn();	
							PowellOptimizer->SetMaximize(true);
							MinimizeFlag = false;
                            overallbestValue = 0;    
						}
					else
						{
							if (! strcmp(metricName, "NCC"))
								{
									registration->SetMetric( NCCmetric );
									std::cout << "Using NCC image similarity" << std::endl;  
								}
							else
								{
									std::cout << "Unknown image similarity " << metricName << std::endl;  
									std::cout << "Using NCC instead" << std::endl;  
									registration->SetMetric( NCCmetric );
								}
						}
				}
		}
          
	// Create the Command observer and register it with the optimizer.
	//
	CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
	
	if (! strcmp(optimizerName, "GD"))
		{
			registration->SetOptimizer(     GDoptimizer     );
			std::cout << "Using GradientDescent (GD) optimizer" << std::endl;  
			GDoptimizer->SetNumberOfIterations(  maxNumberOfIterations  );
			GDoptimizer->AddObserver( itk::IterationEvent(), observer );
		}
	else 
		{
			if (! strcmp(optimizerName, "SPSA"))
				{
					registration->SetOptimizer(     SPSAoptimizer     );
					std::cout << "Using Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer" << std::endl;  
					SPSAoptimizer->SetMaximumNumberOfIterations(  maxNumberOfIterations  );
					SPSAoptimizer->SetA( 0 );
					SPSAoptimizer->SetAlpha( 1.0 );
					SPSAoptimizer->AddObserver( itk::IterationEvent(), observer );
				}
			else
				{
					if (! strcmp(optimizerName, "EV"))
						{
							registration->SetOptimizer(     EVoptimizer     );
							std::cout << "Using OnePlusOneEvolutionary (EV) optimizer" << std::endl;  
							EVoptimizer->SetMaximumIteration(  maxNumberOfIterations  );
							EVoptimizer->AddObserver( itk::IterationEvent(), observer );
							
							generator->Initialize( 20020702 ) ;
							EVoptimizer->SetNormalVariateGenerator( generator ) ;
							
							// default is
							//double initRadius = 1.01;
							//double grow = 1.05 ;
							//EVoptimizer->SetEpsilon(0.00015) ; // minimal search radius
							//
							//double initRadius = 1.25;
							//double grow = 1.5 ;
							//EVoptimizer->SetEpsilon(1.0e-6) ; // minimal search radius

							int n2=numberOfLevels-1;
							double mfactor1 = pow(1.5, n2);
							double initRadius = 1.01 * mfactor1;
							double grow = 1.05 * mfactor1;
							double shrink = pow(grow, -0.25) ;
							EVoptimizer->Initialize(initRadius, grow, shrink) ;
							EVoptimizer->SetEpsilon(0.00015) ; // minimal search radius
							EVoptimizer->Print(std::cout);
						}
					else
						{
							if (! strcmp(optimizerName, "Powell"))
								{
									registration->SetOptimizer(     PowellOptimizer     );
									std::cout << "Using Powell optimizer" << std::endl;  
									
									PowellOptimizer->SetMaximumLineIteration(  maxNumberOfIterations  );
									int n2=numberOfLevels-1;
									double mfactor1 = pow(2.0, n2);
									double mfactor2 = pow(5.0, n2);
									PowellOptimizer->SetStepLength( steplength * mfactor1 );
									PowellOptimizer->SetStepTolerance(  0.0001 * mfactor2 );
									PowellOptimizer->AddObserver( itk::IterationEvent(), observer );
									PowellOptimizer->Print(std::cout);
								}
							else
								{
									// default is RSGD
									registration->SetOptimizer(     RSGDoptimizer     );
									if (! strcmp(optimizerName, "RSGD"))
										{
											std::cout << "Using RegularStepGradientDescent (RSGD) optimizer" << std::endl;  
										}
									else
										{
											std::cout << "Unknown image optimizer " << optimizerName << std::endl;  
											std::cout << "Using RegularStepGradientDescent (RSGD) optimizer instead" << std::endl; 
										}
									RSGDoptimizer->SetNumberOfIterations(  maxNumberOfIterations  );
									RSGDoptimizer->SetRelaxationFactor( 0.8 );
									int n2=numberOfLevels-1;
									double mfactor1 = pow(2.0, n2);
									double mfactor2 = pow(5.0, n2);
									RSGDoptimizer->SetMaximumStepLength( steplength * mfactor1 );
									RSGDoptimizer->SetMinimumStepLength(  0.0001 * mfactor2 );
									
									RSGDoptimizer->AddObserver( itk::IterationEvent(), observer );
									RSGDoptimizer->Print(std::cout);
								}
						}
				}
		}

    registration->SetInterpolator(  interpolator  );


    //  Software Guide : BeginLatex
    //
    //  The transform is constructed using the standard \code{New()} method and
    //  assigning it to a SmartPointer.
    //
    //  \index{itk::AffineTransform!New()}
    //  \index{itk::AffineTransform!Pointer}
    //  \index{itk::Multi\-Resolution\-Image\-Registration\-Method!SetTransform()}
    //
    //  Software Guide : EndLatex 

    TransformType::Pointer   transform  = TransformType::New();

    registration->SetTransform( transform );

    FixedImagePyramidType::Pointer fixedImagePyramid = 
        FixedImagePyramidType::New();
    MovingImagePyramidType::Pointer movingImagePyramid =
        MovingImagePyramidType::New();


    typedef itk::ImageFileReader< FixedImageType  > FixedImageReaderType;
    typedef itk::ImageFileReader< MovingImageType > MovingImageReaderType;

    FixedImageReaderType::Pointer  fixedImageReader  = FixedImageReaderType::New();
    MovingImageReaderType::Pointer movingImageReader = MovingImageReaderType::New();

    fixedImageReader->SetFileName(  fixedName );
    movingImageReader->SetFileName( movingName );

    typedef itk::CastImageFilter< 
        FixedImageType, InternalImageType > FixedCastFilterType;
    typedef itk::CastImageFilter< 
        MovingImageType, InternalImageType > MovingCastFilterType;
    FixedCastFilterType::Pointer fixedCaster   = FixedCastFilterType::New();
    MovingCastFilterType::Pointer movingCaster = MovingCastFilterType::New();

    fixedCaster->SetInput(  fixedImageReader->GetOutput() );
    movingCaster->SetInput( movingImageReader->GetOutput() );

    registration->SetFixedImage(    fixedCaster->GetOutput()    );
    registration->SetMovingImage(   movingCaster->GetOutput()   );

    fixedCaster->Update();

    registration->SetFixedImageRegion( 
                                      fixedCaster->GetOutput()->GetBufferedRegion() );


    //  One of the easiest ways of preparing a consistent set of parameters for
    //  the transform is to use the \doxygen{CenteredTransformInitializer}. Once
    //  the transform is initialized, we can invoke its \code{GetParameters()}
    //  method to extract the array of parameters. Finally the array is passed to
    //  the registration method using its \code{SetInitialTransformParameters()}
    //  method.

    typedef itk::CenteredTransformInitializer< 
        TransformType, 
        FixedImageType, 
        MovingImageType >  TransformInitializerType;
  

    TransformInitializerType::Pointer initializer = TransformInitializerType::New();


    itk::TransformFileReader::Pointer affineReader;
    affineReader = itk::TransformFileReader::New();
    typedef itk::TransformFileReader::TransformListType * TransformListType;
  
    if( dofinName != NULL )
        {
            affineReader->SetFileName( dofinName);
            affineReader->Update();
            TransformListType transforms = affineReader->GetTransformList();
            std::cout << "Number of transforms = " << transforms->size() << std::endl;
            itk::TransformFileReader::TransformListType::const_iterator it = transforms->begin();
            if(!strcmp((*it)->GetNameOfClass(),"AffineTransform"))
                {
                    transform = static_cast<TransformType*>((*it).GetPointer());
                }
            else
                {
                    std::cout << "Input transformation of type " <<  (*it)->GetNameOfClass() << std::endl;
                }
        }
    else
        {
            transform->SetIdentity();
            initializer->SetTransform(   transform );
            initializer->SetFixedImage(  fixedImageReader->GetOutput() );
            initializer->SetMovingImage( movingImageReader->GetOutput() );
          
            if (initWithMoments)
                {  
                    initializer->MomentsOn();
                    initializer->InitializeTransform();
                }
        }
  

    registration->SetInitialTransformParameters( transform->GetParameters() );

	ParametersType overallbestParameters( transform->GetNumberOfParameters() );
	overallbestParameters = transform->GetParameters();

    std::cout << "Initial pars. " << overallbestParameters << std::endl;
	

    //  The set of parameters in the AffineTransform have different
    //  dynamic ranges. Typically the parameters associated with the matrix
    //  have values around $[-1:1]$, although they are not restricted to this
    //  interval.  Parameters associated with translations, on the other hand,
    //  tend to have much higher values, typically in the order of $10.0$ to
    //  $100.0$. This difference in dynamic range negatively affects the
    //  performance of gradient descent optimizers. ITK provides a mechanism to
    //  compensate for such differences in values among the parameters when
    //  they are passed to the optimizer. The mechanism consists of providing an
    //  array of scale factors to the optimizer. These factors re-normalize the
    //  gradient components before they are used to compute the step of the
    //  optimizer at the current iteration. In our particular case, a common
    //  choice for the scale parameters is to set to $1.0$ all those associated
    //  with the matrix coefficients, that is, the first $N \times N$
    //  factors. Then, we set the remaining scale factors to a small value. The
    //  following code sets up the scale coefficients.

    RSGDOptimizerScalesType RSGDoptimizerScales( transform->GetNumberOfParameters() );
    GDOptimizerScalesType   GDoptimizerScales( transform->GetNumberOfParameters() );
    SPSAOptimizerScalesType SPSAoptimizerScales( transform->GetNumberOfParameters() );
    EVOptimizerScalesType   EVoptimizerScales( transform->GetNumberOfParameters() );
	PowellOptimizerScalesType PowellOptimizerScales( transform->GetNumberOfParameters() );


	if (! strcmp(optimizerName, "GD"))
		{
			GDoptimizerScales.Fill(1.0);
			GDoptimizerScales[4]  =  translationScale;
			GDoptimizerScales[5] =  translationScale;
			GDoptimizer->SetScales( GDoptimizerScales );
			GDoptimizer->Print(std::cout);
		}
	else 
		{
			if (! strcmp(optimizerName, "SPSA"))
				{
                    SPSAoptimizerScales.Fill(1.0);
					SPSAoptimizerScales[4]  =  translationScale;
					SPSAoptimizerScales[5] =  translationScale;
					SPSAoptimizer->SetScales( SPSAoptimizerScales );
					SPSAoptimizer->SetInitialPosition( overallbestParameters );
					SPSAoptimizer->Print(std::cout);
				}
			else
				{
					if (! strcmp(optimizerName, "EV"))
						{
							EVoptimizerScales.Fill(1.0/translationScale);
							EVoptimizerScales[4]  =  1.0;
							EVoptimizerScales[5] =  1.0;
							EVoptimizer->SetScales( EVoptimizerScales );
							EVoptimizer->SetInitialPosition( overallbestParameters );
							EVoptimizer->Print(std::cout);
						}
					else
						{
							if (! strcmp(optimizerName, "Powell"))
								{
									PowellOptimizerScales.Fill(1.0/translationScale);
									PowellOptimizerScales[4]  =  1.0;
									PowellOptimizerScales[5] =  1.0;
									PowellOptimizer->SetScales( PowellOptimizerScales );
									PowellOptimizer->SetInitialPosition( overallbestParameters );
									PowellOptimizer->Print(std::cout);
								}
							else
								{
									RSGDoptimizerScales.Fill(1.0);
                                    RSGDoptimizerScales[0] =  rotationScale*xScale;
                                    RSGDoptimizerScales[1] =  rotationScale;
									RSGDoptimizerScales[2] =  rotationScale;
									RSGDoptimizerScales[3] =  rotationScale;
									RSGDoptimizerScales[4] =  rotationScale*yScale;
									RSGDoptimizerScales[5] =  rotationScale;
									RSGDoptimizerScales[6] =  rotationScale;
									RSGDoptimizerScales[7] =  rotationScale;
									RSGDoptimizerScales[8] =  rotationScale*zScale;
									RSGDoptimizerScales[9]  =  translationScale;
									RSGDoptimizerScales[10] =  translationScale;
									RSGDoptimizerScales[11] =  translationScale;
									RSGDoptimizer->SetScales( RSGDoptimizerScales );
									RSGDoptimizer->Print(std::cout);
								}
						}
				}
		}

    // Define whether to calculate the metric derivative by explicitly
    // computing the derivatives of the joint PDF with respect to the Transform
    // parameters, or doing it by progressively accumulating contributions from
    // each bin in the joint PDF.


    //  The step length has to be proportional to the expected values of the
    //  parameters in the search space. Since the expected values of the matrix
    //  coefficients are around $1.0$, the initial step of the optimization
    //  should be a small number compared to $1.0$. As a guideline, it is
    //  useful to think of the matrix coefficients as combinations of
    //  $cos(\theta)$ and $sin(\theta)$.  This leads to use values close to the
    //  expected rotation measured in radians. For example, a rotation of $1.0$
    //  degree is about $0.017$ radians. As in the previous example, the
    //  maximum and minimum step length of the optimizer are set by the
    //  \code{RegistrationInterfaceCommand} when it is called at the beginning
    //  of registration at each multi-resolution level.



    // Create the Command interface observer and register it with the optimizer.
    //
    typedef RegistrationInterfaceCommand<RegistrationType> CommandType;
    CommandType::Pointer command = CommandType::New();
    registration->AddObserver( itk::IterationEvent(), command );
    registration->SetNumberOfLevels( numberOfLevels );

	std::cout << "START REGISTRATION" << std::endl;
  
    try 
        { 
            registration->StartRegistration(); 
        } 
    catch( itk::ExceptionObject & err ) 
        { 
            std::cout << "ExceptionObject caught !" << std::endl; 
            std::cout << err << std::endl; 
            return EXIT_FAILURE;
        } 

    ParametersType finalParameters(transform->GetNumberOfParameters());

	if (useBestResult)
		{
			finalParameters = overallbestParameters;
		}
	else
		{
			finalParameters = registration->GetLastTransformParameters();
		}
  
    double TranslationAlongX = finalParameters[4];
    double TranslationAlongY = finalParameters[5];

  

    unsigned int numberOfIterations;
    double bestValue;

	std::cout << "Optimizer Stopping Condition = ";

	if (! strcmp(optimizerName, "GD"))
		{
			std::cout << GDoptimizer->GetStopCondition() << std::endl;
			bestValue = GDoptimizer->GetValue();
			numberOfIterations = GDoptimizer->GetCurrentIteration();
		}
	else
		{
			if (! strcmp(optimizerName, "SPSA"))
				{
					std::cout << SPSAoptimizer->GetStopCondition() << std::endl;
					bestValue = SPSAoptimizer->GetValue();
					numberOfIterations = SPSAoptimizer->GetCurrentIteration();
				}
			else
				{
					if (! strcmp(optimizerName, "EV"))
						{
							bestValue = EVoptimizer->GetValue();
							numberOfIterations = EVoptimizer->GetCurrentIteration();
						}
					else
						{
							if (! strcmp(optimizerName, "Powell"))
								{
									bestValue = PowellOptimizer->GetValue();
									numberOfIterations = PowellOptimizer->GetCurrentIteration();
								}
							else
								{
									std::cout << RSGDoptimizer->GetStopCondition() << std::endl;
									bestValue = RSGDoptimizer->GetValue();
									numberOfIterations = RSGDoptimizer->GetCurrentIteration();
								}
						}
				}
		}



    // Print out results
    //
    std::cout << "Result = " << std::endl;
    std::cout << " Translation X = " << TranslationAlongX  << std::endl;
    std::cout << " Translation Y = " << TranslationAlongY  << std::endl;
    std::cout << " Iterations    = " << numberOfIterations << std::endl;
    std::cout << " Metric value  = " << bestValue          << std::endl;

    vnl_matrix<double> p(2, 2);
    p[0][0] = (double) finalParameters[0];
    p[0][1] = (double) finalParameters[1];
    p[1][0] = (double) finalParameters[2];
    p[1][1] = (double) finalParameters[3];
    vnl_svd<double> svd(p);
    vnl_matrix<double> r(2, 2);
    r = svd.U() * vnl_transpose(svd.V());
    //double angle = asin(r[1][0]);
  
    std::cout << " Scale 1         = " << svd.W(0)                 << std::endl;
    std::cout << " Scale 2         = " << svd.W(1)                 << std::endl;
    //std::cout << " Angle (degrees) = " << angle * 45.0 / atan(1.0) << std::endl;
  
    double detA = p[0][0]*p[1][1] - p[0][1]*p[1][0];
    std::cout << " det(A) = " << detA << std::endl;

    TransformType::Pointer finalTransform = TransformType::New();

  
    finalTransform->SetCenter( transform->GetCenter() );
    finalTransform->SetParameters( finalParameters );
  
    itk::TransformFileWriter::Pointer affineWriter;
    affineWriter = itk::TransformFileWriter::New();
    if( dofoutName != NULL )
        {
            affineWriter->SetFileName( dofoutName );
            affineWriter->SetInput( finalTransform   );
            affineWriter->Update();
        }
  
    typedef itk::ResampleImageFilter< 
        MovingImageType, 
        FixedImageType >    ResampleFilterType;

    ResampleFilterType::Pointer resample = ResampleFilterType::New();

    resample->SetTransform( finalTransform );
    resample->SetInput( movingImageReader->GetOutput() );

    FixedImageType::Pointer fixedImage = fixedImageReader->GetOutput();

    PixelType backgroundGrayLevel = 0;

    resample->SetSize(    fixedImage->GetLargestPossibleRegion().GetSize() );
    resample->SetOutputOrigin(  fixedImage->GetOrigin() );
    resample->SetOutputSpacing( fixedImage->GetSpacing() );
    resample->SetOutputDirection( fixedImage->GetDirection() );
    resample->SetDefaultPixelValue( backgroundGrayLevel );


    typedef  signed short  OutputPixelType;
    typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
    typedef itk::CastImageFilter< 
        FixedImageType,
        OutputImageType > CastFilterType;
    typedef itk::ImageFileWriter< OutputImageType >  WriterType;

    WriterType::Pointer      writer =  WriterType::New();
    CastFilterType::Pointer  caster =  CastFilterType::New();

    if (movingTransfName != NULL)
        {
            writer->SetFileName( movingTransfName );

            caster->SetInput( resample->GetOutput() );
            writer->SetInput( caster->GetOutput()   );
            writer->Update();
        }
  
    //
    // Generate checkerboards before and after registration
    //
    typedef itk::CheckerBoardImageFilter< FixedImageType > CheckerBoardFilterType;

    CheckerBoardFilterType::Pointer checker = CheckerBoardFilterType::New();

    checker->SetInput1( fixedImage );
    checker->SetInput2( resample->GetOutput() );

    caster->SetInput( checker->GetOutput() );
    writer->SetInput( caster->GetOutput()   );
  
    resample->SetDefaultPixelValue( 0 );

    // Write out checkerboard outputs
    // Before registration
    TransformType::Pointer identityTransform = TransformType::New();
    identityTransform->SetIdentity();
    resample->SetTransform( identityTransform );

    if( checkBeforeName != NULL )
        {
            writer->SetFileName( checkBeforeName );
            writer->Update();
        }

 
    // After registration
    resample->SetTransform( finalTransform );
    if( checkAfterName != NULL )
        {
            writer->SetFileName( checkAfterName );
            writer->Update();
        }


    return EXIT_SUCCESS;
}

