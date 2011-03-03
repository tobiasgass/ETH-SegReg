/*
 * Classifier.h
 *
 *  Created on: Feb 14, 2011
 *      Author: gasst
 */

#ifndef CLASSIFIER_H_
#define CLASSIFIER_H_
#include <vector>
#include "data.h"
#include "forest.h"
#include "randomnaivebayes.h"
#include "pairforest.h"
#include "tree.h"
#include "data.h"
#include "utilities.h"
#include "hyperparameters.h"
#include <libconfig.h++>
#include <boost/numeric/ublas/matrix.hpp>
#include <iostream>
#include "itkImageDuplicator.h"
#include "itkConstNeighborhoodIterator.h"
#include <time.h>
#include <itkImageRandomConstIteratorWithIndex.h>



using namespace std;
using namespace boost::numeric::ublas;
using namespace libconfig;

template<class ImageType>
class segmentationClassifier{
protected:
	FileData m_TrainData;
	Forest * m_Forest;
	std::vector<double> m_weights;
	std::vector<int> m_labelVector;
	matrix<float> m_data;
	matrix<float>m_conf;
	int m_nData;
public:
	typedef typename ImageType::Pointer ImagePointerType;
	typedef typename itk::ImageDuplicator< ImageType > DuplicatorType;
	typedef typename itk::Image<itk::Vector<float,2> ,ImageType::ImageDimension> ProbImageType;
	typedef typename ProbImageType::Pointer ProbImagePointerType;
public:
	segmentationClassifier(){
		m_data=matrix<float>(1,3);
		m_conf=matrix<float>(1,2);
		m_labelVector=std::vector<int>(1);
	};
	void freeMem(){
		delete m_Forest;
		m_data=matrix<float>(0,0);
		m_conf=matrix<float>(0,0);
	}
	void save(string filename){
		m_Forest->save(filename);
	}
	void load(string filename){
		m_Forest->load(filename);
	}
	void setData(ImagePointerType intensities, ImagePointerType labels){
		long int nData=1;
		for (int d=0;d<ImageType::ImageDimension;++d)
			nData*=intensities->GetLargestPossibleRegion().GetSize()[d];
		int nFeatures=3;
		matrix<float> data(nData,nFeatures);
		std::vector<int> labelVector(nData);
		typedef typename itk::ConstNeighborhoodIterator< ImageType > NeighborhoodIteratorType;
		typename NeighborhoodIteratorType::RadiusType radius;
		radius.Fill(5);
		NeighborhoodIteratorType ImageIterator(radius,intensities, intensities->GetLargestPossibleRegion());
		itk::ImageRegionIteratorWithIndex<ImageType> ImageIterator2(intensities, intensities->GetLargestPossibleRegion());
		itk::ImageRegionIteratorWithIndex<ImageType> LabelIterator2(labels, labels->GetLargestPossibleRegion());
		NeighborhoodIteratorType LabelIterator(radius,labels, labels->GetLargestPossibleRegion());
		long int i=0;
		std::vector<int> counts(2,0);
		for (ImageIterator.GoToBegin(),ImageIterator2.GoToBegin(), LabelIterator2.GoToBegin(),
				LabelIterator.GoToBegin();
				!ImageIterator.IsAtEnd() ;
				++ImageIterator,++LabelIterator,++ImageIterator2,++LabelIterator2)
		{

			assert( !LabelIterator.IsAtEnd());
			float centralIntens=1.0*ImageIterator.GetCenterPixel()/65535;
			float centralLabel=LabelIterator.GetCenterPixel()>0;

			if (centralLabel || (counts[1] && 1.0*counts[1]/(counts[1]+counts[0]) > 0.5 )){
				double intens=1.0*ImageIterator2.Get()/65535;
				//			std::cout<<intens<<" "<<centralIntens<<" "<<ImageIterator2.Get()<<std::endl;
				//			data(i,0)=LabelIterator2.Get();
				data(i,0)=intens;
				data(i,1)=intens*intens;
				data(i,2)=fabs(intens);
				labelVector[i]=LabelIterator2.Get()>0;
				counts[labelVector[i]]++;
				//			std::cout<<data(i,0)<<" "<<labelVector[i]<<std::endl;
				i++;
			}

		}
		//		std::cout<<i<<" "<<counts[0]<<" "<<counts[1]<<" "<<1.0*counts[1]/(counts[1]+counts[0])<<std::endl;

		data.resize(i,nFeatures);
		std::vector<int> copy=labelVector;
		labelVector.resize(i);

		m_nData=i;
		std::vector<double> weights(labelVector.size());
		for (i=0;i<labelVector.size();++i){
			if (copy[i]!=labelVector[i]) std::cout<<"bah";
			//			weights[i]=1.0/counts[labelVector[i]];
			weights[i]=1.0;
		}
		m_weights=weights;
		std::cout<<"done adding data. "<<std::endl;
		m_TrainData.setData(data);
		m_TrainData.setLabels(labelVector);



	};
	ImagePointerType eval(ImagePointerType intensities, ImagePointerType labels, ProbImagePointerType & probabilities){

		long int nData=1;
		for (int d=0;d<ImageType::ImageDimension;++d)
			nData*=intensities->GetLargestPossibleRegion().GetSize()[d];

		int nFeatures=3;
		matrix<float> data(nData,nFeatures);
		std::vector<int> labelVector(nData);
		itk::ImageRegionIteratorWithIndex<ImageType> ImageIterator(intensities, intensities->GetLargestPossibleRegion());
		itk::ImageRegionIteratorWithIndex<ImageType> LabelIterator(labels, labels->GetLargestPossibleRegion());
		long int i=0;
		for (ImageIterator.GoToBegin(), LabelIterator.GoToBegin();
				!ImageIterator.IsAtEnd();
				++ImageIterator,++LabelIterator)
		{
			double intens=1.0*ImageIterator.Get()/65535;
			data(i,0)=intens;
			data(i,1)=intens*intens;
			data(i,2)=fabs(intens);
			labelVector[i]=LabelIterator.Get()>0;
			i++;
		}
		m_Forest->eval(data,labelVector,false);
		matrix<float> conf = m_Forest->getConfidences();
		//		(*probabilities)=matrix<double>(nData,2);
		//		for (int i=0;i<nData;++i){
		//			(*probabilities)(i,0)=conf(i,0)/(conf(i,0)+conf(i,1));
		//			(*probabilities)(i,1)=conf(i,1)/(conf(i,0)+conf(i,1));
		//			//			std::cout<<conf(n,0)<<"/"<<conf(n,1)<<std::endl;
		//		}
		std::vector<int> predictions=m_Forest->getPredictions();

		typename DuplicatorType::Pointer duplicator = DuplicatorType::New();
		duplicator->SetInputImage(intensities);
		duplicator->Update();
		ImagePointerType returnImage=duplicator->GetOutput();
		duplicator->Update();
		probabilities=ProbImageType::New();
		probabilities->SetRegions(intensities->GetLargestPossibleRegion());
		probabilities->Allocate();
		itk::ImageRegionIteratorWithIndex<ImageType> LabelImageIterator(returnImage,returnImage->GetLargestPossibleRegion());
		i=0;
		itk::ImageRegionIteratorWithIndex<ProbImageType> probImageIterator(probabilities,probabilities->GetLargestPossibleRegion());

		for (probImageIterator.GoToBegin(),LabelImageIterator.GoToBegin();!LabelImageIterator.IsAtEnd();++i,++LabelImageIterator,++probImageIterator){
			//			LabelImageIterator.Set(predictions[i]*65535);
			double tissue=conf(i,1)/(conf(i,0)+conf(i,1));
			double notTissue=fabs(1-tissue);
			double thresh=0.0;
			//			tissue=tissue>thresh?tissue:0;
			//			notTissue=notTissue>thresh?notTissue:0;
//			tissue=tissue>thresh?1.0:tissue;
//			notTissue=notTissue<thresh?notTissue:1.0;
			itk::Vector<float,2> probs;
			probs[1]=tissue;
			probs[0]=notTissue;
			LabelImageIterator.Set(tissue*65535);
			probImageIterator.Set(probs);//conf(i,1)/(conf(i,0)+conf(i,1)));
			//			std::cout<<conf(i,0)/(conf(i,0)+conf(i,1))<<std::endl;
			//			std::cout<<predictions[i]<<" "<<i<<std::endl;
		}
		//		return returnImage;
		return returnImage;

	}

	void train(){
		std::cout<<"reading config"<<std::endl;
		string confFile("/home/gasst/work/progs/rf/randomForest.conf");///home/gasst/work/progs/rf/src/randomForest.conf");
		HyperParameters hp;
		Config configFile;

		configFile.readFile(confFile.c_str());

		// DATA
		hp.trainData = (const char*) configFile.lookup("Data.trainData");
		hp.trainLabels = (const char*) configFile.lookup("Data.trainLabels");
		hp.testData = (const char*) configFile.lookup("Data.testData");
		hp.testLabels = (const char*) configFile.lookup("Data.testLabels");
		hp.numLabeled = m_nData;//configFile.lookup("Data.numLabeled");
		hp.numClasses = configFile.lookup("Data.numClasses");

		// TREE
		hp.maxTreeDepth = configFile.lookup("Tree.maxDepth");
		hp.bagRatio = configFile.lookup("Tree.bagRatio");
		hp.numRandomFeatures = configFile.lookup("Tree.numRandomFeatures");
		hp.numProjFeatures = configFile.lookup("Tree.numProjFeatures");
		hp.useRandProj = configFile.lookup("Tree.useRandProj");
		hp.useGPU = configFile.lookup("Tree.useGPU");
		hp.useSubSamplingWithReplacement = configFile.lookup("Tree.subSampleWR");
		hp.verbose = configFile.lookup("Tree.verbose");
		hp.useInfoGain = configFile.lookup("Tree.useInfoGain");


		// FOREST
		hp.numTrees = configFile.lookup("Forest.numTrees");
		hp.useSoftVoting = configFile.lookup("Forest.useSoftVoting");
		hp.saveForest = configFile.lookup("Forest.saveForest");

		std::cout<<"creating forest"<<std::endl;
		m_Forest= new Forest(hp);
		std::cout<<"training forest"<<std::endl;
		m_Forest->train(m_TrainData.getData(),m_TrainData.getLabels(),m_weights);//	,m_weights);
		std::cout<<"done"<<std::endl;
	};
	double posterior(float intensity, int label){

		double intens=1.0*intensity/65535;
		m_data(0,0)=intens;
		m_data(0,1)=intens*intens;
		m_data(0,2)=fabs(intens);

		m_Forest->eval(m_data,m_labelVector,false);

		m_conf = m_Forest->getConfidences();
		return m_conf(0,label)/(m_conf(0,0)+m_conf(0,1));
	}


};

template<class ImageType>
class pairwiseSegmentationClassifier : public segmentationClassifier<ImageType>{
private:
	int m_radius;
public:
	typedef typename ImageType::Pointer ImagePointerType;
	typedef typename itk::ImageDuplicator< ImageType > DuplicatorType;
public:
	pairwiseSegmentationClassifier(){

	}
	void setData(ImagePointerType intensities, ImagePointerType labels){
		m_radius=5;
		//maximal size
		long int nData=1;
		for (int d=0;d<ImageType::ImageDimension;++d)
			nData*=intensities->GetLargestPossibleRegion().GetSize()[d];
		std::cout<<nData<<std::endl;
		nData*=pow(1.0*2*m_radius+1,ImageType::ImageDimension);
		std::cout<<nData<<" computed"<<std::endl;
		int nFeatures=10;
		matrix<float> data(nData,nFeatures);
		std::cout<<nData<<" matrix allocated"<<std::endl;
		std::vector<int> labelVector(nData);
		typedef typename itk::ConstNeighborhoodIterator< ImageType > NeighborhoodIteratorType;
		typename NeighborhoodIteratorType::RadiusType radius;
		radius.Fill(m_radius);
		NeighborhoodIteratorType ImageIterator(radius,intensities, intensities->GetLargestPossibleRegion());
		itk::ImageRegionIteratorWithIndex<ImageType> ImageIterator2(intensities, intensities->GetLargestPossibleRegion());
		itk::ImageRegionIteratorWithIndex<ImageType> LabelIterator2(labels, labels->GetLargestPossibleRegion());
		NeighborhoodIteratorType LabelIterator(radius,labels, labels->GetLargestPossibleRegion());
		long int i=0;
		std::vector<int> counts(2,0);
		for (ImageIterator.GoToBegin(),ImageIterator2.GoToBegin(), LabelIterator2.GoToBegin(),
				LabelIterator.GoToBegin();
				!ImageIterator.IsAtEnd() ;
				++ImageIterator,++LabelIterator,++ImageIterator2,++LabelIterator2)
		{
			assert( !LabelIterator.IsAtEnd());
			float centralIntens=1.0*ImageIterator.GetCenterPixel()/65535;
			float centralLabel=LabelIterator.GetCenterPixel()>0;
			if (centralLabel || (counts[1] && 1.0*counts[1]/(counts[1]+counts[0]) > 0.5 )){
				for (int r=0;r<ImageIterator.Size();++r,++i){
					float intens=1.0*ImageIterator.GetPixel(r)/65535;
					int label=LabelIterator.GetPixel(r)>0;
					data(i,0)=centralIntens;
					data(i,1)=intens;
					data(i,2)=data(i,0)-data(i,1);
					data(i,3)=fabs(data(i,0)-data(i,1));
					data(i,4)=(data(i,3))*(data(i,3));
					data(i,6)=data(i,0)*data(i,0);
					data(i,7)=data(i,0)*data(i,1);
					data(i,8)=fabs(data(i,7));
					data(i,9)=centralLabel;
					labelVector[i]=label;
					counts[label]++;
					//				std::cout<<labelVector[i]<<" "<<label<<" "<<intens<<" "<<centralIntens<<std::endl;
				}
			}
		}
		std::cout<<counts[0]<<" "<<counts[1]<<" "<<1.0*counts[1]/(counts[1]+counts[0])<<std::endl;
		data.resize(i,nFeatures);
		std::vector<int> copy=labelVector;
		labelVector.resize(i);
		this->m_nData=i;


		std::vector<double> weights(labelVector.size());

		for (i=0;i<labelVector.size();++i){
			weights[i]=1.0;
			//			weights[i]=1.0/counts[labelVector[i]];
		}
		this->m_weights=weights;
		std::cout<<"done adding data. "<<std::endl;
		this->m_TrainData.setData(data);
		this->m_TrainData.setLabels(labelVector);



	};
	void eval(float * probs, int nIntensities){
		int nFeatures=10;
		matrix<float> data(nIntensities*nIntensities*2,nFeatures);
		std::vector<int> labelVector(nIntensities*nIntensities*2);
		int idx=0;
		for (int i1=0;i1<nIntensities;++i1){
			for (int i2=0;i2<nIntensities;++i2){
				for (int s=0;s<2;++s,++idx){
					data(idx,0)=1.0/nIntensities*i1;
					data(idx,1)=1.0/nIntensities*i2;
					data(idx,2)=data(idx,0)-data(idx,1);
					data(idx,3)=fabs(data(idx,0)-data(idx,1));
					data(idx,4)=(data(idx,3))*(data(idx,3));
					data(idx,6)=data(idx,0)*data(idx,0);
					data(idx,7)=data(idx,0)*data(idx,1);
					data(idx,8)=fabs(data(idx,7));
					data(idx,9)=s;
					labelVector[idx]=0;
				}

			}
		}
		this->m_Forest->eval(data,labelVector,false);
		matrix<float> conf = this->m_Forest->getConfidences();
		idx=0;
		for (int i1=0;i1<nIntensities;++i1){
			for (int i2=0;i2<nIntensities;++i2){
				for (int s=0;s<2;++s,++idx){
					//					std::cout<<i1<<" "<<i2<<" "<<s<<" "<<conf(idx,1)<<std::endl;
					probs[idx]=conf(idx,1);
				}
			}
		}

	}
	ImagePointerType eval(matrix<float> &data, std::vector<int> &labelVector, matrix<float> * probabilities){
		this->m_Forest->eval(data,labelVector,false);
		matrix<float> conf = this->m_Forest->getConfidences();
		(*probabilities)=matrix<double>(data.size1(),2);
		for (int i=0;i<data.size1();++i){
			//			std::cout<<conf(i,0)<<" "<<conf(i,1)<<std::endl;
			(*probabilities)(i,0)=conf(i,0)/(conf(i,0)+conf(i,1));
			(*probabilities)(i,1)=conf(i,1)/(conf(i,0)+conf(i,1));
		}
		return NULL;

	}
};
template<class ImageType>
class truePairwiseSegmentationClassifier : public segmentationClassifier<ImageType>{
private:
	int m_radius;
public:
	typedef typename ImageType::Pointer ImagePointerType;
	typedef typename itk::ImageDuplicator< ImageType > DuplicatorType;
public:
	truePairwiseSegmentationClassifier(){

	}
	void train(){
		std::cout<<"reading config"<<std::endl;
		string confFile("/home/gasst/work/progs/rf/randomForest.conf");///home/gasst/work/progs/rf/src/randomForest.conf");
		HyperParameters hp;
		Config configFile;

		configFile.readFile(confFile.c_str());

		// DATA
		hp.trainData = (const char*) configFile.lookup("Data.trainData");
		hp.trainLabels = (const char*) configFile.lookup("Data.trainLabels");
		hp.testData = (const char*) configFile.lookup("Data.testData");
		hp.testLabels = (const char*) configFile.lookup("Data.testLabels");
		hp.numLabeled = this->m_nData;//configFile.lookup("Data.numLabeled");
		hp.numClasses = 4;//configFile.lookup("Data.numClasses");

		// TREE
		hp.maxTreeDepth = configFile.lookup("Tree.maxDepth");
		hp.bagRatio = configFile.lookup("Tree.bagRatio");
		hp.numRandomFeatures = configFile.lookup("Tree.numRandomFeatures");
		hp.numProjFeatures = configFile.lookup("Tree.numProjFeatures");
		hp.useRandProj = configFile.lookup("Tree.useRandProj");
		hp.useGPU = configFile.lookup("Tree.useGPU");
		hp.useSubSamplingWithReplacement = configFile.lookup("Tree.subSampleWR");
		hp.verbose = configFile.lookup("Tree.verbose");
		hp.useInfoGain = configFile.lookup("Tree.useInfoGain");


		// FOREST
		hp.numTrees = configFile.lookup("Forest.numTrees");
		hp.useSoftVoting = configFile.lookup("Forest.useSoftVoting");
		hp.saveForest = configFile.lookup("Forest.saveForest");

		std::cout<<"creating forest"<<std::endl;
		this->m_Forest= new Forest(hp);
		std::cout<<"training forest"<<std::endl;
		this->m_Forest->train(this->m_TrainData.getData(),this->m_TrainData.getLabels(),this->m_weights);
		std::cout<<"done"<<std::endl;
	};

	void setData(ImagePointerType intensities, ImagePointerType labels){
		int maxTrain=100000;
		m_radius=2;
		//maximal size
		long int nData=1;
		for (int d=0;d<ImageType::ImageDimension;++d)
			nData*=intensities->GetLargestPossibleRegion().GetSize()[d];
		std::cout<<nData<<std::endl;
//		nData*=pow(1.0*2*m_radius+1,ImageType::ImageDimension);
		std::cout<<nData<<" computed"<<std::endl;
		int nFeatures=9;
		matrix<float> data(maxTrain,nFeatures);
		std::cout<<nData<<" matrix allocated"<<std::endl;
		std::vector<int> labelVector(maxTrain);
		typedef typename itk::ImageRandomConstIteratorWithIndex< ImageType > IteratorType;
		IteratorType ImageIterator(intensities, intensities->GetLargestPossibleRegion());
		ImageIterator.SetNumberOfSamples(nData);
		int i=0;
		std::vector<int> counts(4,0);
		ImageIterator.GoToBegin();

		for (;!ImageIterator.IsAtEnd() ;
				++ImageIterator)
		{


			if (i>=maxTrain){
				break;
			}
			float centralIntens=1.0*ImageIterator.Get()/65535;
			float centralLabel=labels->GetPixel(ImageIterator.GetIndex())>0;

			if (centralLabel || ((counts[1]||counts[3]) && 1.0*(counts[1]+counts[3]) > 0.5* (counts[1]+counts[0]+counts[2]+counts[3]))) {

				IteratorType NeighbImageIterator(intensities, intensities->GetLargestPossibleRegion());
				NeighbImageIterator.SetNumberOfSamples(nData);

				for (NeighbImageIterator.GoToBegin();
						!NeighbImageIterator.IsAtEnd() ;
						++NeighbImageIterator)
				{
					if (i>=maxTrain){
						break;
					}
					float intens=1.0*NeighbImageIterator.Get()/65535;
					int label=labels->GetPixel(NeighbImageIterator.GetIndex())>0;
//					std::cout<<i<<" "<<labelVector[i]<<" "<<label<<" "<<intens<<" "<<centralIntens<<std::endl;

					if (label || ((counts[0]||counts[2]) && 1.0*(counts[0]+counts[2]) > 0.5* (counts[1]+counts[0]+counts[2]+counts[3]))) {
						if (label ||(counts[3]||counts[2]) && 1.0*(counts[3]+counts[2]) > 0.5* (counts[1]+counts[0]+counts[2]+counts[3])) {

						data(i,0)=centralIntens;
						data(i,1)=intens;
						data(i,2)=data(i,0)-data(i,1);
						data(i,3)=fabs(data(i,0)-data(i,1));
						data(i,4)=(data(i,3))*(data(i,3));
						data(i,6)=data(i,0)*data(i,0);
						data(i,7)=data(i,0)*data(i,1);
						data(i,8)=fabs(data(i,7));
						labelVector[i]=centralLabel + 2*label;
						counts[centralLabel + 2*label]++;
						++i;
						}
					}
				}


				//
				//					}
				//				}
			}
		}
		std::cout<<counts[0]<<" "<<counts[1]<<" "<<counts[2]<<" "<<counts[3]<<" "<<1.0*counts[1]/(counts[1]+counts[0])<<std::endl;
		data.resize(i,nFeatures);
		std::vector<int> copy=labelVector;
		labelVector.resize(i);
		this->m_nData=i;


		std::vector<double> weights(labelVector.size());

		for (i=0;i<labelVector.size();++i){
//						weights[i]=1.0;
			weights[i]=1.0/counts[labelVector[i]];
		}
		this->m_weights=weights;
		std::cout<<"done adding data. "<<std::endl;
		this->m_TrainData.setData(data);
		this->m_TrainData.setLabels(labelVector);



	};
	void eval(float * probs, int nIntensities){
		int nFeatures=9;
		matrix<float> data(nIntensities*nIntensities*2,nFeatures);
		std::vector<int> labelVector(nIntensities*nIntensities*2);
		int idx=0;
		for (int i1=0;i1<nIntensities;++i1){
			for (int i2=0;i2<nIntensities;++i2){
				data(idx,0)=1.0/nIntensities*i1;
				data(idx,1)=1.0/nIntensities*i2;
				data(idx,2)=data(idx,0)-data(idx,1);
				data(idx,3)=fabs(data(idx,0)-data(idx,1));
				data(idx,4)=(data(idx,3))*(data(idx,3));
				data(idx,6)=data(idx,0)*data(idx,0);
				data(idx,7)=data(idx,0)*data(idx,1);
				data(idx,8)=fabs(data(idx,7));
				//						data(idx,9)=s;
				labelVector[idx]=0;
			}
		}
		this->m_Forest->eval(data,labelVector,false);
		matrix<float> conf = this->m_Forest->getConfidences();
		idx=0;
		int idx2=0;
		for (int i1=0;i1<nIntensities;++i1){
			for (int i2=0;i2<nIntensities;++i2,++idx2){
				double sum=conf(idx2,0)+conf(idx2,1)+conf(idx2,2)+conf(idx2,3);
				for (int l=0;l<4;++l,++idx){
					probs[idx]=conf(idx2,l);///sum;//					std::cout<<i1<<" "<<i2<<" "<<s<<" "<<conf(idx,1)<<std::endl;
				}
			}
		}

	}
	ImagePointerType eval(matrix<float> &data, std::vector<int> &labelVector, matrix<float> * probabilities){
		this->m_Forest->eval(data,labelVector,false);
		matrix<float> conf = this->m_Forest->getConfidences();
		(*probabilities)=matrix<double>(data.size1(),2);
		for (int i=0;i<data.size1();++i){
			//			std::cout<<conf(i,0)<<" "<<conf(i,1)<<std::endl;
			(*probabilities)(i,0)=conf(i,0)/(conf(i,0)+conf(i,1));
			(*probabilities)(i,1)=conf(i,1)/(conf(i,0)+conf(i,1));
		}
		return NULL;

	}
};
#endif /* CLASSIFIER_H_ */
