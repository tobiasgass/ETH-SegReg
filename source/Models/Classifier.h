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
#include <itkImageRandomNonRepeatingConstIteratorWithIndex.h>

#include <itkImageRegionIteratorWithIndex.h>
#include <itkImageRegionIterator.h>



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
	std::vector<int> m_counts;
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
		int maxTrain=1000000;
		//maximal size
		long int nData=1;
		for (int d=0;d<ImageType::ImageDimension;++d)
			nData*=intensities->GetLargestPossibleRegion().GetSize()[d];
		maxTrain=maxTrain>nData?nData:maxTrain;
		std::cout<<maxTrain<<" computed"<<std::endl;
		int nFeatures=3;
		matrix<float> data(maxTrain,nFeatures);
		std::cout<<maxTrain<<" matrix allocated"<<std::endl;
		std::vector<int> labelVector(maxTrain);
		typedef typename itk::ImageRandomConstIteratorWithIndex< ImageType > IteratorType;
		//		typedef typename itk::ImageRegionIteratorWithIndex< ImageType > IteratorType;
		IteratorType ImageIterator(intensities, intensities->GetLargestPossibleRegion());
		ImageIterator.SetNumberOfSamples(maxTrain);
		int i=0;
		m_counts=std::vector<int> (2+256+1,0);
		ImageIterator.GoToBegin();
		for (;!ImageIterator.IsAtEnd() ;
				++ImageIterator)
		{
			int label=labels->GetPixel(ImageIterator.GetIndex())>0;
			//			if (label || (m_counts[1] && 1.0*m_counts[1]/(m_counts[1]+m_counts[0]) > 0.5 )){
			double intens=1.0*ImageIterator.Get()/256;
			data(i,0)=intens;
			data(i,1)=intens*intens;
			data(i,2)=fabs(intens);
			labelVector[i]=label;
			m_counts[labelVector[i]]++;
			m_counts[2+intens]++;
			m_counts[2+256]++;
			//				std::cout<<ImageIterator.GetIndex()<<" "<<data(i,0)<<" "<<labelVector[i]<<std::endl;
			i++;
			//			}

		}
		std::cout<<i<<" "<<m_counts[0]<<" "<<m_counts[1]<<" "<<1.0*m_counts[1]/(m_counts[1]+m_counts[0])<<std::endl;

		data.resize(i,nFeatures);
		std::vector<int> copy=labelVector;
		labelVector.resize(i);

		m_nData=i;
		std::vector<double> weights(labelVector.size());
		for (i=0;i<(int)labelVector.size();++i){
			//			if (copy[i]!=labelVector[i]) std::cout<<"bah";
			//			weights[i]=1.0/counts[labelVector[i]];
			//			weights[i]=1.0/(m_counts[0]+m_counts[1]);
			weights[i]=1.0;
			//			weights[i]=labelVector[i]?1:1.5;
		}
		m_weights=weights;
		std::cout<<"done adding data. "<<std::endl;
		m_TrainData.setData(data);
		m_TrainData.setLabels(labelVector);
	};

	void eval(int nIntensities, float * probs){
		matrix<float> data(nIntensities,3);
		std::vector<int> labelVector(nIntensities,0);
		for (int i=0;i<nIntensities;++i){
			double intens=1.0*i;
			data(i,0)=intens;
			data(i,1)=intens*intens;
			data(i,2)=fabs(intens);
			labelVector[i]=0;
		}
		m_Forest->eval(data,labelVector,false);
		matrix<float> conf = m_Forest->getConfidences();
		std::cout<<conf.size1()<<" "<<conf.size2()<<std::endl;
		std::vector<float> p_S(2);
		p_S[0]=1.0*m_counts[0]/(m_counts[0]+m_counts[1]);
		p_S[1]=1.0*m_counts[1]/(m_counts[0]+m_counts[1]);

		std::cout<<p_S[0]<<" "<<p_S[1]<<std::endl;

		double sum1=0.0, sum2=0.0;

		for (int i=0;i<nIntensities;++i){
			double p_i=1.0*m_counts[2+i]/m_counts[2+nIntensities];
			for (int s=0;s<2;++s){
				probs[s+2*i]=conf(i,s)/p_S[s]*p_i;
				//				std::cout<<1.0*i5<<" "<<s<<" "<<probs[s+2*i]<<" "<<conf(i,s)<<std::endl;
			}
			sum1+=probs[0+2*i];
			sum2+=probs[1+2*i];
		}
		//		std::cout<<sum1<<" "<<sum2<<std::endl;
	}
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
		hp.numRandomFeatures = 2;//configFile.lookup("Tree.numRandomFeatures");
		hp.numProjFeatures = 1;//configFile.lookup("Tree.numProjFeatures");
		hp.useRandProj = configFile.lookup("Tree.useRandProj");
		hp.useGPU = configFile.lookup("Tree.useGPU");
		hp.useSubSamplingWithReplacement = configFile.lookup("Tree.subSampleWR");
		hp.verbose = configFile.lookup("Tree.verbose");
		hp.useInfoGain = configFile.lookup("Tree.useInfoGain");


		// FOREST
		hp.numTrees = 50;//configFile.lookup("Forest.numTrees");
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
		hp.numClasses = configFile.lookup("Data.numClasses");

		// TREE
		hp.maxTreeDepth = configFile.lookup("Tree.maxDepth");
		hp.bagRatio = configFile.lookup("Tree.bagRatio");
		hp.numRandomFeatures = 6;//configFile.lookup("Tree.numRandomFeatures");
		hp.numProjFeatures = 4;//configFile.lookup("Tree.numProjFeatures");
		hp.useRandProj = configFile.lookup("Tree.useRandProj");
		hp.useGPU = configFile.lookup("Tree.useGPU");
		hp.useSubSamplingWithReplacement = configFile.lookup("Tree.subSampleWR");
		hp.verbose = configFile.lookup("Tree.verbose");
		hp.useInfoGain = configFile.lookup("Tree.useInfoGain");


		// FOREST
		hp.numTrees = 100;//configFile.lookup("Forest.numTrees");
		hp.useSoftVoting = configFile.lookup("Forest.useSoftVoting");
		hp.saveForest = configFile.lookup("Forest.saveForest");

		std::cout<<"creating forest"<<std::endl;
		this->m_Forest= new Forest(hp);
		std::cout<<"training forest"<<std::endl;
		this->m_Forest->train(this->m_TrainData.getData(),this->m_TrainData.getLabels(),this->m_weights);//	,this->m_weights);
		std::cout<<"done"<<std::endl;
	};
	void setData(ImagePointerType intensities, ImagePointerType labels){
		typedef typename itk::ConstNeighborhoodIterator<ImageType> NeighborhoodIteratorType;
		typename NeighborhoodIteratorType::RadiusType radius;

		int maxTrain=1000000;
		//maximal size
		long int nData=1;
		int nNeighb=1.0;
		for (int d=0;d<ImageType::ImageDimension;++d){
			nData*=intensities->GetLargestPossibleRegion().GetSize()[d];
			radius[d]=2;
			nNeighb*=radius[d]*2;
		}
		std::cout<<nData<<std::endl;
		std::cout<<nData<<" computed"<<std::endl;
		int nFeatures=10;
		matrix<float> data(maxTrain,nFeatures);
		std::cout<<maxTrain<<" matrix allocated"<<std::endl;
		std::vector<int> labelVector(maxTrain);
		typedef typename itk::ImageRandomNonRepeatingConstIteratorWithIndex< ImageType > IteratorType;
		IteratorType ImageIterator(intensities, intensities->GetLargestPossibleRegion());
		NeighborhoodIteratorType NeighbImageIterator(radius,intensities, intensities->GetLargestPossibleRegion());
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

			//			IteratorType NeighbImageIterator(intensities, intensities->GetLargestPossibleRegion());
			//			int maxNeighb=int(sqrt(double(maxTrain)));
			//			NeighbImageIterator.SetNumberOfSamples(nData>maxNeighb?maxNeighb:nData);
			NeighbImageIterator.SetLocation(ImageIterator.GetIndex());
			for (int n=0;n<NeighbImageIterator.Size();++n){
				bool inBounds=false;
				float intens=1.0*NeighbImageIterator.GetPixel(n,inBounds)/65535;
				if (inBounds){

					int label=labels->GetPixel(NeighbImageIterator.GetIndex(n))>0;
					data(i,0)=centralIntens;
					data(i,1)=intens;
					data(i,2)=data(i,0)-data(i,1);
					data(i,3)=fabs(data(i,0)-data(i,1));
					data(i,4)=(data(i,3))*(data(i,3));
					data(i,6)=data(i,0)*data(i,0);
					data(i,7)=data(i,0)*data(i,1);
					data(i,8)=fabs(data(i,7));
					data(i,9)=label;
					labelVector[i]=centralLabel;
					counts[centralLabel + 2*label]++;
					++i;
				}
				if (i>=maxTrain){
					break;
				}
			}
		}
		std::cout<<counts[0]<<" "<<counts[1]<<" "<<counts[2]<<" "<<counts[3]<<std::endl;

		data.resize(i,nFeatures);
		std::vector<int> copy=labelVector;
		labelVector.resize(i);
		this->m_nData=i;


		std::vector<double> weights(labelVector.size());

		for (i=0;i<labelVector.size();++i){
			weights[i]=1.0;
			//			if (labelVector[i]!=3){
			//				weights[i]=1.0/1.5;
			//			}
			//			else
			//				weights[i]=1.0;///(counts[labelVector[i]]+counts[labelVector[i]+2]);
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
		std::cout<<"prepared eval data"<<std::endl;
		this->m_Forest->eval(data,labelVector,false);
		std::cout<<"evaluated data"<<std::endl;
		matrix<float> conf = this->m_Forest->getConfidences();
		std::cout<<conf.size1()*conf.size2()<<" "<<2*nIntensities*nIntensities*2<<std::endl;
		idx=0;
		int idx2=0;
		for (int i1=0;i1<nIntensities;++i1){
			for (int i2=0;i2<nIntensities;++i2){
				for (int s=0;s<2;++s,++idx){
					for (int s2=0;s2<2;++s2,++idx2){
						//						std::cout<<idx<<" "<<idx2<<std::endl;
						probs[idx2]=conf(idx,s2);
					}

				}
			}
		}
		std::cout<<"finished"<<std::endl;

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
	std::vector<int> counts;

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
		hp.numTrees = 100;//configFile.lookup("Forest.numTrees");
		hp.useSoftVoting = configFile.lookup("Forest.useSoftVoting");
		hp.saveForest = configFile.lookup("Forest.saveForest");

		std::cout<<"creating forest"<<std::endl;
		this->m_Forest= new Forest(hp);
		std::cout<<"training forest"<<std::endl;
		this->m_Forest->train(this->m_TrainData.getData(),this->m_TrainData.getLabels(),this->m_weights);
		std::cout<<"done"<<std::endl;
	};

	void setData(ImagePointerType intensities, ImagePointerType labels){
		int maxTrain=2000000;
		m_radius=3;
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
		counts=std::vector<int>(4,0);
		ImageIterator.GoToBegin();

		for (;!ImageIterator.IsAtEnd() ;
				++ImageIterator)
		{


			if (i>=maxTrain){
				break;
			}
			float centralIntens=1.0*ImageIterator.Get()/65535;
			float centralLabel=labels->GetPixel(ImageIterator.GetIndex())>0;

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

		std::cout<<counts[0]<<" "<<counts[1]<<" "<<counts[2]<<" "<<counts[3]<<" "<<1.0*counts[1]/(counts[1]+counts[0])<<std::endl;
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
		int nFeatures=9;
		matrix<float> data(nIntensities*nIntensities,nFeatures);
		std::vector<int> labelVector(nIntensities*nIntensities);
		int idx=0;
		for (int i1=0;i1<nIntensities;++i1){
			for (int i2=0;i2<nIntensities;++i2,++idx){
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
				for (int s=0;s<2;++s){
					for (int s2=0;s2<2;++s2,++idx){
						int c=counts[s2+2*s];
						if (c){
							probs[s2+2*s+2*2*i2+2*2*nIntensities*i1]=conf(i2+i1*nIntensities,s2+2*s)/c;
						}
						else{
							probs[s2+2*s+2*2*i2+2*2*nIntensities*i1]=0.0;
						}
					}
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
class intensityLikelihoodClassifier{
public:
	typedef typename ImageType::Pointer ImagePointerType;
	typedef typename itk::ImageDuplicator< ImageType > DuplicatorType;
	typedef typename itk::Image<itk::Vector<float,2> ,ImageType::ImageDimension> ProbImageType;
	typedef typename ProbImageType::Pointer ProbImagePointerType;
	typedef typename itk::ImageRegionIterator< ImageType > IteratorType;

protected:
	matrix<float>m_conf;
	int m_nData;
	ImagePointerType trainImage,labelImage;
	int m_nIntensities;
public:
	intensityLikelihoodClassifier(){
		//		m_conf=matrix<float>(1,2);
		m_nIntensities=255;
	};
	void freeMem(){
	}
	void save(string filename){
	}
	void load(string filename){
	}
	void setData(ImagePointerType intensities, ImagePointerType labels){
		trainImage=intensities;
		labelImage=labels;
	};

	void eval(int nIntensities, float * probs){
		for (int i=0;i<nIntensities;++i){
			for (int s=0;s<2;++s){
				probs[s+2*i]=m_conf(i,s);
			}
		}
	}
	void train(){
		m_conf=matrix<float>(m_nIntensities,2);
		IteratorType imageIterator(trainImage, trainImage->GetLargestPossibleRegion());
		IteratorType labelIterator(labelImage, labelImage->GetLargestPossibleRegion());
		std::vector<int> counts(2);
		std::vector<double> mean(2);
		std::vector<double> variance(2);
		double globalMean=0.0,globalVar=0.0;
		int globalCount=0.0;
		for (imageIterator.GoToBegin(),labelIterator.GoToBegin();!imageIterator.IsAtEnd();++imageIterator,++labelIterator){
			int label=labelIterator.Get()>0;
			int intensity=int(imageIterator.Get()/m_nIntensities);
			counts[label]++;
			mean[label]+=intensity;
			variance[label]+=intensity*intensity;
			globalMean+=intensity;
			globalVar+=intensity;
			globalCount++;
		}
		globalMean/=globalCount;
		globalVar/=globalCount;
		globalVar-=globalMean*globalMean;
		std::cout<<globalMean<<" "<<globalCount<<std::endl;
		std::cout<<counts[0]<<" "<<counts[1]<<std::endl;
		for (int i=0;i<2;++i){
			mean[i]/=counts[i];
			variance[i]/=counts[i];
			variance[i]-=mean[i]*mean[i];
			std::cout<<i<<" "<<mean[i]<<" "<<variance[i]<<std::endl;
		}
		m_conf=matrix<float>(m_nIntensities,2);
		for (int i=0;i<m_nIntensities;++i){
			for (int s=0;s<2;++s){
				//				m_conf(i,s)=1.0/sqrt(2*3.14*sqrt(variance[s]))*exp(-0.5*(i-mean[s])*(i-mean[s])/(variance[s]));
				m_conf(i,s)=exp(-0.5*(i-mean[s])*(i-mean[s])/(variance[s]));
			}
		}
	};
	ImagePointerType eval(ImagePointerType intensities, ImagePointerType labels, ProbImagePointerType & probabilities){

		return NULL;

	}

	double posterior(float intensity, int label){
		return 1.0;
	}


};
template<class ImageType>
class pairwiseIntensityLikelihood{
public:
	typedef typename ImageType::Pointer ImagePointerType;
	typedef typename itk::ImageDuplicator< ImageType > DuplicatorType;
	typedef typename itk::Image<itk::Vector<float,2> ,ImageType::ImageDimension> ProbImageType;
	typedef typename ProbImageType::Pointer ProbImagePointerType;
	typedef typename itk::ImageRegionIterator< ImageType > IteratorType;

protected:
	matrix<float>m_conf;
	int m_nData;
	ImagePointerType m_trainImage,m_labelImage;
	int m_nIntensities;
	float* singleCounts;
	float* intensCounts;
public:
	pairwiseIntensityLikelihood(){
		//		m_conf=matrix<float>(1,2);
		m_nIntensities=255;
	};
	void freeMem(){
		delete singleCounts;
		delete intensCounts;
	}
	void save(string filename){
	}
	void load(string filename){
	}
	void setData(ImagePointerType intensities, ImagePointerType labels){
		m_trainImage=intensities;
		m_labelImage=labels;
	};

	void eval(int nIntensities, float * probs){
		int test[2*256*256]={0};

		int idx2=0;
		for (int i2=0;i2<nIntensities;++i2){
			for (int s=0;s<2;++s){
				for (int i1=0;i1<nIntensities;++i1,++idx2){

					int idx=i1+s*256+i2*2*256;
					//					std::cout<<idx<<" "<<idx2<<std::endl;
					if (intensCounts[i1]){
						probs[idx]=singleCounts[idx]/intensCounts[i1];
					}
					else{
						probs[idx]=0.0;
					}
					//					std::cout<<i1<<" "<<i2<<" "<<s<<" "<<probs[i1+s*256+i2*2*256]<<" "<<i1+s*256+i2*2*256<<" "<<intensCounts[i1]<<std::endl;
					if (test[idx]){
						std::cout<<"arlarm "<<idx<<std::endl;
					}
					test[idx]++;
				}
			}

		}
	}
	void train(){
		typedef typename itk::ConstNeighborhoodIterator<ImageType> NeighborhoodIteratorType;
		typename NeighborhoodIteratorType::RadiusType radius;

		int maxTrain=1000000;
		//maximal size
		long int nData=1;
		int nNeighb=1.0;
		for (int d=0;d<ImageType::ImageDimension;++d){
			nData*=m_trainImage->GetLargestPossibleRegion().GetSize()[d];
			radius[d]=2;
			nNeighb*=radius[d]*2;
		}
		std::cout<<nData<<std::endl;
		std::cout<<nData<<" computed"<<std::endl;
		int nFeatures=10;
		std::cout<<maxTrain<<" matrix allocated"<<std::endl;
		typedef typename itk::ImageRandomNonRepeatingConstIteratorWithIndex< ImageType > IteratorType;
		IteratorType ImageIterator(m_trainImage, m_trainImage->GetLargestPossibleRegion());
		NeighborhoodIteratorType NeighbImageIterator(radius,m_trainImage, m_trainImage->GetLargestPossibleRegion());
		ImageIterator.SetNumberOfSamples(nData);
		int i=0;
		std::vector<int> counts(4,0);
		ImageIterator.GoToBegin();
		singleCounts= new float[2*256*256];
		for (int i=0;i<2*256*256;++i )singleCounts[i]=0.0;
		intensCounts= new float[256];
		for (int i=0;i<256;++i )intensCounts[i]=0.0;
		for (;!ImageIterator.IsAtEnd() ;
				++ImageIterator)
		{
			int centralIntens=1.0*ImageIterator.Get()/256;
			float centralLabel=m_labelImage->GetPixel(ImageIterator.GetIndex())>0;

			//			IteratorType NeighbImageIterator(m_trainImage, m_trainImage->GetLargestPossibleRegion());
			//			int maxNeighb=int(sqrt(double(maxTrain)));
			//			NeighbImageIterator.SetNumberOfSamples(nData>maxNeighb?maxNeighb:nData);
			NeighbImageIterator.SetLocation(ImageIterator.GetIndex());
			for (int n=0;n<NeighbImageIterator.Size();++n){
				bool inBounds=false;
				int intens=1.0*NeighbImageIterator.GetPixel(n,inBounds)/256;
				if (inBounds){
					int idx=intens+centralLabel*256+centralIntens*2*256;
					singleCounts[idx]++;
					intensCounts[intens]++;
					//					std::cout<<idx<<" "<<singleCounts[idx]<<" "<<intensCounts[intens]<<" "<<intens<<std::endl;
					++i;

				}

			}
		}

		this->m_nData=i;


	}
	ImagePointerType eval(ImagePointerType intensities, ImagePointerType labels, ProbImagePointerType & probabilities){

		return NULL;

	}

	double posterior(float intensity, int label){
		return 1.0;
	}


};
template<class ImageType>
class relativeFrequencyPairwiseIntensityLikelihood{
public:
	typedef typename ImageType::Pointer ImagePointerType;
	typedef typename itk::ImageDuplicator< ImageType > DuplicatorType;
	typedef typename itk::Image<itk::Vector<float,2> ,ImageType::ImageDimension> ProbImageType;
	typedef typename ProbImageType::Pointer ProbImagePointerType;
	typedef typename itk::ImageRegionIterator< ImageType > IteratorType;

protected:
	matrix<float>m_conf;
	int m_nData;
	ImagePointerType m_trainImage,m_labelImage;
	int m_nIntensities;
	float* singleCounts;
	float* intensCounts;
public:
	relativeFrequencyPairwiseIntensityLikelihood(){
		//		m_conf=matrix<float>(1,2);
		m_nIntensities=255;
	};
	void freeMem(){
		delete singleCounts;
		delete intensCounts;
	}
	void save(string filename){
	}
	void load(string filename){
	}
	void setData(ImagePointerType intensities, ImagePointerType labels){
		m_trainImage=intensities;
		m_labelImage=labels;
	};

	void eval(int nIntensities, float * probs){
		int test[2*256*256]={0};

		int idx2=0;
		int idx=0;
		for (int i1=0;i1<nIntensities;++i1){
			for (int i2=0;i2<nIntensities;++i2,++idx){
				for (int s=0;s<2;++s){
					for (int s2=0;s2<2;++s2,++idx2){
						int idx3=s2+2*s+i2*2*2+i1*2*2*nIntensities;
						if (idx2!=idx3)
							std::cout<<idx2<<" "<<idx3<<std::endl;
						if (intensCounts[idx]){
							probs[idx2]=singleCounts[idx2]/intensCounts[idx];
						}
						else{
							probs[idx2]=0.0;
						}

					}
				}
			}
		}

		//		for (int i2=0;i2<nIntensities;++i2){
		//			for (int s=0;s<2;++s){
		//				for (int i1=0;i1<nIntensities;++i1,++idx2){
		//					for (int s2=0;s2<2;++s2){
		//
		//						int idx=i1+s2*256+2*s*256+i2*2*2*256;
		//						//					std::cout<<idx<<" "<<idx2<<std::endl;
		//						if (intensCounts[i1+256*i2]){
		//							probs[idx]=singleCounts[idx]/intensCounts[i1+256*i2];
		//						}
		//						else{
		//							probs[idx]=0.0;
		//						}
		//						//					std::cout<<i1<<" "<<i2<<" "<<s<<" "<<probs[i1+s*256+i2*2*256]<<" "<<i1+s*256+i2*2*256<<" "<<intensCounts[i1]<<std::endl;
		//						if (test[idx]){
		//							std::cout<<"arlarm "<<idx<<std::endl;
		//						}
		//						test[idx]++;
		//					}
		//				}
		//			}
		//
		//		}
	}
	void train(){
		typedef typename itk::ConstNeighborhoodIterator<ImageType> NeighborhoodIteratorType;
		typename NeighborhoodIteratorType::RadiusType radius;

		int maxTrain=1000000;
		//maximal size
		long int nData=1;
		int nNeighb=1.0;
		int rad=2;
		for (int d=0;d<ImageType::ImageDimension;++d){
			nData*=m_trainImage->GetLargestPossibleRegion().GetSize()[d];
			radius[d]=rad;
			nNeighb*=radius[d]*rad;
		}
		std::cout<<nData<<std::endl;
		std::cout<<nData<<" computed"<<std::endl;
		int nFeatures=10;
		std::cout<<maxTrain<<" matrix allocated"<<std::endl;
		typedef typename itk::ImageRandomNonRepeatingConstIteratorWithIndex< ImageType > IteratorType;
		IteratorType ImageIterator(m_trainImage, m_trainImage->GetLargestPossibleRegion());
		NeighborhoodIteratorType NeighbImageIterator(radius,m_trainImage, m_trainImage->GetLargestPossibleRegion());
		ImageIterator.SetNumberOfSamples(nData);
		int i=0;
		std::vector<int> counts(4,0);
		ImageIterator.GoToBegin();
		singleCounts= new float[2*2*256*256];
		for (int i=0;i<2*2*256*256;++i )singleCounts[i]=0.0;
		intensCounts= new float[256*256];
		for (int i=0;i<256*256;++i )intensCounts[i]=0.0;
		for (;!ImageIterator.IsAtEnd() ;
				++ImageIterator)
		{
			int centralIntens=1.0*ImageIterator.Get()/256;
			float centralLabel=m_labelImage->GetPixel(ImageIterator.GetIndex())>0;

			//			IteratorType NeighbImageIterator(m_trainImage, m_trainImage->GetLargestPossibleRegion());
			//			int maxNeighb=int(sqrt(double(maxTrain)));
			//			NeighbImageIterator.SetNumberOfSamples(nData>maxNeighb?maxNeighb:nData);
			NeighbImageIterator.SetLocation(ImageIterator.GetIndex());
			for (int n=0;n<NeighbImageIterator.Size();++n){
				bool inBounds=false;
				int intens=1.0*NeighbImageIterator.GetPixel(n,inBounds)/256;
				if (inBounds){
					int label=m_labelImage->GetPixel(NeighbImageIterator.GetIndex(n))>0;

					int idx=centralLabel+2*label+4*centralIntens+4*256*intens;
					singleCounts[idx]++;
					intensCounts[centralIntens+256*intens]++;
					//					std::cout<<idx<<" "<<singleCounts[idx]<<" "<<intensCounts[intens]<<" "<<intens<<std::endl;
					++i;

				}

			}
		}
		std::cout<<i<<" training observation pairs."<<std::endl;
		this->m_nData=i;


	}
	ImagePointerType eval(ImagePointerType intensities, ImagePointerType labels, ProbImagePointerType & probabilities){

		return NULL;

	}

	double posterior(float intensity, int label){
		return 1.0;
	}


};

#endif /* CLASSIFIER_H_ */
