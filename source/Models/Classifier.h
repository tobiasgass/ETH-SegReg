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
	matrix<float> m_conf;
	int m_nData;
	std::vector<int> m_counts;
public:
	typedef typename ImageType::Pointer ImagePointerType;
	typedef typename itk::ImageDuplicator< ImageType > DuplicatorType;

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
        
        //1. make histogram over data

        //2. create forest out of histogrammized data

        //3. save histogramm mapping

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

#endif /* CLASSIFIER_H_ */
