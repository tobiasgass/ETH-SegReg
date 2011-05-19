/*
 * TRW-S-Registration.h
 *
 *  Created on: Dec 3, 2010
 *      Author: gasst
 */

#ifndef TRW_S_REGISTRATION_H_
#define TRW_S_REGISTRATION_H_
#include "BaseMRF.h"
#include "typeTruncatedQuadratic2D.h"
#include "typeGeneral.h"
#include "MRFEnergy.h"
#include "minimize.cpp"
#include "treeProbabilities.cpp"
template<class TGraphModel>
class TRWS_MRFSolver : public BaseMRFSolver<TGraphModel>{
public:
	typedef BaseMRFSolver<TGraphModel> Superclass;
	typedef typename Superclass::LabelType LabelType;
	typedef Graph::Real Real;
	typedef TGraphModel GraphModelType;
	typedef typename GraphModelType::LabelMapperType LabelMapperType;

	//	typedef Graph::Real Real;
	typedef TypeGeneral TRWType;
	//	typedef TypeTruncatedQuadratic2D TRWType;
	typedef MRFEnergy<TRWType> MRFType;
	typedef typename MRFType::NodeId NodeType;
	//	typedef typename Superclass::DeformationFieldType DeformationFieldType;

protected:
	MRFType* optimizer;
	NodeType* nodes;
	double m_unaryWeight,m_pairwiseWeight;
	bool secondPairwise;
	bool verbose;
public:
	TRWS_MRFSolver(GraphModelType * graphModel, double unaryWeight=1.0, double pairwiseWeight=1.0, bool secondPairwisePotential=false)
	:Superclass(graphModel),secondPairwise(secondPairwisePotential)
	{
		verbose=false;
		m_unaryWeight=unaryWeight;
		m_pairwiseWeight=pairwiseWeight;
		createGraph();
	}
	~TRWS_MRFSolver()
		{
			delete[] nodes;
			delete optimizer;

		}
	virtual void createGraph(){
		//		TRWType::GlobalSize globalSize(labelSampling,labelSampling);
		//		optimizer = new MRFType(globalSize);
		TRWType::GlobalSize globalSize();
		optimizer = new MRFType(TRWType::GlobalSize());
		int nNodes=this->m_nNodes;

		nodes = new NodeType[nNodes];
		if (verbose) std::cout<<"starting graph init"<<std::endl;
		GraphModelType* graph=this->m_GraphModel;
		int nLabels=this->m_nLabels;


		clock_t start = clock();
		//		traverse grid
		TRWType::REAL D[nLabels];

		for (int d=0;d<nNodes;++d){
			//set up unary costs at current position
			for (int l1=0;l1<nLabels;++l1)
			{
//				std::cout<<d<<" "<<l1<<" "<<nNodes<<" "<<nLabels<<std::endl;
				D[l1]=m_unaryWeight*graph->getUnaryPotential(d,l1);
			}
			nodes[d] = optimizer->AddNode(TRWType::LocalSize(nLabels), TRWType::NodeData(D));
			//			nodes[currentIntIndex] = optimizer->AddNode(TRWType::LocalSize(), TRWType::NodeData(D));
		}
		clock_t finish1 = clock();
		float t = (float) ((double)(finish1 - start) / CLOCKS_PER_SEC);
		if (verbose) std::cout<<"Finished unary potential initialisation after "<<t<<" seconds"<<std::endl;
		//


		double weight=m_pairwiseWeight;
		for (int d=0;d<nNodes;++d){
			std::vector<int> neighbours= graph->getForwardNeighbours(d);
			int nNeighbours=neighbours.size();
			for (int i=0;i<nNeighbours;++i){
				TRWType::REAL V[nLabels*nLabels];
				for (int l1=0;l1<nLabels;++l1){
					for (int l2=0;l2<nLabels;++l2){
						V[l1*nLabels+l2]=m_pairwiseWeight*graph->getPairwisePotential(d,neighbours[i],l1,l2);
					}
				}
				optimizer->AddEdge(nodes[d], nodes[neighbours[i]], TRWType::EdgeData(TRWType::GENERAL,V));
				//				optimizer->AddEdge(nodes[currentIntIndex], nodes[neighbours[i]], TRWType::EdgeData(weight, weight, 8*weight));
			}
		}
		clock_t finish = clock();
		t = (float) ((double)(finish - start) / CLOCKS_PER_SEC);
		if (verbose) std::cout<<"Finished init after "<<t<<" seconds"<<std::endl;

	}

	virtual void optimize(){
		MRFEnergy<TRWType>::Options options;
		TRWType::REAL energy, lowerBound;
		options.m_iterMax = 20; // maximum number of iterations
		options.m_printMinIter=1100;
		options.m_printIter=1100;
		clock_t start = clock();
		optimizer->Minimize_TRW_S(options, lowerBound, energy);
		clock_t finish = clock();
		float t = (float) ((double)(finish - start) / CLOCKS_PER_SEC);
		std::cout<<"Finished after "<<t<<" , resulting energy is "<<energy<<" with lower bound "<< lowerBound ;//<< std::endl;

	}

	virtual LabelType getLabelAtIndex(int index){
		TRWType::Label l=optimizer->GetSolution(nodes[index]);
		//		int labelIndex=l.m_kx+l.m_ky*labelSampling;
		int labelIndex=l;
		return LabelMapperType::getLabel(labelIndex);

	}
};
template<class TGraphModel>
class TRWS_SimpleMRFSolver : public BaseMRFSolver<TGraphModel>{
public:
	typedef BaseMRFSolver<TGraphModel> Superclass;
	typedef typename Superclass::LabelType LabelType;
	typedef Graph::Real Real;
	typedef TGraphModel GraphModelType;
	typedef typename GraphModelType::LabelMapperType LabelMapperType;

	//	typedef Graph::Real Real;
	typedef TypePotts TRWType;
	//	typedef TypeTruncatedQuadratic2D TRWType;
	typedef MRFEnergy<TRWType> MRFType;
	typedef typename MRFType::NodeId NodeType;
	//	typedef typename Superclass::DeformationFieldType DeformationFieldType;

protected:
	MRFType* optimizer;
	NodeType* nodes;
	double m_unaryWeight,m_pairwiseWeight;
	bool secondPairwise;
	bool verbose;
public:
	TRWS_SimpleMRFSolver(GraphModelType * graphModel, double unaryWeight=1.0, double pairwiseWeight=1.0, bool secondPairwisePotential=false)
	:Superclass(graphModel),secondPairwise(secondPairwisePotential)
	{
		verbose=true;
		m_unaryWeight=unaryWeight;
		m_pairwiseWeight=pairwiseWeight;
		createGraph();
	}
	~TRWS_SimpleMRFSolver()
		{
			delete[] nodes;
			delete optimizer;

		}
	virtual void createGraph(){
		//		TRWType::GlobalSize globalSize(labelSampling,labelSampling);
		//		optimizer = new MRFType(globalSize);
		TRWType::GlobalSize globalSize(this->m_nLabels);
		optimizer = new MRFType(TRWType::GlobalSize(this->m_nLabels));
		int nNodes=this->m_nNodes;

		nodes = new NodeType[nNodes];
		if (verbose) std::cout<<"starting graph init"<<std::endl;
		GraphModelType* graph=this->m_GraphModel;
		int nLabels=this->m_nLabels;
//		int runningIndex=0;

		clock_t start = clock();
		//		traverse grid
		TRWType::REAL D[nLabels];

		for (int d=0;d<nNodes;++d){
			//set up unary costs at current position
			for (int l1=0;l1<nLabels;++l1)
			{
//				std::cout<<d<<" "<<l1<<" "<<nNodes<<" "<<nLabels<<std::endl;
				D[l1]=m_unaryWeight*graph->getUnaryPotential(d,l1);
			}
			nodes[d] = optimizer->AddNode(TRWType::LocalSize(), TRWType::NodeData(D));
			//			nodes[currentIntIndex] = optimizer->AddNode(TRWType::LocalSize(), TRWType::NodeData(D));
		}
		clock_t finish1 = clock();
		float t = (float) ((double)(finish1 - start) / CLOCKS_PER_SEC);
		std::cout<<"Finished unary potential initialisation after "<<t<<" seconds"<<std::endl;
		//


//		double weight=m_pairwiseWeight;
		for (int d=0;d<nNodes;++d){
			std::vector<int> neighbours= graph->getForwardNeighbours(d);
			int nNeighbours=neighbours.size();
			for (int i=0;i<nNeighbours;++i){
				optimizer->AddEdge(nodes[d], nodes[neighbours[i]], TRWType::EdgeData(m_pairwiseWeight*graph->getWeight(d,i)));
				//				optimizer->AddEdge(nodes[currentIntIndex], nodes[neighbours[i]], TRWType::EdgeData(weight, weight, 8*weight));
			}
		}
		clock_t finish = clock();
		t = (float) ((double)(finish - start) / CLOCKS_PER_SEC);
//		std::cout<<"Finished init after "<<t<<" seconds"<<std::endl;

	}

	virtual void optimize(){
		MRFEnergy<TRWType>::Options options;
		TRWType::REAL energy, lowerBound;
		options.m_iterMax = 10; // maximum number of iterations
		options.m_printMinIter=11;
		options.m_printIter=11;
		clock_t start = clock();
		optimizer->Minimize_TRW_S(options, lowerBound, energy);
		clock_t finish = clock();
		float t = (float) ((double)(finish - start) / CLOCKS_PER_SEC);
		std::cout<<"Finished after "<<t<<" , resulting energy is "<<energy<<" with lower bound "<< lowerBound;// << std::endl;

	}

	virtual LabelType getLabelAtIndex(int index){
		TRWType::Label l=optimizer->GetSolution(nodes[index]);
		//		int labelIndex=l.m_kx+l.m_ky*labelSampling;
		int labelIndex=l;
		return LabelMapperType::getLabel(labelIndex);

	}
};
#endif /* TRW_S_REGISTRATION_H_ */
