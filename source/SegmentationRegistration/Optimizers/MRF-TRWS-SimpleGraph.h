#include "Log.h"
/*
 * TRW-S-Registration.h
 *
 *  Created on: Dec 3, 2010
 *      Author: gasst
 */

#ifndef TRW_S_H_
#define TRW_S_H_
#include "typeTruncatedQuadratic2D.h"
#include "typeGeneral.h"
#include "MRFEnergy.h"
#include "minimize.cpp"
#include "treeProbabilities.cpp"
template<class TGraphModel>
class TRWS_MRFSolver {
public:
	typedef TGraphModel GraphModelType;

	typedef TypeGeneral TRWType;
	//	typedef TypeTruncatedQuadratic2D TRWType;
	typedef MRFEnergy<TRWType> MRFType;
	typedef typename MRFType::NodeId NodeType;
	//	typedef typename Superclass::DeformationFieldType DeformationFieldType;

protected:
	MRFType* optimizer;
	NodeType* nodes;
	double m_unaryWeight,m_pairwiseWeight;
	bool verbose;
    GraphModelType * m_GraphModel;
    int nNodes;
public:
	TRWS_MRFSolver(GraphModelType * graphModel, double unaryWeight=1.0, double pairwiseWeight=1.0, bool verb=false)
    {
		m_GraphModel=graphModel;
        verbose=verb;
		m_unaryWeight=unaryWeight;
		m_pairwiseWeight=pairwiseWeight;
		createGraph();
	}
	~TRWS_MRFSolver()
		{
			delete [] nodes;
			delete optimizer;

		}
	virtual void createGraph(){
		//		TRWType::GlobalSize globalSize(labelSampling,labelSampling);
		//		optimizer = new MRFType(globalSize);
		TRWType::GlobalSize globalSize();
		optimizer = new MRFType(TRWType::GlobalSize());
		LOGV(1)<<"starting graph init"<<std::endl;
		GraphModelType* graph=this->m_GraphModel;
        nNodes=graph->nNodes();
		nodes = new NodeType[nNodes];

        int nLabels=graph->nLabels();


		clock_t start = clock();
		//		traverse grid
		TRWType::REAL D[nLabels];

		for (int d=0;d<nNodes;++d){
			//set up unary costs at current position
			for (int l1=0;l1<nLabels;++l1)
			{
//				LOG<<d<<" "<<l1<<" "<<nNodes<<" "<<nLabels<<std::endl;
				D[l1]=m_unaryWeight*graph->getUnaryPotential(d,l1);
			}
			nodes[d] = optimizer->AddNode(TRWType::LocalSize(nLabels), TRWType::NodeData(D));
			//			nodes[currentIntIndex] = optimizer->AddNode(TRWType::LocalSize(), TRWType::NodeData(D));
		}
		clock_t finish1 = clock();
		float t = (float) ((double)(finish1 - start) / CLOCKS_PER_SEC);
		LOGV(1)<<"Finished unary potential initialisation after "<<t<<" seconds"<<std::endl;
		//

        //#define POTTS
        //#define BACKWARD
		
		for (int d=0;d<nNodes;++d){
			std::vector<int> neighbours= graph->getForwardNeighbours(d);
			int nNeighbours=neighbours.size();
			for (int i=0;i<nNeighbours;++i){
				TRWType::REAL V[nLabels*nLabels];
				TRWType::REAL V2[nLabels*nLabels];
                double lambda=1;//graph->getWeight(d,neighbours[i]);
                double lambda2=1;//graph->getWeight(neighbours[i],d);
				for (int l1=0;l1<nLabels;++l1){
					for (int l2=0;l2<nLabels;++l2){
						V[l1+nLabels*l2]=lambda*m_pairwiseWeight*graph->getPairwisePotential(d,neighbours[i],l1,l2);
                        //						V[l1+nLabels*l2+nLabels*nLabels]=lambda2*m_pairwiseWeight*graph->getPairwisePotential(l1,l2);
                        ///V2[l1*nLabels+l2]=lambda2*m_pairwiseWeight*graph->getPairwisePotential(neighbours[i],d,l1,l2);
                        //LOG<<"PAIRWISE " <<d<<" "<<i<<" "<<l1<<" "<<l2<<" "<<                            V[l1+nLabels*l2]<<endl;

                    }
				}
#ifdef POTTS
                optimizer->AddEdge(nodes[neighbours[i]], nodes[d], TRWType::EdgeData(TRWType::POTTS,lambda2));
#ifdef BACKWARD
                optimizer->AddEdge(nodes[neighbours[i]], nodes[d], TRWType::EdgeData(TRWType::POTTS,lambda2));
#endif
#else
                optimizer->AddEdge(nodes[d], nodes[neighbours[i]], TRWType::EdgeData(TRWType::GENERAL,V));
#ifdef BACKWARD
                optimizer->AddEdge(nodes[neighbours[i]], nodes[d], TRWType::EdgeData(TRWType::GENERAL,V2));
#endif
#endif
			}
		}
		clock_t finish = clock();
		t = (float) ((double)(finish - start) / CLOCKS_PER_SEC);
		LOGV(1)<<"Finished init after "<<t<<" seconds"<<std::endl;

	}

	virtual void optimize(int maxiter){
		MRFEnergy<TRWType>::Options options;
		TRWType::REAL energy, lowerBound;
		options.m_iterMax = maxiter; // maximum number of iterations
		options.m_printMinIter=11;
		options.m_printIter=11;
        options.verbose=verbose;
		clock_t start = clock();
		optimizer->Minimize_TRW_S(options, lowerBound, energy);
		clock_t finish = clock();
		float t = (float) ((double)(finish - start) / CLOCKS_PER_SEC);
		LOG<<"Finished after "<<t<<" , resulting energy is "<<energy<<" with lower bound "<< lowerBound ;//<< std::endl;

	}
    virtual std::vector<int> getLabels(){
        std::vector<int> labels(nNodes);
        for (int i=0;i<nNodes;++i){
            labels[i]=optimizer->GetSolution(nodes[i]);
        }
        return labels;
    }

};
#if 1
template<class TGraphModel>
class TRWS_SimpleMRFSolver {
public:
	
	typedef TGraphModel GraphModelType;
    typedef TypeBinaryFast TRWType;
    typedef MRFEnergy<TRWType> MRFType;
	typedef typename MRFType::NodeId NodeType;
	
protected:
	MRFType* optimizer;
	NodeType* nodes;
	double m_unaryWeight,m_pairwiseWeight;
	bool verbose;
    int nNodes;
    GraphModelType * m_graphModel;
public:
	TRWS_SimpleMRFSolver(GraphModelType * graphModel, double unaryWeight=1.0, double pairwiseWeight=1.0, bool verb=false)

	{
        m_graphModel=graphModel;
		verbose=verb;
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
		optimizer = new MRFType(TRWType::GlobalSize());
		LOGV(1)<<"starting graph init"<<std::endl;
		GraphModelType* graph=this->m_graphModel;
		nNodes=graph->nNodes();

		nodes = new NodeType[nNodes];

		int nLabels=graph->nLabels();
//		int runningIndex=0;

		clock_t start = clock();
		//		traverse grid
		TRWType::REAL D[nLabels];

		for (int d=0;d<nNodes;++d){
			//set up unary costs at current position
			for (int l1=0;l1<nLabels;++l1)
			{
				D[l1]=m_unaryWeight*graph->getUnaryPotential(d,l1);
			}

			nodes[d] = optimizer->AddNode(TRWType::LocalSize(), TRWType::NodeData(D[0],D[1]));

		}
		clock_t finish1 = clock();
		float t = (float) ((double)(finish1 - start) / CLOCKS_PER_SEC);
		LOG<<"Finished unary potential initialisation after "<<t<<" seconds"<<std::endl;
		//


//		double weight=m_pairwiseWeight;
		for (int d=0;d<nNodes;++d){
			std::vector<int> neighbours= graph->getForwardNeighbours(d);
			int nNeighbours=neighbours.size();
			for (int i=0;i<nNeighbours;++i){
                double l1=m_pairwiseWeight*graph->getPairwisePotential(d,i,1,0);
                //double l2=m_pairwiseWeight*graph->getWeight(i,d);
                TRWType::EdgeData edge(0,l1,l1,0);
                //                TRWType::EdgeData edge2(0,l2,l2,0);

				optimizer->AddEdge(nodes[d], nodes[neighbours[i]], edge);
                //optimizer->AddEdge( nodes[neighbours[i]],nodes[d], edge);


				//				optimizer->AddEdge(nodes[currentIntIndex], nodes[neighbours[i]], TRWType::EdgeData(weight, weight, 8*weight));
			}
		}
		clock_t finish = clock();
		t = (float) ((double)(finish - start) / CLOCKS_PER_SEC);
//		LOG<<"Finished init after "<<t<<" seconds"<<std::endl;

	}

	virtual void optimize(int optiter){
		MRFEnergy<TRWType>::Options options;
		TRWType::REAL energy, lowerBound;
		options.m_iterMax = optiter; // maximum number of iterations
		options.m_printMinIter=1;
		options.m_printIter=1;
        options.verbose=verbose;
		clock_t start = clock();
        optimizer->Minimize_TRW_S(options, lowerBound, energy);
		clock_t finish = clock();
		float t = (float) ((double)(finish - start) / CLOCKS_PER_SEC);
		LOG<<"Finished after "<<t<<" , resulting energy is "<<energy<<" with lower bound "<< lowerBound;// << std::endl;

	}
 virtual std::vector<int> getLabels(){
        std::vector<int> labels(nNodes);
        for (int i=0;i<nNodes;++i){
            labels[i]=optimizer->GetSolution(nodes[i]);
        }
        return labels;
    }

};
#endif
#endif /* TRW_S_REGISTRATION_H_ */
