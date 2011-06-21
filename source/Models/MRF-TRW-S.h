/*
 * TRW-S-Registration.h
 *
 *  Created on: Dec 3, 2010
 *      Author: gasst
 */

#ifndef TRW_S_SRS_H_
#define TRW_S_SRS_H_
#include "BaseMRF.h"
#include "typeTruncatedQuadratic2D.h"
#include "typeGeneral.h"
#include "MRFEnergy.h"
#include "minimize.cpp"
#include "treeProbabilities.cpp"

template<class TGraphModel>
class TRWS_SRSMRFSolver {
public:


	typedef Graph::Real Real;
	typedef TGraphModel GraphModelType;
	
	//	typedef Graph::Real Real;
	typedef TypeGeneral TRWType;
	typedef MRFEnergy<TRWType> MRFType;
	typedef typename MRFType::NodeId NodeType;
    int m_nNodes,m_nLabels;

protected:
	MRFType* optimizer;
	NodeType* nodes;
	double m_unaryWeight,m_pairwiseWeight;
	bool verbose;
    GraphModelType * m_GraphModel;

public:
	TRWS_SRSMRFSolver(GraphModelType * graphModel, double unaryWeight=1.0, double pairwiseWeight=1.0, bool vverbose=false)
      :m_GraphModel(graphModel)
	{
		verbose=vverbose;
		m_unaryWeight=unaryWeight;
		m_pairwiseWeight=pairwiseWeight;
        m_nNodes=m_GraphModel->nNodes();
		m_nLabels=LabelMapperType::nLabels;
		createGraph();
	}
	~TRWS_SRSMRFSolver()
    {
        delete nodes[];
        delete optimizer;

    }
	virtual void createGraph(){
		//		TRWType::GlobalSize globalSize(labelSampling,labelSampling);
		//		optimizer = new MRFType(globalSize);
	
        TRWType::GlobalSize globalSize();
		optimizer = new MRFType(TRWType::GlobalSize());
		int nNodes=this->m_nNodes;
        int nRegNodes=graph->nRegNodes();
        int nSegNodes=graph->nSegNodes();
		nodes = new NodeType[nNodes];
		if (verbose) std::cout<<"starting graph init"<<std::endl;
		GraphModelType* graph=this->m_GraphModel;
		int nLabels=this->m_nLabels;
        int nRegLabels=graph->nRegLabels();
        int nSegLabels=graph->nSegLabels();

		clock_t start = clock();
		//		traverse grid
        
        //RegUnaries
		TRWType::REAL D1[nRegLabels];
		for (int d=0;d<nRegNodes;++d){
            for (int l1=0;l1<nRegLabels;++l1)
                {
                    D1[l1]=m_unaryWeight*graph->getUnaryRegistrationPotential(d,l1);
                }
			nodes[d] = optimizer->AddNode(TRWType::LocalSize(nRegLabels), TRWType::NodeData(D1));
		}
        //SegUnaries
		TRWType::REAL D2[nSegLabels];
		for (int d=0;d<nSegNodes;++d){
            for (int l1=0;l1<nSegLabels;++l1)
                {
                    D2[l1]=m_unaryWeight*graph->getUnarySegmentationPotential(d,l1);
                }
			nodes[nRegNodes+d] = optimizer->AddNode(TRWType::LocalSize(nSegLabels), TRWType::NodeData(D2));
		}

		clock_t finish1 = clock();
		float t = (float) ((double)(finish1 - start) / CLOCKS_PER_SEC);
		if (verbose) std::cout<<"Finished unary potential initialisation after "<<t<<" seconds"<<std::endl;

		// Pairwise potentials
        for (int d=0;d<nRegNodes;++d){
            
			{//pure Registration
                std::vector<int> neighbours= graph->getForwardRegistrationNeighbours(d);
                int nNeighbours=neighbours.size();
                for (int i=0;i<nNeighbours;++i){
                    TRWType::REAL V[nRegLabels*nRegLabels];
                    for (int l1=0;l1<nRegLabels;++l1){
                        for (int l2=0;l2<nRegLabels;++l2){
                            V[l1*nRegLabels+l2]=m_pairwiseWeight*graph->getPairwiseRegistrationPotential(d,neighbours[i],l1,l2);
                        }
                    }
                    optimizer->AddEdge(nodes[d], nodes[neighbours[i]], TRWType::EdgeData(TRWType::GENERAL,V));
                    //				optimizer->AddEdge(nodes[currentIntIndex], nodes[neighbours[i]], TRWType::EdgeData(weight, weight, 8*weight));
                }

            }
            {//SegReg
                std::vector<int> neighbours= graph->getForwardSegRegNeighbours(d);
                int nNeighbours=neighbours.size();
                for (int i=0;i<nNeighbours;++i){
                    TRWType::REAL V[nRegLabels*nSegLabels];
                    for (int l1=0;l1<nRegLabels;++l1){
                        for (int l2=0;l2<nSegLabels;++l2){
                            V[l1+l2*nRegLabels]=m_pairwiseWeight*graph->getPairwiseSegRegPotential(d,neighbours[i],l1,l2);
                        }
                    }
                    optimizer->AddEdge(nodes[d], nodes[neighbours[i]], TRWType::EdgeData(TRWType::GENERAL,V));

                }
            
            }
        }
        for (int d=0;d<nSegNodes;++d){
            //pure Segmentation
            std::vector<int> neighbours= graph->getForwardSegmentationNeighbours(d);
            int nNeighbours=neighbours.size();
            for (int i=0;i<nNeighbours;++i){
                optimizer->AddEdge(nodes[d], nodes[neighbours[i]], TRWType::EdgeData(TRWType::POTTS,graph->getSegmentationWeight(d,nodes[neighbors[i]]));
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

#endif /* TRW_S_REGISTRATION_H_ */

#if 0
bool 	TransformPhysicalPointToIndex (const Point< TCoordRep, VImageDimension > &point, IndexType &index) const
template<class TCoordRep >
bool 	TransformPhysicalPointToContinuousIndex (const Point< TCoordRep, VImageDimension > &point, ContinuousIndex< TCoordRep, VImageDimension > &index) const
template<class TCoordRep >
void 	TransformContinuousIndexToPhysicalPoint (const ContinuousIndex< TCoordRep, VImageDimension > &index, Point< TCoordRep, VImageDimension > &point) const
template<class TCoordRep >
void 	TransformIndexToPhysicalPoint (const IndexType &index, Point< TCoordRep, VImageDimension > &point) const
#endif
