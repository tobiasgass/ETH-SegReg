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


    typedef TGraphModel GraphModelType;
	
	//	typedef Graph::Real Real;
	typedef TypeGeneral TRWType;
	typedef MRFEnergy<TRWType> MRFType;
	typedef typename MRFType::NodeId NodeType;

protected:
	MRFType* optimizer;
	double m_unarySegmentationWeight,m_pairwiseSegmentationWeight;
	double m_unaryRegistrationWeight,m_pairwiseRegistrationWeight;
	double m_pairwiseSegmentationRegistrationWeight;
	bool verbose;
    GraphModelType * m_GraphModel;
    int nNodes, nRegNodes, nSegNodes;
    NodeType* segNodes;
    NodeType* regNodes;
    clock_t m_start;
public:
	TRWS_SRSMRFSolver(GraphModelType * graphModel,
                      double unaryRegWeight=1.0, 
                      double pairwiseRegWeight=1.0, 
                      double unarySegWeight=1.0, 
                      double pairwiseSegWeight=1.0, 
                      double pairwiseSegRegWeight=1.0,
                      bool vverbose=false)
      :m_GraphModel(graphModel)
	{
		verbose=vverbose;
		m_unarySegmentationWeight=unarySegWeight;
		m_pairwiseSegmentationWeight=pairwiseSegWeight;
        m_unaryRegistrationWeight=unaryRegWeight;
        m_pairwiseRegistrationWeight=pairwiseRegWeight;
        m_pairwiseSegmentationRegistrationWeight=pairwiseSegRegWeight;
        createGraph();
	}
	~TRWS_SRSMRFSolver()
    {
        delete [] segNodes;
        delete [] regNodes;
        delete optimizer;

    }
	virtual void createGraph(){
		//		TRWType::GlobalSize globalSize(labelSampling,labelSampling);
		//		optimizer = new MRFType(globalSize);
        if (verbose) std::cout<<"starting graph init"<<std::endl;
        GraphModelType* graph=this->m_GraphModel;
        TRWType::GlobalSize globalSize();
		optimizer = new MRFType(TRWType::GlobalSize());
		nNodes=graph->nNodes();
        nRegNodes=graph->nRegNodes();
        nSegNodes=graph->nSegNodes();
       
        segNodes = new NodeType[nSegNodes];
        regNodes = new NodeType[nRegNodes];


        int nRegLabels=graph->nRegLabels();
        int nSegLabels=graph->nSegLabels();

		clock_t start = clock();
        m_start=start;
		//		traverse grid
        if (verbose) std::cout<<"RegUnaries "<<nRegNodes<<std::endl;
        //RegUnaries
		TRWType::REAL D1[nRegLabels];
		for (int d=0;d<nRegNodes;++d){
            for (int l1=0;l1<nRegLabels;++l1)
                {
                    D1[l1]=m_unaryRegistrationWeight*graph->getUnaryRegistrationPotential(d,l1);
                }
			regNodes[d] = optimizer->AddNode(TRWType::LocalSize(nRegLabels), TRWType::NodeData(D1));
		}
        if (verbose) std::cout<<"SegUnaries"<<std::endl;

        //SegUnaries
		TRWType::REAL D2[nSegLabels];
		for (int d=0;d<nSegNodes;++d){
            for (int l1=0;l1<nSegLabels;++l1)
                {
                    D2[l1]=m_unarySegmentationWeight*graph->getUnarySegmentationPotential(d,l1);
                }
			segNodes[d] = optimizer->AddNode(TRWType::LocalSize(nSegLabels), TRWType::NodeData(D2));
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
                    //std::cout<<d<<" "<<regNodes[d]<<" "<<i<<" "<<neighbours[i]<<std::endl;
                    TRWType::REAL V[nRegLabels*nRegLabels];
                    for (int l1=0;l1<nRegLabels;++l1){
                        for (int l2=0;l2<nRegLabels;++l2){
                            V[l1*nRegLabels+l2]=m_pairwiseRegistrationWeight*graph->getPairwiseRegistrationPotential(d,neighbours[i],l1,l2);
                        }
                    }
                    optimizer->AddEdge(regNodes[d], regNodes[neighbours[i]], TRWType::EdgeData(TRWType::GENERAL,V));
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
                            V[l1+l2*nRegLabels]=m_pairwiseSegmentationRegistrationWeight*graph->getPairwiseSegRegPotential(d,neighbours[i],l1,l2);
                        }
                    }
                    optimizer->AddEdge(regNodes[d], segNodes[neighbours[i]], TRWType::EdgeData(TRWType::GENERAL,V));

                }
            
            }
        }
        //  std::cout<<" reg and segreg pairwise pots" <<std::endl;
        TRWType::REAL V[nSegLabels*nSegLabels];
        for (int d=0;d<nSegNodes;++d){
            //pure Segmentation
            std::vector<int> neighbours= graph->getForwardSegmentationNeighbours(d);
            int nNeighbours=neighbours.size();
            for (int i=0;i<nNeighbours;++i){
                double lambda=m_pairwiseSegmentationWeight*graph->getSegmentationWeight(d,neighbours[i]);
                for (int l1=0;l1<nSegLabels;++l1){
					for (int l2=0;l2<nSegLabels;++l2){
						V[l1*nSegLabels+l2]=lambda*(l1!=l2);//graph->getPairwisePotential(l1,l2);
					}
				}
                optimizer->AddEdge(segNodes[d], segNodes[neighbours[i]], TRWType::EdgeData(TRWType::GENERAL,V));
                //                optimizer->AddEdge(segNodes[d], segNodes[neighbours[i]], TRWType::EdgeData(TRWType::POTTS,lambda));
              
            }
        }    
        //std::cout<<" seg pairwise pots" <<std::endl;
        clock_t finish = clock();
        t = (float) ((double)(finish - start) / CLOCKS_PER_SEC);
        if (verbose) std::cout<<"Finished init after "<<t<<" seconds"<<std::endl;

    }
    

    virtual void optimize(){
        MRFEnergy<TRWType>::Options options;
        TRWType::REAL energy, lowerBound;
        options.m_iterMax = 20; // maximum number of iterations
        options.m_printMinIter=1;
        options.m_printIter=1;
        clock_t start = clock();
        optimizer->Minimize_TRW_S(options, lowerBound, energy);
        clock_t finish = clock();
        float t = (float) ((double)(finish - m_start) / CLOCKS_PER_SEC);
        std::cout<<"Finished after "<<t<<" , resulting energy is "<<energy<<" with lower bound "<< lowerBound ;//<< std::endl;

    }

    virtual std::vector<int> getDeformationLabels(){
        std::vector<int> labels(nRegNodes);
        for (int i=0;i<nRegNodes;++i){
            labels[i]=optimizer->GetSolution(regNodes[i]);
        }
        return labels;
    }
    virtual std::vector<int> getSegmentationLabels(){
        std::vector<int> labels(nSegNodes);
        int c=0;
        for (int i=0;i<nSegNodes;++i){
            labels[i]=optimizer->GetSolution(segNodes[i]);
            if (labels[i])
                ++c;
        }
        //        std::cout <<" coutn"<<c<<std::endl;
        return labels;
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
