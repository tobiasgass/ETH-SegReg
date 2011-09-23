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
#include <vector>
#include <google/heap-profiler.h>
#include "ordering.cpp"
//#include "malloc.c"
using namespace std;

template<class TGraphModel>
class TRWS_SRSMRFSolver {
public:


    typedef TGraphModel GraphModelType;
	

	typedef TypeGeneral TRWType;
	typedef MRFEnergy<TRWType> MRFType;
    typedef TRWType::REAL Real;
	typedef typename MRFType::NodeId NodeType;
	typedef typename MRFType::EdgeId EdgeType;

protected:
    MRFType m_optimizer;
	double m_unarySegmentationWeight,m_pairwiseSegmentationWeight;
	double m_unaryRegistrationWeight,m_pairwiseRegistrationWeight;
	double m_pairwiseSegmentationRegistrationWeight;
	bool verbose;
    GraphModelType * m_GraphModel;
    int nNodes, nRegNodes, nSegNodes, nEdges;
    vector<NodeType> segNodes;
    vector<NodeType> regNodes;
    clock_t m_start;
    int nRegLabels;
    int nSegLabels;
public:
	TRWS_SRSMRFSolver(GraphModelType * graphModel,
                      double unaryRegWeight=1.0, 
                      double pairwiseRegWeight=1.0, 
                      double unarySegWeight=1.0, 
                      double pairwiseSegWeight=1.0, 
                      double pairwiseSegRegWeight=1.0,
                      bool vverbose=false)
        :m_optimizer(TRWType::GlobalSize()),m_GraphModel(graphModel)
	{
		verbose=vverbose;
		m_unarySegmentationWeight=unarySegWeight;
		m_pairwiseSegmentationWeight=pairwiseSegWeight;
        m_unaryRegistrationWeight=unaryRegWeight;
        m_pairwiseRegistrationWeight=pairwiseRegWeight;
        m_pairwiseSegmentationRegistrationWeight=pairwiseSegRegWeight;
        //createGraph();
        //init();
    }
    TRWS_SRSMRFSolver()  :m_optimizer(TRWType::GlobalSize()){}
	~TRWS_SRSMRFSolver()
    {
    }

 
	virtual void createGraph(){

        if (verbose) std::cout<<std::endl<<"starting graph init"<<std::endl;
        GraphModelType* graph=this->m_GraphModel;
    
		nNodes=graph->nNodes();
        nEdges=graph->nEdges();
        nRegNodes=graph->nRegNodes();
        nSegNodes=graph->nSegNodes();

        segNodes = vector<NodeType>(nSegNodes,NULL);
        regNodes = vector<NodeType>(nRegNodes,NULL);
   
		clock_t start = clock();
        m_start=start;

        int edgeCount=0;
        nRegLabels=graph->nRegLabels();
        nSegLabels=graph->nSegLabels();

		//		traverse grid
        if ( (m_pairwiseSegmentationRegistrationWeight || m_unaryRegistrationWeight>0 || m_pairwiseRegistrationWeight>0) && nRegLabels){
            //RegUnaries
            TRWType::REAL D1[nRegLabels];
            for (int d=0;d<nRegNodes;++d){
                for (int l1=0;l1<nRegLabels;++l1)
                    {
                        D1[l1]=m_unaryRegistrationWeight*graph->getUnaryRegistrationPotential(d,l1);
                    }
                regNodes[d] = 
                    m_optimizer.AddNode(TRWType::LocalSize(nRegLabels), TRWType::NodeData(D1));
                // Pairwise potentials
            }
            TRWType::REAL Vreg[nRegLabels*nRegLabels];
            for (int l1=0;l1<nRegLabels;++l1){
                for (int l2=0;l2<nRegLabels;++l2){
                    Vreg[l1*nRegLabels+l2]=0;
                }
            }

            for (int d=0;d<nRegNodes;++d){
            
                {//pure Registration
                    std::vector<int> neighbours= graph->getForwardRegistrationNeighbours(d);
                    int nNeighbours=neighbours.size();
                    for (int i=0;i<nNeighbours;++i){
                        //std::cout<<d<<" "<<regNodes[d]<<" "<<i<<" "<<neighbours[i]<<std::endl;

                        for (int l1=0;l1<nRegLabels;++l1){
                            for (int l2=0;l2<nRegLabels;++l2){
                                Vreg[l1*nRegLabels+l2]=m_pairwiseRegistrationWeight*graph->getPairwiseRegistrationPotential(d,neighbours[i],l1,l2);
                            }
                        }
                    
                    
                        // edges[edgeCount]=
                        m_optimizer.AddEdge(regNodes[d], regNodes[neighbours[i]], TRWType::EdgeData(TRWType::GENERAL,Vreg));
                        edgeCount++;
                    }
                
                }
            }
            
        }
        if ( (m_pairwiseSegmentationRegistrationWeight || m_unarySegmentationWeight>0 || m_pairwiseSegmentationWeight) && nSegLabels){

            //SegUnaries
            TRWType::REAL D2[nSegLabels];
            for (int d=0;d<nSegNodes;++d){
                for (int l1=0;l1<nSegLabels;++l1)
                    {
                        D2[l1]=m_unarySegmentationWeight*graph->getUnarySegmentationPotential(d,l1);
                        //  std::cout<<d<<" "<< D2[l1] <<std::endl;
                    }
                segNodes[d] = 
                    m_optimizer.AddNode(TRWType::LocalSize(nSegLabels), TRWType::NodeData(D2));
                
                //  std::cout<<" reg and segreg pairwise pots" <<std::endl;
       
            }
            
            TRWType::REAL VsrsBack[nRegLabels*nSegLabels];
            for (int d=0;d<nSegNodes;++d){   
                TRWType::REAL Vseg[nSegLabels*nSegLabels];
                //                TRWType::REAL Vseg2[nSegLabels*nSegLabels];
                
                //pure Segmentation
                std::vector<int> neighbours= graph->getForwardSegmentationNeighbours(d);
                int nNeighbours=neighbours.size();
                for (int i=0;i<nNeighbours;++i){
                    double lambda=m_pairwiseSegmentationWeight*graph->getSegmentationWeight(d,neighbours[i]);   
                    //std::cout<<graph->getSegmentationWeight(d,neighbours[i])<<" "<<m_pairwiseSegmentationWeight<<" "<<(double)m_pairwiseSegmentationWeight*graph->getSegmentationWeight(d,neighbours[i])<<std::endl;
                    //double lambda2=m_pairwiseSegmentationWeight*graph->getSegmentationWeight(neighbours[i],d);
                    for (int l1=0;l1<nSegLabels;++l1){
                        for (int l2=0;l2<nSegLabels;++l2){
                            Vseg[l1*nSegLabels+l2]=lambda*(l1!=l2);
                            //  Vseg[l1*nSegLabels+l2+nSegLabels*nSegLabels]=lambda2*(l1!=l2);
                            //      Vseg2[l1*nSegLabels+l2]=lambda2*(l1!=l2);
                        }
                    }
                    //m_optimizer.AddEdge(segNodes[d], segNodes[neighbours[i]], TRWType::EdgeData(TRWType::POTTS,lambda));
                    m_optimizer.AddEdge(segNodes[d], segNodes[neighbours[i]], TRWType::EdgeData(TRWType::GENERAL,Vseg));
                    //m_optimizer.AddEdge(segNodes[neighbours[i]],segNodes[d], TRWType::EdgeData(TRWType::GENERAL,Vseg2));
                    edgeCount++;
                    
                }
                if (m_pairwiseSegmentationRegistrationWeight>0 && nRegLabels){
                    std::vector<int> segRegNeighbors=graph->getSegRegNeighbors(d);
                    nNeighbours=segRegNeighbors.size();
                    for (int i=0;i<nNeighbours;++i){
                        
                        for (int l1=0;l1<nRegLabels;++l1){
                            for (int l2=0;l2<nSegLabels;++l2){
                                //forward
                                VsrsBack[l1+l2*nRegLabels]=m_pairwiseSegmentationRegistrationWeight*graph->getPairwiseRegSegPotential(segRegNeighbors[i],d,l1,l2);
                            }
                        }
                        m_optimizer.AddEdge(regNodes[segRegNeighbors[i]], segNodes[d]             , TRWType::EdgeData(TRWType::GENERAL,VsrsBack));
                        edgeCount++;
                    }
                }
                
            }
        }
        clock_t finish = clock();
        clock_t t = (float) ((double)(finish - start) / CLOCKS_PER_SEC);
        if (verbose) std::cout<<"Finished init after "<<t<<" seconds"<<std::endl;
        nEdges=edgeCount;

    }
    

    virtual void optimize(int maxIter=20){
        if (verbose) std::cout<<"EDGES " <<nEdges<<endl;
        //m_optimizer.SetAutomaticOrdering();

        MRFEnergy<TRWType>::Options options;
        TRWType::REAL energy=-1, lowerBound=-1;
        options.m_iterMax = maxIter; // maximum number of iterations
        options.m_printMinIter=1;
        options.m_printIter=1;
        options.verbose=verbose;
        options.m_eps=-1;
        m_optimizer.Minimize_TRW_S(options, lowerBound, energy);
        clock_t finish = clock();
        float t = (float) ((double)(finish - m_start) / CLOCKS_PER_SEC);
        std::cout<<"Finished after "<<t<<" , resulting energy is "<<energy<<" with lower bound "<< lowerBound ;//std::endl;

        if (verbose){
            std::cout<<std::endl;
            //            evalSolution();
        }
    }

    virtual std::vector<int> getDeformationLabels(){
        std::vector<int> labels(nRegNodes,0);
        if (nRegLabels){
            for (int i=0;i<nRegNodes;++i){
                labels[i]=m_optimizer.GetSolution(regNodes[i]);
            }
        }
        return labels;
    }
    virtual std::vector<int> getSegmentationLabels(){
        std::vector<int> labels(nSegNodes,0);
        if (nSegLabels){
            for (int i=0;i<nSegNodes;++i){
                labels[i]=m_optimizer.GetSolution(segNodes[i]);
                //labels[i]=this->m_GraphModel->getUnarySegmentationPotential(i,1)*nSegNodes;//m_optimizer.GetSolution(segNodes[i]);
            }
        }
        return labels;
    }

    void evalSolution(){
        double sumUReg=0,sumUSeg=0,sumPSeg=0,sumPSegReg=0;
        GraphModelType* graph=this->m_GraphModel;
        
		clock_t start = clock();
        m_start=start;
        if (nRegLabels){
            for (int d=0;d<nRegNodes;++d){
                sumUReg+=m_unaryRegistrationWeight*graph->getUnaryRegistrationPotential(d,m_optimizer.GetSolution(regNodes[d]));
            }
        }
        if (nSegLabels){
            for (int d=0;d<nSegNodes;++d){
                sumUSeg+=m_unarySegmentationWeight*graph->getUnarySegmentationPotential(d,m_optimizer.GetSolution(segNodes[d]));
                if (nRegLabels){
                    std::vector<int> neighbours= graph->getForwardSegmentationNeighbours(d);
                    int nNeighbours=neighbours.size();
                    for (int i=0;i<nNeighbours;++i){
                        sumPSeg+=m_pairwiseSegmentationWeight*graph->getSegmentationWeight(d,neighbours[i])*(m_optimizer.GetSolution(segNodes[d])!=m_optimizer.GetSolution(segNodes[neighbours[i]]));
                    }
                    std::vector<int> segRegNeighbors=graph->getSegRegNeighbor(d);
                    for (unsigned int n=0;n<segRegNeighbors.size();++n){
                        sumPSegReg+=m_pairwiseSegmentationRegistrationWeight*graph->getPairwiseRegSegPotential(segRegNeighbors[n],d,m_optimizer.GetSolution(regNodes[segRegNeighbors[n]]),m_optimizer.GetSolution(segNodes[d]));
                    }
                }


            }   
        }
        std::cout<<"RegU :\t\t"<<sumUReg<<endl
                 <<"SegU :\t\t"<<sumUSeg<<endl
                 <<"SegP :\t\t"<<sumPSeg<<endl
                 <<"SegRegP :\t"<<sumPSegReg<<endl;
        clock_t finish = clock();
        double t = (float) ((double)(finish - start) / CLOCKS_PER_SEC);
        if (verbose) std::cout<<"Finished init after "<<t<<" seconds"<<std::endl;
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
