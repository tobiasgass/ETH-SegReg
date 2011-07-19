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
//#include "malloc.c"
using namespace std;

template<class TGraphModel>
class TRWS_SRSMRFSolver {
public:


    typedef TGraphModel GraphModelType;
	
	//	typedef Graph::Real Real;
	typedef TypeGeneral TRWType;
	typedef MRFEnergy<TRWType> MRFType;
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
    vector<EdgeType> edges;
    vector< vector<int> > pointers;
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
        cout<<"construction"<<endl;
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
        for (unsigned int i=0; i<pointers.size();++i){
        }
        cout<<"deleted mem"<<endl;
     }

    void init2(){
        clock_t start = clock();
        m_start=start;
        int nLabels=10;
        int nNodes=3000000;
        

        for (unsigned int i=0;i<nNodes;++i){
            pointers.push_back( vector<int>(nLabels*nLabels));
        }
        //cout<<pointers[0]<<endl;
        sleep(1);
        cout<<"allocated"<<endl;
        HeapProfilerDump("dump after alloc");
        for (unsigned int i=0;i<pointers.size();++i){
            //  pointers[i]->clear();
            //            delete pointers[i];
            //pointers[i]=NULL;
        }
        HeapProfilerDump("dump after delete");
        sleep(1);
        nEdges=1;
      
        
    }
    void init(){        
        GraphModelType* graph=this->m_GraphModel;
        clock_t start = clock();
        m_start=start;
        
        nEdges=1;
        nRegNodes=graph->nRegNodes();
        nSegNodes=graph->nSegNodes();
        int nLabels=10;
        int nLabels2=3;
        int nNodes=3000000;
        //m_optimizer=MRFType(TRWType::GlobalSize());
        vector<NodeType> nodes(nNodes*2, NULL);


        TRWType::GlobalSize globalSize();
        TRWType::REAL D1[nLabels];
        TRWType::REAL D2[nLabels2];
        for (int n=0;n<nNodes;++n){
            nodes[n]=m_optimizer.AddNode(TRWType::LocalSize(nLabels), TRWType::NodeData(D1));
            nodes[n+nNodes]=m_optimizer.AddNode(TRWType::LocalSize(nLabels2), TRWType::NodeData(D2));
        }
        TRWType::REAL V[nLabels*nLabels];
        TRWType::REAL V2[nLabels*nLabels];
        int c=0;
        for (int n=0;n<nNodes-1;++n){
            assert(nodes[n]);
            assert(nodes[n+1]);
            assert(nodes[n+nNodes-1]);
            assert(n+nNodes-1<nNodes*2-2);
            // edges[n]= 
                  m_optimizer.AddEdge(nodes[n], nodes[n+1], TRWType::EdgeData(TRWType::GENERAL,V));
            //            edges[n+nNodes-1]= 
            m_optimizer.AddEdge(nodes[n], nodes[n+nNodes-1], TRWType::EdgeData(TRWType::GENERAL,V2));
            c+=2;
        }
        std::cout<<edges.size()<<" "<<c<<std::endl;
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
        int nRegLabels=graph->nRegLabels();
        int nSegLabels=graph->nSegLabels();

		//		traverse grid
        if (verbose) std::cout<<"RegUnaries "<<nRegNodes<<std::endl;
        //RegUnaries
		TRWType::REAL D1[nRegLabels];
		for (int d=0;d<nRegNodes;++d){
            for (int l1=0;l1<nRegLabels;++l1)
                {
                    D1[l1]=m_unaryRegistrationWeight*graph->getUnaryRegistrationPotential(d,l1);
                }
            regNodes[d] = 
                m_optimizer.AddNode(TRWType::LocalSize(nRegLabels), TRWType::NodeData(D1));
            
		}
        if (verbose) std::cout<<"SegUnaries"<<std::endl;

        //SegUnaries
		TRWType::REAL D2[nSegLabels];
		for (int d=0;d<nSegNodes;++d){
            for (int l1=0;l1<nSegLabels;++l1)
                {
                    D2[l1]=m_unarySegmentationWeight*graph->getUnarySegmentationPotential(d,l1);
                }
            segNodes[d] = 
                m_optimizer.AddNode(TRWType::LocalSize(nSegLabels), TRWType::NodeData(D2));
		}

		clock_t finish1 = clock();
		float t = (float) ((double)(finish1 - start) / CLOCKS_PER_SEC);
		if (verbose) std::cout<<"Finished unary potential initialisation after "<<t<<" seconds"<<std::endl;

		// Pairwise potentials
#if 0
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
#endif

        //  std::cout<<" reg and segreg pairwise pots" <<std::endl;
        TRWType::REAL Vseg[nSegLabels*nSegLabels];
        TRWType::REAL Vsrs[nRegLabels*nSegLabels];

        for (int d=0;d<nSegNodes;++d){
            //pure Segmentation
            std::vector<int> neighbours= graph->getForwardSegmentationNeighbours(d);
            int nNeighbours=neighbours.size();
            for (int i=0;i<nNeighbours;++i){
                double lambda=m_pairwiseSegmentationWeight*graph->getSegmentationWeight(d,neighbours[i]);
                for (int l1=0;l1<nSegLabels;++l1){
					for (int l2=0;l2<nSegLabels;++l2){
						Vseg[l1*nSegLabels+l2]=lambda*(l1!=l2);//graph->getPairwisePotential(l1,l2);
					}
				}
                m_optimizer.AddEdge(segNodes[d], segNodes[neighbours[i]], TRWType::EdgeData(TRWType::GENERAL,Vseg));
                edgeCount++;
              
            }
            int segRegNeighbor=graph->getSegRegNeighbor(d);
            for (int l1=0;l1<nRegLabels;++l1){
                for (int l2=0;l2<nSegLabels;++l2){
                    Vsrs[l1+l2*nRegLabels]=m_pairwiseSegmentationRegistrationWeight*graph->getPairwiseSegRegPotential(segRegNeighbor,d,l1,l2);
                }
            }
            m_optimizer.AddEdge(regNodes[segRegNeighbor], segNodes[d], TRWType::EdgeData(TRWType::GENERAL,Vsrs));
            edgeCount++;

        }   
        clock_t finish = clock();
        t = (float) ((double)(finish - start) / CLOCKS_PER_SEC);
        if (verbose) std::cout<<"Finished init after "<<t<<" seconds"<<std::endl;
        std::cout<<edgeCount<<" "<<nEdges<<endl;
        nEdges=edgeCount;

    }
    

    virtual void optimize(){
        std::cout<<"EDGES " <<nEdges<<endl;
        MRFEnergy<TRWType>::Options options;
        TRWType::REAL energy=-1, lowerBound=-1;
        options.m_iterMax = 20; // maximum number of iterations
        options.m_printMinIter=1;
        options.m_printIter=1;
        options.verbose=verbose;
        m_optimizer.Minimize_TRW_S(options, lowerBound, energy);
        clock_t finish = clock();
        float t = (float) ((double)(finish - m_start) / CLOCKS_PER_SEC);
        std::cout<<"Finished after "<<t<<" , resulting energy is "<<energy<<" with lower bound "<< lowerBound << std::endl;

    }

    virtual std::vector<int> getDeformationLabels(){
        std::vector<int> labels(nRegNodes);
        for (int i=0;i<nRegNodes;++i){
            labels[i]=m_optimizer.GetSolution(regNodes[i]);
        }
        return labels;
    }
    virtual std::vector<int> getSegmentationLabels(){
        std::vector<int> labels(nSegNodes);
        int c=0;
        for (int i=0;i<nSegNodes;++i){
            labels[i]=m_optimizer.GetSolution(segNodes[i]);
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
