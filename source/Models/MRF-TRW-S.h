#include "Log.h"
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
#include "typeTruncatedQuadratic3D.h"
#include "typeGeneral.h"
#include "MRFEnergy.h"
#include "minimize.cpp"
#include "treeProbabilities.cpp"
#include <vector>
#include <google/heap-profiler.h>
#include "ordering.cpp"
#include <limits.h>
//#include "malloc.c"
using namespace std;
double tOpt=0;
double tUnary=0;
double tPairwise=0;

template<class TGraphModel>
class TRWS_SRSMRFSolver {
public:


    typedef TGraphModel GraphModelType;
	typedef typename GraphModelType::Pointer GraphModelPointerType;

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
    GraphModelPointerType m_GraphModel;
    int nNodes, nRegNodes, nSegNodes, nEdges;
    vector<NodeType> segNodes;
    vector<NodeType> regNodes;
    clock_t m_start;
    int nRegLabels;
    int nSegLabels;
    bool m_segmented, m_registered;
public:
	TRWS_SRSMRFSolver(GraphModelPointerType  graphModel,
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
    TRWS_SRSMRFSolver()  :m_optimizer(TRWType::GlobalSize()){
        
    }
	~TRWS_SRSMRFSolver()
    {
    }

 
	virtual void createGraph(){
        clock_t start = clock();
        {
            m_segmented=false; 
            m_registered=false;
            m_segmented=0;
            m_registered=0;
        }
        LOGV(1)<<"starting graph init"<<std::endl;
        this->m_GraphModel->Init();
        clock_t endUnary = clock();
        double t1 = (float) ((double)(endUnary - start) / CLOCKS_PER_SEC);
        tUnary+=t1;       

		nNodes=this->m_GraphModel->nNodes();
        nEdges=this->m_GraphModel->nEdges();
        nRegNodes=this->m_GraphModel->nRegNodes();
        nSegNodes=this->m_GraphModel->nSegNodes();

        segNodes = vector<NodeType>(nSegNodes,NULL);
        regNodes = vector<NodeType>(nRegNodes,NULL);
   
        m_start=start;

        int edgeCount=0;
        nRegLabels=this->m_GraphModel->nRegLabels();
        nSegLabels=this->m_GraphModel->nSegLabels();
        logSetStage("Potential Functions");
		//		traverse grid
        if ( (m_pairwiseSegmentationRegistrationWeight || m_unaryRegistrationWeight>0 || m_pairwiseRegistrationWeight>0) && nRegLabels){
            //RegUnaries
            clock_t startUnary = clock();
            m_registered=true;
            TRWType::REAL D1[nRegLabels];
            for (int d=0;d<nRegNodes;++d){
                for (int l1=0;l1<nRegLabels;++l1)
                    {
                        if (m_unaryRegistrationWeight>0)
                            D1[l1]=m_unaryRegistrationWeight*this->m_GraphModel->getUnaryRegistrationPotential(d,l1);
                        else
                            D1[l1]=0;
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
            clock_t endUnary = clock();
            double t = (float) ((double)(endUnary - startUnary) / CLOCKS_PER_SEC);
            LOGV(1)<<"Registration Unaries took "<<t<<" seconds."<<endl;
            tUnary+=t;
            for (int d=0;d<nRegNodes;++d){
            
                {//pure Registration
                    std::vector<int> neighbours= this->m_GraphModel->getForwardRegistrationNeighbours(d);
                    int nNeighbours=neighbours.size();
                    for (int i=0;i<nNeighbours;++i){
                        //LOG<<d<<" "<<regNodes[d]<<" "<<i<<" "<<neighbours[i]<<std::endl;

                        for (int l1=0;l1<nRegLabels;++l1){
                            for (int l2=0;l2<nRegLabels;++l2){
                                if (m_pairwiseRegistrationWeight>0)
                                    Vreg[l1*nRegLabels+l2]=m_pairwiseRegistrationWeight*this->m_GraphModel->getPairwiseRegistrationPotential(d,neighbours[i],l1,l2);
                                else{
                                    Vreg[l1*nRegLabels+l2]=0;
                                }
                            }
                        }
                    
                    
                        // edges[edgeCount]=
                        m_optimizer.AddEdge(regNodes[d], regNodes[neighbours[i]], TRWType::EdgeData(TRWType::GENERAL,Vreg));
                        edgeCount++;
                    }
                
                }
            }
            clock_t endPairwise = clock();
         
            t = (float) ((double)(endPairwise-endUnary ) / CLOCKS_PER_SEC);
            LOGV(1)<<"Registration pairwise took "<<t<<" seconds."<<endl;

            tPairwise+=t;
        }
        if ( (m_pairwiseSegmentationRegistrationWeight || m_unarySegmentationWeight>0 || m_pairwiseSegmentationWeight) && nSegLabels){
            m_segmented=true;
            //SegUnaries
            clock_t startUnary = clock();
            TRWType::REAL D2[nSegLabels];
            for (int d=0;d<nSegNodes;++d){
                for (int l1=0;l1<nSegLabels;++l1)
                    {
                        D2[l1]=m_unarySegmentationWeight*this->m_GraphModel->getUnarySegmentationPotential(d,l1);
                        //  LOG<<d<<" "<< D2[l1] <<std::endl;
                    }
                segNodes[d] = 
                    m_optimizer.AddNode(TRWType::LocalSize(nSegLabels), TRWType::NodeData(D2));
                
                //  LOG<<" reg and segreg pairwise pots" <<std::endl;
       
            }
            clock_t endUnary = clock();
            double t = (float) ((double)(endUnary - startUnary) / CLOCKS_PER_SEC);
            LOGV(1)<<"Segmentation Unaries took "<<t<<" seconds."<<endl;
            LOGV(1)<<"Approximate size of seg unaries: "<<1.0/(1024*1024)*nSegNodes*nSegLabels*sizeof(double)<<" mb."<<endl;

            TRWType::REAL VsrsBack[nRegLabels*nSegLabels];
            int nSegEdges=0,nSegRegEdges=0;
            for (int d=0;d<nSegNodes;++d){   
                TRWType::REAL Vseg[nSegLabels*nSegLabels];
                //pure Segmentation
                std::vector<int> neighbours= this->m_GraphModel->getForwardSegmentationNeighbours(d);
                int nNeighbours=neighbours.size();
                for (int i=0;i<nNeighbours;++i){
                    nSegEdges++;
                    for (int l1=0;l1<nSegLabels;++l1){
                        for (int l2=0;l2<nSegLabels;++l2){
                            double lambda =m_pairwiseSegmentationWeight*this->m_GraphModel->getPairwiseSegmentationPotential(d,neighbours[i],l1,l2);
                            Vseg[l1+nSegLabels*l2]=lambda;
                            //LOG<<"PAIRWISE " <<d<<" "<<i<<" "<<l1<<" "<<l2<<" "<<                            Vseg[l1+nSegLabels*l2]<<endl;
                        }
                    }
                    m_optimizer.AddEdge(segNodes[d], segNodes[neighbours[i]], TRWType::EdgeData(TRWType::GENERAL,Vseg));
                    edgeCount++;
                    
                }
                if (m_pairwiseSegmentationRegistrationWeight>0 && (nRegLabels>1)){
                    std::vector<int> segRegNeighbors=this->m_GraphModel->getSegRegNeighbors(d);
                    nNeighbours=segRegNeighbors.size();
                    if (nNeighbours==0) {LOG<<"ERROR: node "<<d<<" seems to have no neighbors."<<std::endl;}
                    for (int i=0;i<nNeighbours;++i){
                        for (int l1=0;l1<nRegLabels;++l1){
                            for (int l2=0;l2<nSegLabels;++l2){
                                //forward
                                VsrsBack[l1+l2*nRegLabels]=m_pairwiseSegmentationRegistrationWeight*this->m_GraphModel->getPairwiseRegSegPotential(segRegNeighbors[i],d,l1,l2);
                            }
                        }
                        m_optimizer.AddEdge(regNodes[segRegNeighbors[i]], segNodes[d], TRWType::EdgeData(TRWType::GENERAL,VsrsBack));
                        edgeCount++;
                        nSegRegEdges++;
                    }
                }
                
            }
            clock_t endPairwise = clock();
            t = (float) ((double)(endPairwise-endUnary ) / CLOCKS_PER_SEC);
            LOGV(1)<<"Segmentation + SRS pairwise took "<<t<<" seconds."<<endl;
            LOGV(1)<<"Approximate size of seg pairwise: "<<1.0/(1024*1024)*nSegEdges*nSegLabels*nSegLabels*sizeof(double)<<" mb."<<endl;
            LOGV(1)<<"Approximate size of SRS pairwise: "<<1.0/(1024*1024)*nSegRegEdges*nSegLabels*nRegLabels*sizeof(double)<<" mb."<<endl;
            
        }
        clock_t finish = clock();
        double t = (float) ((double)(finish - start) / CLOCKS_PER_SEC);
        //tInterpolation+=t;
        LOGV(1)<<"Finished init after "<<t<<" seconds"<<std::endl;
        nEdges=edgeCount;
        logResetStage;
    }
    

    virtual void optimize(int maxIter=20){
        LOGV(1)<<"Total number of MRF edges: " <<nEdges<<endl;
        //m_optimizer.SetAutomaticOrdering();

        MRFEnergy<TRWType>::Options options;
        TRWType::REAL energy=-1, lowerBound=-1;
        options.m_iterMax = maxIter; // maximum number of iterations
        options.m_printMinIter=1;
        options.m_printIter=1;
        options.verbose=verbose;
        options.m_eps=-1;
        logSetStage("Optimizer");
        clock_t opt_start=clock();
        m_optimizer.Minimize_TRW_S(options, lowerBound, energy);
        clock_t finish = clock();
        logResetStage;         
        tOpt+=((double)(finish-opt_start)/CLOCKS_PER_SEC);
        float t = (float) ((double)(finish - m_start) / CLOCKS_PER_SEC);
        LOG<<"Finished optimization after "<<t<<" , resulting energy is "<<energy<<" with lower bound "<< lowerBound <<std::endl;


    }

    virtual std::vector<int> getDeformationLabels(){
        std::vector<int> labels(nRegNodes,0);
        if (m_registered){
            for (int i=0;i<nRegNodes;++i){

                labels[i]=m_optimizer.GetSolution(regNodes[i]);
            }
        }
        return labels;
    }
    virtual std::vector<int> getSegmentationLabels(){
        std::vector<int> labels(nSegNodes,0);
        if (m_segmented){
            for (int i=0;i<nSegNodes;++i){
                labels[i]=m_optimizer.GetSolution(segNodes[i]);
                //labels[i]=this->m_GraphModel->getUnarySegmentationPotential(i,1)*nSegNodes;//m_optimizer.GetSolution(segNodes[i]);
            }
        }
        return labels;
    }

    void evalSolution(){
        double sumUReg=0,sumUSeg=0,sumPSeg=0,sumPSegReg=0;
        
		clock_t start = clock();
        m_start=start;
        if (nRegLabels){
            for (int d=0;d<nRegNodes;++d){
                sumUReg+=m_unaryRegistrationWeight*this->m_GraphModel->getUnaryRegistrationPotential(d,m_optimizer.GetSolution(regNodes[d]));
            }
        }
        if (nSegLabels){
            for (int d=0;d<nSegNodes;++d){
                sumUSeg+=m_unarySegmentationWeight*this->m_GraphModel->getUnarySegmentationPotential(d,m_optimizer.GetSolution(segNodes[d]));
                if (nRegLabels){
                    std::vector<int> neighbours= this->m_GraphModel->getForwardSegmentationNeighbours(d);
                    int nNeighbours=neighbours.size();
                    for (int i=0;i<nNeighbours;++i){
                        sumPSeg+=m_pairwiseSegmentationWeight*this->m_GraphModel->getSegmentationWeight(d,neighbours[i])*(m_optimizer.GetSolution(segNodes[d])!=m_optimizer.GetSolution(segNodes[neighbours[i]]));
                    }
                    std::vector<int> segRegNeighbors=this->m_GraphModel->getSegRegNeighbor(d);
                    for (unsigned int n=0;n<segRegNeighbors.size();++n){
                        sumPSegReg+=m_pairwiseSegmentationRegistrationWeight*this->m_GraphModel->getPairwiseRegSegPotential(segRegNeighbors[n],d,m_optimizer.GetSolution(regNodes[segRegNeighbors[n]]),m_optimizer.GetSolution(segNodes[d]));
                    }
                }


            }   
        }
        LOG<<"RegU :\t\t"<<sumUReg<<endl
                 <<"SegU :\t\t"<<sumUSeg<<endl
                 <<"SegP :\t\t"<<sumPSeg<<endl
                 <<"SegRegP :\t"<<sumPSegReg<<endl;
        clock_t finish = clock();
        double t = (float) ((double)(finish - start) / CLOCKS_PER_SEC);
        LOGV(1)<<"Finished init after "<<t<<" seconds"<<std::endl;
    }

};
template<class TGraphModel>
class TRWS_SRSMRFSolverTruncQuadrat2D {
public:


    typedef TGraphModel GraphModelType;
	typedef typename GraphModelType::LabelMapperType LabelMapperType;
    
	typedef TypeTruncatedQuadratic2D TRWType;
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
    bool m_segmented, m_registered;
public:
	TRWS_SRSMRFSolverTruncQuadrat2D(GraphModelType * graphModel,
                                    double unaryRegWeight=1.0, 
                                    double pairwiseRegWeight=1.0, 
                                    double unarySegWeight=1.0, 
                                    double pairwiseSegWeight=1.0, 
                                    double pairwiseSegRegWeight=1.0,
                                    bool vverbose=false)
        :m_optimizer(TRWType::GlobalSize(2*LabelMapperType::nDisplacementSamples+1,2*LabelMapperType::nDisplacementSamples+1)),m_GraphModel(graphModel)
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
    TRWS_SRSMRFSolverTruncQuadrat2D()  :m_optimizer(TRWType::GlobalSize(7,7)){
        
    }
    ~TRWS_SRSMRFSolverTruncQuadrat2D()
    {
    }

 
    virtual void createGraph(){
        clock_t start = clock();
        {
            m_segmented=false; 
            m_registered=false;
            m_segmented=0;
            m_registered=0;
        }
        LOGV(1)<<"starting graph init"<<std::endl;
        this->m_GraphModel->Init();
        clock_t endUnary = clock();
        double t1 = (float) ((double)(endUnary - start) / CLOCKS_PER_SEC);
        tUnary+=t1;       
        nNodes=this->m_GraphModel->nNodes();
        nEdges=this->m_GraphModel->nEdges();
        nRegNodes=this->m_GraphModel->nRegNodes();
        nSegNodes=this->m_GraphModel->nSegNodes();

        segNodes = vector<NodeType>(nSegNodes,NULL);
        regNodes = vector<NodeType>(nRegNodes,NULL);
   
        m_start=start;

        int edgeCount=0;
        nRegLabels=this->m_GraphModel->nRegLabels();
        nSegLabels=this->m_GraphModel->nSegLabels();

        //		traverse grid
        if ( (m_pairwiseSegmentationRegistrationWeight || m_unaryRegistrationWeight>0 || m_pairwiseRegistrationWeight>0) && nRegLabels){
            //RegUnaries
            m_registered=true;
            clock_t startUnary = clock();

            TRWType::REAL D1[nRegLabels];
            for (int d=0;d<nRegNodes;++d){
                for (int l1=0;l1<nRegLabels;++l1)
                    {
                        if (m_unaryRegistrationWeight>0)
                            D1[l1]=m_unaryRegistrationWeight*this->m_GraphModel->getUnaryRegistrationPotential(d,l1);
                        else
                            D1[l1]=0;
                    }
                regNodes[d] = 
                    m_optimizer.AddNode(TRWType::LocalSize(), TRWType::NodeData(D1));
                // Pairwise potentials
            }
            clock_t endUnary = clock();
            for (int d=0;d<nRegNodes;++d){
            
                {//pure Registration
                    std::vector<int> neighbours= this->m_GraphModel->getForwardRegistrationNeighbours(d);
                    int nNeighbours=neighbours.size();
                    for (int i=0;i<nNeighbours;++i){
                        //LOG<<d<<" "<<regNodes[d]<<" "<<i<<" "<<neighbours[i]<<std::endl;
                        // edges[edgeCount]=
                        m_optimizer.AddEdge(regNodes[d], regNodes[neighbours[i]], TRWType::EdgeData(m_pairwiseRegistrationWeight,m_pairwiseRegistrationWeight,std::numeric_limits<double>::max()));//1000000*(m_pairwiseRegistrationWeight)));
                        edgeCount++;
                    }
                
                }
            }
            clock_t endPairwise = clock();
            double t = (float) ((double)(endUnary - startUnary) / CLOCKS_PER_SEC);
            tUnary+=t;
            t = (float) ((double)(endPairwise-endUnary ) / CLOCKS_PER_SEC);
            tPairwise+=t;
            
        }
    
        clock_t finish = clock();
        double t = (float) ((double)(finish - start) / CLOCKS_PER_SEC);
        LOGV(1)<<"Finished init after "<<t<<" seconds"<<std::endl;
        nEdges=edgeCount;

    }
    

    virtual void optimize(int maxIter=20){
        LOGV(1)<<"Total number of MRF edges: " <<nEdges<<endl;
        //m_optimizer.SetAutomaticOrdering();

        MRFEnergy<TRWType>::Options options;
        TRWType::REAL energy=-1, lowerBound=-1;
        options.m_iterMax = maxIter; // maximum number of iterations
        options.m_printMinIter=1;
        options.m_printIter=1;
        options.verbose=verbose;
        options.m_eps=-1;
        clock_t opt_start=clock();
        m_optimizer.Minimize_TRW_S(options, lowerBound, energy);
        clock_t finish = clock();
        tOpt+=((double)(finish-opt_start)/CLOCKS_PER_SEC);
        float t = (float) ((double)(finish - m_start) / CLOCKS_PER_SEC);
        LOG<<"Finished after "<<t<<" , resulting energy is "<<energy<<" with lower bound "<< lowerBound ;//std::endl;

        if (verbose){
            LOG<<std::endl;
            //            evalSolution();
        }
    }

    virtual std::vector<int> getDeformationLabels(){
        std::vector<int> labels(nRegNodes,0);
        if (m_registered){
            for (int i=0;i<nRegNodes;++i){
                TRWType::Label l=m_optimizer.GetSolution(regNodes[i]);
                labels[i]=l.m_kx+l.m_ky*(2*LabelMapperType::nDisplacementSamples+1);
            }
        }
        return labels;
    }
    virtual std::vector<int> getSegmentationLabels(){
        std::vector<int> labels(nSegNodes,0);
        if (m_segmented){
            for (int i=0;i<nSegNodes;++i){
                //labels[i]=m_optimizer.GetSolution(segNodes[i]);
                //labels[i]=this->m_GraphModel->getUnarySegmentationPotential(i,1)*nSegNodes;//m_optimizer.GetSolution(segNodes[i]);
            }
        }
        return labels;
    }

    void evalSolution(){
        double sumUReg=0,sumUSeg=0,sumPSeg=0,sumPSegReg=0;
        
        clock_t start = clock();
        m_start=start;
        if (nRegLabels){
            for (int d=0;d<nRegNodes;++d){
                sumUReg+=m_unaryRegistrationWeight*this->m_GraphModel->getUnaryRegistrationPotential(d,m_optimizer.GetSolution(regNodes[d]));
            }
        }
        if (nSegLabels){
            for (int d=0;d<nSegNodes;++d){
                sumUSeg+=m_unarySegmentationWeight*this->m_GraphModel->getUnarySegmentationPotential(d,m_optimizer.GetSolution(segNodes[d]));
                if (nRegLabels){
                    std::vector<int> neighbours= this->m_GraphModel->getForwardSegmentationNeighbours(d);
                    int nNeighbours=neighbours.size();
                    for (int i=0;i<nNeighbours;++i){
                        //sumPSeg+=m_pairwiseSegmentationWeight*this->m_GraphModel->getSegmentationWeight(d,neighbours[i])*(m_optimizer.GetSolution(segNodes[d])!=m_optimizer.GetSolution(segNodes[neighbours[i]]));
                    }
                    std::vector<int> segRegNeighbors=this->m_GraphModel->getSegRegNeighbor(d);
                    for (unsigned int n=0;n<segRegNeighbors.size();++n){
                        sumPSegReg+=m_pairwiseSegmentationRegistrationWeight*this->m_GraphModel->getPairwiseRegSegPotential(segRegNeighbors[n],d,m_optimizer.GetSolution(regNodes[segRegNeighbors[n]]),m_optimizer.GetSolution(segNodes[d]));
                    }
                }


            }   
        }
        LOG<<"RegU :\t\t"<<sumUReg<<endl
                 <<"SegU :\t\t"<<sumUSeg<<endl
                 <<"SegP :\t\t"<<sumPSeg<<endl
                 <<"SegRegP :\t"<<sumPSegReg<<endl;
        clock_t finish = clock();
        double t = (float) ((double)(finish - start) / CLOCKS_PER_SEC);
        LOGV(1)<<"Finished init after "<<t<<" seconds"<<std::endl;
    }

};
template<class TGraphModel>
class TRWS_SRSMRFSolverTruncQuadrat3D {
public:


    typedef TGraphModel GraphModelType;
	typedef typename GraphModelType::LabelMapperType LabelMapperType;
    
	typedef TypeTruncatedQuadratic3D TRWType;
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
    bool m_segmented, m_registered;
public:
	TRWS_SRSMRFSolverTruncQuadrat3D(GraphModelType * graphModel,
                                    double unaryRegWeight=1.0, 
                                    double pairwiseRegWeight=1.0, 
                                    double unarySegWeight=1.0, 
                                    double pairwiseSegWeight=1.0, 
                                    double pairwiseSegRegWeight=1.0,
                                    bool vverbose=false)
        :m_optimizer(TRWType::GlobalSize(2*LabelMapperType::nDisplacementSamples+1,2*LabelMapperType::nDisplacementSamples+1,2*LabelMapperType::nDisplacementSamples+1)),m_GraphModel(graphModel)
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
    TRWS_SRSMRFSolverTruncQuadrat3D()  :m_optimizer(TRWType::GlobalSize(2*LabelMapperType::nDisplacementSamples+1,2*LabelMapperType::nDisplacementSamples+1,2*LabelMapperType::nDisplacementSamples+1)){
        
    }
    ~TRWS_SRSMRFSolverTruncQuadrat3D()
    {
    }

 
    virtual void createGraph(){
        clock_t start = clock();
        {
            m_segmented=false; 
            m_registered=false;
            m_segmented=0;
            m_registered=0;
        }
        LOGV(1)<<"Computing potential functions and setting up MRF."<<std::endl;
        this->m_GraphModel->Init();
        clock_t endUnary = clock();
        double t1 = (float) ((double)(endUnary - start) / CLOCKS_PER_SEC);
        tUnary+=t1;       
        nNodes=this->m_GraphModel->nNodes();
        nEdges=this->m_GraphModel->nEdges();
        nRegNodes=this->m_GraphModel->nRegNodes();
        nSegNodes=this->m_GraphModel->nSegNodes();

        segNodes = vector<NodeType>(nSegNodes,NULL);
        regNodes = vector<NodeType>(nRegNodes,NULL);
   
        m_start=start;

        int edgeCount=0;
        nRegLabels=this->m_GraphModel->nRegLabels();
        nSegLabels=this->m_GraphModel->nSegLabels();

        //		traverse grid
        if ( (m_pairwiseSegmentationRegistrationWeight || m_unaryRegistrationWeight>0 || m_pairwiseRegistrationWeight>0) && nRegLabels){
            //RegUnaries
            m_registered=true;
            clock_t startUnary = clock();

            TRWType::REAL D1[nRegLabels];
            for (int d=0;d<nRegNodes;++d){
                for (int l1=0;l1<nRegLabels;++l1)
                    {
                        D1[l1]=m_unaryRegistrationWeight*this->m_GraphModel->getUnaryRegistrationPotential(d,l1);
                    }
                regNodes[d] = 
                    m_optimizer.AddNode(TRWType::LocalSize(), TRWType::NodeData(D1));
                // Pairwise potentials
            }
            clock_t endUnary = clock();
            for (int d=0;d<nRegNodes;++d){
            
                {//pure Registration
                    std::vector<int> neighbours= this->m_GraphModel->getForwardRegistrationNeighbours(d);
                    int nNeighbours=neighbours.size();
                    for (int i=0;i<nNeighbours;++i){
                        //LOG<<d<<" "<<regNodes[d]<<" "<<i<<" "<<neighbours[i]<<std::endl;
                        // edges[edgeCount]=
                        m_optimizer.AddEdge(regNodes[d], regNodes[neighbours[i]], TRWType::EdgeData(m_pairwiseRegistrationWeight,m_pairwiseRegistrationWeight,m_pairwiseRegistrationWeight,5));
                        edgeCount++;
                    }
                
                }
            }
            clock_t endPairwise = clock();
            double t = (float) ((double)(endUnary - startUnary) / CLOCKS_PER_SEC);
            tUnary+=t;
            t = (float) ((double)(endPairwise-endUnary ) / CLOCKS_PER_SEC);
            tPairwise+=t;
            
        }
        if ( (m_pairwiseSegmentationRegistrationWeight || m_unarySegmentationWeight>0 || m_pairwiseSegmentationWeight) && nSegLabels){
            m_segmented=true;
            //SegUnaries
            TRWType::REAL D2[nSegLabels];
            for (int d=0;d<nSegNodes;++d){
                for (int l1=0;l1<nSegLabels;++l1)
                    {
                        D2[l1]=m_unarySegmentationWeight*this->m_GraphModel->getUnarySegmentationPotential(d,l1);
                        //  LOG<<d<<" "<< D2[l1] <<std::endl;
                    }
                segNodes[d] = 
                    m_optimizer.AddNode(TRWType::LocalSize(), TRWType::NodeData(D2));
                
                //  LOG<<" reg and segreg pairwise pots" <<std::endl;
       
            }
            
            TRWType::REAL VsrsBack[nRegLabels*nSegLabels];
            for (int d=0;d<nSegNodes;++d){   
                TRWType::REAL Vseg[nSegLabels*nSegLabels];
                //pure Segmentation
                std::vector<int> neighbours= this->m_GraphModel->getForwardSegmentationNeighbours(d);
                int nNeighbours=neighbours.size();
                for (int i=0;i<nNeighbours;++i){
                
                    for (int l1=0;l1<nSegLabels;++l1){
                        for (int l2=0;l2<nSegLabels;++l2){
                            double lambda =m_pairwiseSegmentationWeight*this->m_GraphModel->getPairwiseSegmentationPotential(d,neighbours[i],l1,l2);
#if 0
                            double lambda2=m_pairwiseSegmentationWeight*this->m_GraphModel->getSegmentationWeight(d,neighbours[i]);

                            if (lambda2!=lambda){
                                LOG<<l1<<" "<<l2<<" "<<lambda2<<" "<<lambda<<endl;
                            }
#endif
                            Vseg[l1*nSegLabels+l2]=lambda;
                            //LOG<<l1<<" "<<l2<<" "<<                            Vseg[l1*nSegLabels+l2]<<endl;

                        }
                    }
                    //m_optimizer.AddEdge(segNodes[d], segNodes[neighbours[i]], TRWType::EdgeData(TRWType::GENERAL,Vseg));
                    edgeCount++;
                    
                }
                if (m_pairwiseSegmentationRegistrationWeight>0 && nRegLabels){
                    std::vector<int> segRegNeighbors=this->m_GraphModel->getSegRegNeighbors(d);
                    nNeighbours=segRegNeighbors.size();
                    for (int i=0;i<nNeighbours;++i){
                        
                        for (int l1=0;l1<nRegLabels;++l1){
                            for (int l2=0;l2<nSegLabels;++l2){
                                //forward
                                VsrsBack[l1+l2*nRegLabels]=m_pairwiseSegmentationRegistrationWeight*this->m_GraphModel->getPairwiseRegSegPotential(segRegNeighbors[i],d,l1,l2);
                            }
                        }
                        //m_optimizer.AddEdge(regNodes[segRegNeighbors[i]], segNodes[d]             , TRWType::EdgeData(TRWType::GENERAL,VsrsBack));
                        edgeCount++;
                    }
                }
                
            }
        }
        clock_t finish = clock();
        double t = (float) ((double)(finish - start) / CLOCKS_PER_SEC);
        LOGV(1)<<"Finished init after "<<t<<" seconds"<<std::endl;
        nEdges=edgeCount;

    }
    

    virtual void optimize(int maxIter=20){
        LOGV(1)<<"Total number of MRF edges: " <<nEdges<<endl;
        //m_optimizer.SetAutomaticOrdering();

        MRFEnergy<TRWType>::Options options;
        TRWType::REAL energy=-1, lowerBound=-1;
        options.m_iterMax = maxIter; // maximum number of iterations
        options.m_printMinIter=1;
        options.m_printIter=1;
        options.verbose=verbose;
        options.m_eps=-1;
        clock_t opt_start=clock();
        m_optimizer.Minimize_TRW_S(options, lowerBound, energy);
        clock_t finish = clock();
        tOpt+=((double)(finish-opt_start)/CLOCKS_PER_SEC);
        float t = (float) ((double)(finish - m_start) / CLOCKS_PER_SEC);
        LOG<<"Finished after "<<t<<" , resulting energy is "<<energy<<" with lower bound "<< lowerBound ;//std::endl;

       
    }

    virtual std::vector<int> getDeformationLabels(){
        std::vector<int> labels(nRegNodes,0);
        int fac=2*LabelMapperType::nDisplacementSamples+1;
        if (m_registered){
            for (int i=0;i<nRegNodes;++i){
                TRWType::Label l=m_optimizer.GetSolution(regNodes[i]);
                labels[i]=l.m_kx+l.m_ky*(fac)+l.m_kz*fac*fac;
            }
        }
        return labels;
    }
    virtual std::vector<int> getSegmentationLabels(){
        std::vector<int> labels(nSegNodes,0);
        if (m_segmented){
            for (int i=0;i<nSegNodes;++i){
                //labels[i]=m_optimizer.GetSolution(segNodes[i]);
                //labels[i]=this->m_GraphModel->getUnarySegmentationPotential(i,1)*nSegNodes;//m_optimizer.GetSolution(segNodes[i]);
            }
        }
        return labels;
    }

    void evalSolution(){
        double sumUReg=0,sumUSeg=0,sumPSeg=0,sumPSegReg=0;
        
        clock_t start = clock();
        m_start=start;
        if (nRegLabels){
            for (int d=0;d<nRegNodes;++d){
                sumUReg+=m_unaryRegistrationWeight*this->m_GraphModel->getUnaryRegistrationPotential(d,m_optimizer.GetSolution(regNodes[d]));
            }
        }
        if (nSegLabels){
            for (int d=0;d<nSegNodes;++d){
                sumUSeg+=m_unarySegmentationWeight*this->m_GraphModel->getUnarySegmentationPotential(d,m_optimizer.GetSolution(segNodes[d]));
                if (nRegLabels){
                    std::vector<int> neighbours= this->m_GraphModel->getForwardSegmentationNeighbours(d);
                    int nNeighbours=neighbours.size();
                    for (int i=0;i<nNeighbours;++i){
                        //sumPSeg+=m_pairwiseSegmentationWeight*this->m_GraphModel->getSegmentationWeight(d,neighbours[i])*(m_optimizer.GetSolution(segNodes[d])!=m_optimizer.GetSolution(segNodes[neighbours[i]]));
                    }
                    std::vector<int> segRegNeighbors=this->m_GraphModel->getSegRegNeighbor(d);
                    for (unsigned int n=0;n<segRegNeighbors.size();++n){
                        sumPSegReg+=m_pairwiseSegmentationRegistrationWeight*this->m_GraphModel->getPairwiseRegSegPotential(segRegNeighbors[n],d,m_optimizer.GetSolution(regNodes[segRegNeighbors[n]]),m_optimizer.GetSolution(segNodes[d]));
                    }
                }


            }   
        }
        LOG<<"RegU :\t\t"<<sumUReg<<endl
                 <<"SegU :\t\t"<<sumUSeg<<endl
                 <<"SegP :\t\t"<<sumPSeg<<endl
                 <<"SegRegP :\t"<<sumPSegReg<<endl;
        clock_t finish = clock();
        double t = (float) ((double)(finish - start) / CLOCKS_PER_SEC);
        LOGV(1)<<"Finished init after "<<t<<" seconds"<<std::endl;
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
