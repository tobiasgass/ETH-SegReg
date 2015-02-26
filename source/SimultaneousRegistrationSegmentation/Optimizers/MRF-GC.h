#include "Log.h"
/*
 * TRW-S-Registration.h
 *
 *  Created on: Dec 3, 2010
 *      Author: gasst
 */

#ifndef GC_REGISTRATION_H_
#define GC_REGISTRATION_H_
#include "graph.h"
#include "BaseMRF.h"

namespace SRS{
template<class TGraphModel>
class GC_MRFSolver {
public:
	
	typedef TGraphModel GraphModelType;
	typedef typename GraphModelType::LabelMapperType LabelMapperType;

	//typedef BGraph<float,float,float> MRFType;
	//typedef Graph<short ,short ,long long> MRFType;

    ///will not link against GCO if different from GCO_ENERGYTYPE defines
    ///can probably still be linked against original maxflo
    typedef Graph<float ,float ,double> MRFType;
	typedef MRFType::node_id NodeType;
protected:
	MRFType* optimizer;
	NodeType* nodes;
	double m_unaryWeight,m_pairwiseWeight;
	bool secondPairwise;
	bool verbose;
    GraphModelType * m_graphModel;
    int nNodes;
    double m_multiplier;
public:
	GC_MRFSolver(GraphModelType * graphModel, double unaryWeight=1.0, double pairwiseWeight=1.0, bool verb=false)

	{
        m_graphModel= graphModel;
		verbose=verbose;
		m_unaryWeight=unaryWeight;
		m_pairwiseWeight=pairwiseWeight;
        m_multiplier=5000;
		createGraph();
	}
	~GC_MRFSolver()
	{
		delete optimizer;

	}
	virtual void createGraph(){

		LOGV(1)<<"starting graph init"<<std::endl;
		GraphModelType* graph=this->m_graphModel;
        nNodes=graph->nNodes();
        
        int nLabels=graph->nLabels();
		optimizer = new MRFType(nNodes,graph->nEdges());

		optimizer->add_node(nNodes);



		clock_t start = clock();
		//		traverse grid
		float D[nLabels];

		for (int d=0;d<nNodes;++d){
			//set up unary costs at current position
			for (int l1=0;l1<nLabels;++l1)
			{
				D[l1]=m_multiplier*m_unaryWeight*graph->getUnaryPotential(d,l1);
                LOGV(9)<<d<<" "<<l1<<" "<<D[l1]<<" "<<nLabels<<std::endl;
			}
			optimizer->add_tweights(d,D[0],D[1]);
		}
		clock_t finish1 = clock();
		float t = (float) ((double)(finish1 - start) / CLOCKS_PER_SEC);
		LOGV(1)<<"Finished unary potential initialisation after "<<t<<" seconds"<<std::endl;
		//
		int vertCount=0;
		for (int d=0;d<nNodes;++d){
			std::vector<int> neighbours= graph->getForwardNeighbours(d);
			int nNeighbours=neighbours.size();
			vertCount+=nNeighbours;
			for (int i=0;i<nNeighbours;++i){
				assert(neighbours[i]<nNodes);
                //                LOG<<" edge " << m_pairwiseWeight*graph->getWeight(d,neighbours[i])<<" "<< m_pairwiseWeight*graph->getWeight(neighbours[i],d) << std::endl;
                double lambda1=m_multiplier*m_pairwiseWeight*graph->getPairwisePotential(d,neighbours[i],1,0);
                double lambda2=m_multiplier*m_pairwiseWeight*graph->getPairwisePotential(d,neighbours[i],0,1);
                LOGV(9)<<VAR(lambda1)<<" "<<VAR(lambda2)<<endl;
				optimizer -> add_edge(d,neighbours[i],lambda1,lambda2);
			}
		}
		LOGV(2)<<vertCount<<" "<<graph->nEdges()<<std::endl;
		clock_t finish = clock();
		t = (float) ((double)(finish - start) / CLOCKS_PER_SEC);
		LOGV(1)<<"Finished init after "<<t<<" seconds"<<std::endl;

	}

	virtual void optimize(int optiter){

		clock_t start = clock();
		LOGV(1)<<"starting maxFlow"<<std::endl;

		float flow = optimizer -> maxflow();
		clock_t finish = clock();
		float t = (float) ((double)(finish - start) / CLOCKS_PER_SEC);
		LOG<<"Finished after "<<t<<" , resulting energy is "<<flow;//<< std::endl;

	}
    virtual std::vector<int> getLabels(){
        //GraphModelType* graph=this->m_graphModel;
        std::vector<int> labels(nNodes);
        for (int i=0;i<nNodes;++i){
            labels[i]=optimizer->what_segment(i) == MRFType::SOURCE;
            //labels[i]=graph->getUnaryPotential(i,1)*graph->nNodes();//optimizer->what_segment(i) == MRFType::SOURCE;
        }
        return labels;
    }
};

template<class TGraphModel>
class GC_MRFSolverSeg {
public:
	
	typedef TGraphModel GraphModelType;
	typedef typename GraphModelType::LabelMapperType LabelMapperType;

	//typedef BGraph<float,float,float> MRFType;
	//typedef Graph<short,short,long long> MRFType;
    typedef Graph<float ,float ,double> MRFType;
	typedef MRFType::node_id NodeType;
protected:
	MRFType* optimizer;
	NodeType* nodes;
	double m_unaryWeight,m_pairwiseWeight;
	bool secondPairwise;
	bool verbose;
    GraphModelType * m_graphModel;
    int nNodes;
    double m_multiplier;
public:
	GC_MRFSolverSeg(GraphModelType * graphModel, double unaryWeight=1.0, double pairwiseWeight=1.0, bool verb=false)

	{
        m_graphModel= graphModel;
		verbose=verb;
		m_unaryWeight=unaryWeight;
        m_multiplier=1000;
		m_pairwiseWeight=pairwiseWeight;
		//createGraph();
	}
	~GC_MRFSolverSeg()
	{
		delete optimizer;

	}
	virtual void createGraph(){

		LOGV(1)<<"starting graph init"<<std::endl;
		GraphModelType* graph=this->m_graphModel;
        nNodes=graph->nSegNodes();
        
        int nLabels=graph->nSegLabels();
        LOGV(1)<<VAR(nNodes)<<" "<<VAR(nLabels)<<" "<<VAR(graph->nSegEdges())<<endl;
		optimizer = new MRFType(nNodes,graph->nSegEdges());

		optimizer->add_node(nNodes);



		clock_t start = clock();
		//		traverse grid
		float D[nLabels];

		for (int d=0;d<nNodes;++d){
			//set up unary costs at current position
			for (int l1=0;l1<nLabels;++l1)
			{
				D[l1]=m_multiplier*m_unaryWeight*graph->getUnarySegmentationPotential(d,l1);
                LOGV(9)<<d<<" "<<l1<<" "<<D[l1]<<" "<<nLabels<<std::endl;
			}
			optimizer->add_tweights(d,D[0],D[1]);
		}
		clock_t finish1 = clock();
		float t = (float) ((double)(finish1 - start) / CLOCKS_PER_SEC);
		LOGV(1)<<"Finished unary potential initialisation after "<<t<<" seconds"<<std::endl;
		//
		int vertCount=0;
		for (int d=0;d<nNodes;++d){
			std::vector<int> neighbours= graph->getForwardSegmentationNeighbours(d);
			int nNeighbours=neighbours.size();
			vertCount+=nNeighbours;
			for (int i=0;i<nNeighbours;++i){
				assert(neighbours[i]<nNodes);
                //                LOG<<" edge " << m_pairwiseWeight*graph->getWeight(d,neighbours[i])<<" "<< m_pairwiseWeight*graph->getWeight(neighbours[i],d) << std::endl;
                double lambda1=m_multiplier*m_pairwiseWeight*graph->getPairwiseSegmentationPotential(d,neighbours[i],1,0);
                double lambda2=m_multiplier*m_pairwiseWeight*graph->getPairwiseSegmentationPotential(d,neighbours[i],0,1);
                LOGV(9)<<VAR(lambda1)<<" "<<VAR(lambda2)<<endl;
                optimizer -> add_edge(d,neighbours[i],lambda1,lambda2);
			}
		}
		LOGV(2)<<vertCount<<" "<<graph->nEdges()<<std::endl;
		clock_t finish = clock();
		t = (float) ((double)(finish - start) / CLOCKS_PER_SEC);
		LOGV(1)<<"Finished init after "<<t<<" seconds"<<std::endl;

	}

	virtual void optimize(int optiter){

		clock_t start = clock();
		LOGV(1)<<"starting maxFlow"<<std::endl;

		float flow = optimizer -> maxflow();
		clock_t finish = clock();
		float t = (float) ((double)(finish - start) / CLOCKS_PER_SEC);
		LOG<<"Finished after "<<t<<" , resulting energy is "<<flow;//<< std::endl;

	}
    virtual std::vector<int> getLabels(){
        //GraphModelType* graph=this->m_graphModel;
        std::vector<int> labels(nNodes);
        for (int i=0;i<nNodes;++i){
            labels[i]=optimizer->what_segment(i) == MRFType::SOURCE;
            //labels[i]=graph->getUnaryPotential(i,1)*graph->nNodes();//optimizer->what_segment(i) == MRFType::SOURCE;
        }
        return labels;
    }
};

}//namespace
#endif /* TRW_S_REGISTRATION_H_ */
