/*
 * TRW-S-Registration.h
 *
 *  Created on: Dec 3, 2010
 *      Author: gasst
 */

#ifndef GC_REGISTRATION_H_
#define GC_REGISTRATION_H_
#include "BaseMRF.h"
#include "bgraph.h"
template<class TGraphModel>
class GC_MRFSolver : public BaseMRFSolver<TGraphModel>{
public:
	typedef BaseMRFSolver<TGraphModel> Superclass;
	typedef typename Superclass::LabelType LabelType;
	typedef Graph::Real Real;
	typedef TGraphModel GraphModelType;
	typedef typename GraphModelType::LabelMapperType LabelMapperType;

	typedef BGraph<float,float,float> MRFType;
	typedef MRFType::node_id NodeType;
protected:
	MRFType* optimizer;
	NodeType* nodes;
	double m_unaryWeight,m_pairwiseWeight;
	bool secondPairwise;
	bool verbose;
public:
	GC_MRFSolver(GraphModelType * graphModel, double unaryWeight=1.0, double pairwiseWeight=1.0, bool secondPairwisePotential=false)
	:Superclass(graphModel),secondPairwise(secondPairwisePotential)
	{
		verbose=true;
		m_unaryWeight=unaryWeight;
		m_pairwiseWeight=pairwiseWeight;
		createGraph();
	}
	~GC_MRFSolver()
	{
		delete optimizer;

	}
	virtual void createGraph(){

		if (verbose) std::cout<<"starting graph init"<<std::endl;
		GraphModelType* graph=this->m_GraphModel;
		int nLabels=this->m_nLabels;
		int nNodes=this->m_nNodes;
		optimizer = new MRFType(nNodes,graph->nVertices());

		optimizer->add_node(nNodes);



		clock_t start = clock();
		//		traverse grid
		float D[nLabels];

		for (int d=0;d<nNodes;++d){
			//set up unary costs at current position
			for (int l1=0;l1<nLabels;++l1)
			{

				D[l1]=m_unaryWeight*graph->getUnaryPotential(d,l1);
//				std::cout<<d<<" "<<l1<<" "<<D[l1]<<" "<<nLabels<<std::endl;
			}
			optimizer->add_tweights(d,D[0],D[1]);
		}
		clock_t finish1 = clock();
		float t = (float) ((double)(finish1 - start) / CLOCKS_PER_SEC);
		if (verbose) std::cout<<"Finished unary potential initialisation after "<<t<<" seconds"<<std::endl;
		//
		int vertCount=0;
		for (int d=0;d<nNodes;++d){
			std::vector<int> neighbours= graph->getForwardNeighbours(d);
			int nNeighbours=neighbours.size();
			vertCount+=nNeighbours;
			for (int i=0;i<nNeighbours;++i){
				assert(neighbours[i]<nNodes);
                //                std::cout<<" edge " << m_pairwiseWeight*graph->getWeight(d,neighbours[i])<<" "<< m_pairwiseWeight*graph->getWeight(neighbours[i],d) << std::endl;
				optimizer -> add_edge(d,neighbours[i], m_pairwiseWeight*graph->getWeight(d,neighbours[i]), m_pairwiseWeight*graph->getWeight(neighbours[i],d));
			}
		}
		std::cout<<vertCount<<" "<<graph->nVertices()<<std::endl;
		clock_t finish = clock();
		t = (float) ((double)(finish - start) / CLOCKS_PER_SEC);
		if (verbose) std::cout<<"Finished init after "<<t<<" seconds"<<std::endl;

	}

	virtual void optimize(){

		clock_t start = clock();
		if (verbose) std::cout<<"starting maxFlow"<<std::endl;

		float flow = optimizer -> maxflow();
		clock_t finish = clock();
		float t = (float) ((double)(finish - start) / CLOCKS_PER_SEC);
		std::cout<<"Finished after "<<t<<" , resulting energy is "<<flow;//<< std::endl;

	}

	virtual LabelType getLabelAtIndex(int index){
		//		int labelIndex=l.m_kx+l.m_ky*labelSampling;
		int labelIndex=optimizer->what_segment(index) == MRFType::SOURCE;
		return LabelMapperType::getLabel(labelIndex);

	}
};

#endif /* TRW_S_REGISTRATION_H_ */
