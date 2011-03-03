/*
 * FAST-PD-Registration-mrf.h
 *
 *  Created on: Nov 30, 2010
 *      Author: gasst
 */

#ifndef FAST_PD_MRF_H_
#define FAST_PD_MRF_H_
#include "BaseMRF.h"
#include "Fast_PD3.h"

template<class TGraphModel>
class NewFastPDMRFSolver : public BaseMRFSolver<TGraphModel>{
public:

	typedef BaseMRFSolver<TGraphModel> Superclass;
	typedef typename Superclass::LabelType LabelType;
	typedef Graph::Real Real;
	typedef TGraphModel GraphModelType;
	typedef typename GraphModelType::LabelMapperType LabelMapperType;
protected:
	std::vector<int> pairs;
	Real * unaryPotentials;
	Real * pairwisePotentials;
	Real * pairwisePotentials2;
	bool verbose ;
	std::vector<Real>  edgeWeights;
	CV_Fast_PD * optimizer;
	double m_pairwiseWeight,m_unaryWeight;
	bool secondPairwise;
public:
	NewFastPDMRFSolver(GraphModelType * graphModel, double unaryWeight=1.0, double pairwiseWeight=1.0, bool secondPairwisePotential=false)
	:Superclass(graphModel),secondPairwise(secondPairwisePotential)
	{
		verbose=true;
		//		pairs is an array of the form [na nb nc nd...] where (na,nb),(nc,nd) are edges in the graph and na,... are indices of the nodes
		if (verbose) std::cout<<"allocating "<<this->m_nPairs<<" pairs"<<std::endl;
		pairs=std::vector<int>(this->m_nPairs*2);
		if (verbose) std::cout<<"allocating "<<this->m_nNodes*this->m_nLabels<<"unary potentials"<<std::endl;
		unaryPotentials=new Real[this->m_nNodes*this->m_nLabels];
		if (verbose) std::cout<<"allocating "<<this->m_nLabels*this->m_nLabels*this->m_nPairs<<" pairwise potentials"<<std::endl;
		//		pairwisePotentials= new Real[this->m_nLabels*this->m_nLabels];
		pairwisePotentials= new Real[this->m_nLabels*this->m_nLabels*this->m_nPairs];

		if (secondPairwise)
			pairwisePotentials2= new Real[this->m_nLabels*this->m_nLabels];
		if (verbose) std::cout<<"allocating "<<this->m_nPairs<<" edgeweights"<<std::endl;
		edgeWeights=std::vector<Real>(this->m_nPairs,1.0);//new Real[this->m_nPairs];//={1.0};
		m_unaryWeight=unaryWeight;
		m_pairwiseWeight=pairwiseWeight;
		if (verbose) std::cout<<this->m_nNodes<<" "<<this->m_nLabels<<" "<<std::endl;
		//memset( edgeWeights, 1.0,this->m_nPairs*sizeof(Real) );

		createGraph();
	}
	~NewFastPDMRFSolver(){
//		delete[] unaryPotentials;
//		delete[] pairwisePotentials;
//		delete[] pairwisePotentials2;
		delete optimizer;
	}
	/*
	 * initialize graph, compute potentials, set up neighbourhood structure
	 */
	virtual void createGraph(){
		if (verbose) std::cout<<"starting graph init"<<std::endl;
		GraphModelType* graph=this->m_GraphModel;
		int nLabels=this->m_nLabels;
		int runningIndex=0;
		//		traverse grid
		for (int d=0;d<this->m_nNodes;++d)
		{
			// get current indices both integer, in the grid plane and in the image plane
			// get forward neighbors of current grid point, both grid index and image plane index
			std::vector<int> neighbours=graph->getForwardNeighbours(d);
			int nNeighbours=neighbours.size();
//			std::cout<<d<<" "<<nNeighbours<<std::endl;
			for (int i=0;i<nNeighbours;++i){
				int pairIndex=runningIndex+i*2;
				pairs[pairIndex]=d;
				pairs[pairIndex+1]=neighbours[i];
				edgeWeights[(pairIndex)/2]=graph->getWeight(d,neighbours[i]);

				for (int l1=0;l1<nLabels;++l1){
					for (int l2=0;l2<nLabels;++l2){
//						std::cout<<pairIndex/2<<" "<<l1<< " "<<l2<<" "<<l1*nLabels+l2 + pairIndex/2*nLabels*nLabels<<std::endl;
						pairwisePotentials[l1*nLabels+l2 + pairIndex/2*nLabels*nLabels]=m_pairwiseWeight*graph->getPairwisePotential(d,neighbours[i],l1,l2);
						if (secondPairwise){
							pairwisePotentials2[l1*nLabels+l2 + pairIndex/2*nLabels*nLabels]=m_pairwiseWeight*graph->getPairwisePotential2(l1,l2);
						}
					}
				}
			}
			runningIndex+=nNeighbours*2;
			//set up unary costs at current position
			for (int l1=0;l1<nLabels;++l1){
				unaryPotentials[l1*this->m_nNodes+d]=m_unaryWeight*graph->getUnaryPotential(d,l1);
			}

		}

		if (verbose) std::cout<<"initialised basic graph structure and unary potentials"<<std::endl;
		//		traverse labels

		//create optimizer object
		if (verbose) std::cout<<"initialising fastPD with "<<this->m_nNodes<<" nodes, "<< this->m_nLabels<<" labels, "<<this->m_nPairs<<" pairs"<<std::endl;
		if (secondPairwise){
			optimizer= new CV_Fast_PD(this->m_nNodes,this->m_nLabels,unaryPotentials,this->m_nPairs,&pairs[0],pairwisePotentials, pairwisePotentials2,20,&edgeWeights[0],int(this->m_nLabels/4));
		}else{
			optimizer= new CV_Fast_PD(this->m_nNodes,this->m_nLabels,unaryPotentials,this->m_nPairs,&pairs[0],pairwisePotentials, pairwisePotentials,20,&edgeWeights[0],int(this->m_nLabels/4));
		}
		//		this->m_unaryPotentialFunction->freeMemory();
	}
	virtual void optimize(){
		optimizer->run();
	}


	virtual LabelType getLabelAtIndex(int index){

		LabelType l=LabelMapperType::getLabel(optimizer->_pinfo[index].label);
		return l;
	}
};

#endif /* FAST_PD_REGISTRATION_MRF_H_ */
