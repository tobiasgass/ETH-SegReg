#pragma once
#include "BaseMRF.h"
#include "Log.h"
#include "GCoptimization.h"
#include <vector>
#include <map>
//#include <google/heap-profiler.h>
#include <limits.h>
#include <time.h>
#include <vector>



namespace SRS{


/** \brief
   * Wrapper for Olga Vekslers Multilabel graph cut library
   */
template<class TGraphModel>
class GCO_SRSMRFSolver :public BaseMRFSolver<TGraphModel>{
public:

    //typedef short EnergyType;
    typedef float EnergyType;

    typedef GCO_SRSMRFSolver<TGraphModel> Self;
    typedef TGraphModel GraphModelType;
	typedef typename GraphModelType::Pointer GraphModelPointerType;

    static const int D = GraphModelType::ImageType::ImageDimension;
    typedef GCoptimizationGeneralGraph MRFType;

    typedef EnergyType (* SmoothFn)(int s1,int s2,int l1, int l2);
protected:
    MRFType * m_optimizer;
    double m_unaryRegistrationWeight,m_unarySegmentationWeight;
	static double m_pairwiseSegmentationRegistrationWeight,m_pairwiseRegistrationWeight,m_pairwiseSegmentationWeight;
	int verbose;
    
    int nNodes, nRegNodes, nSegNodes, nEdges;
    clock_t m_start;
    int nRegLabels;
    int nSegLabels;
    bool m_segment, m_register,m_coherence;
    double m_lastLowerBound;
    std::vector<int> m_labelOrder;
    int m_zeroDisplacementLabel;
    bool m_deleteRegNeighb;
    

    
    //ugly  static members because of GCO
    static std::vector<std::vector<std::vector<std::map<int,float> > > > (*regPairwise);
    static std::vector<std::vector<std::vector<float > > > *srsPairwise;
    static std::vector<std::vector<std::vector<std::vector<float> > > > *segPairwise;
    static int S0,S1;
    static int GLOBALnRegNodes,GLOBALnSegNodes,GLOBALnSegLabels,GLOBALnRegLabels;
    static GraphModelPointerType m_GraphModel;
    static bool m_cachePotentials;
   
    static int getRelativeNodeIndex(int idx1,int idx2, int S0, int S1=-1 ){
        //returns the neighbor direction. 
        //idx1= d0 + d1*S0 + d2*S0*S1 
        //idx2 is either
        //1 d0+1 +  d1   *S0 +  d2   *S0*S1
        //2 d0   + (d1+1)*S0 +  d2   *S0*S1
        //3 d0   +  d1   *S0 + (d2+1)*S0*S1
        int diff =  idx2-idx1;
        if (diff == 1	)		return 0;
        if (diff == S0	)	return 1;
        if (diff ==	S0*S1)	return 2;	
        std::cerr<<"Error! idx difference doesn make sense : "<<VAR(diff)<<std::endl;
        return -1;
    }
   

    //neighbor structure for GCO
    int * m_numberOfNeighborsofEachNode;
    int ** m_neighbourArray,*m_segNeighbors,*m_regNeighbors; 
    EnergyType ** m_weights,*m_segWeights,*m_regWeights;

public:
    static EnergyType GLOBALsmoothFunction(int node1, int node2, int label1, int label2){
        float pot=-1;
        if (node1>node2){
            int tmp=node1;        node1=node2; node2=tmp;
            tmp=label1; label1=label2; label2=tmp;
        }

        if (m_cachePotentials){
            if (node1>=GLOBALnRegNodes && node2>=GLOBALnRegNodes){
                //segmentation pairwise!
                if ( (label1<GLOBALnRegLabels) || (label2<GLOBALnRegLabels) ){
                    pot=0.0;
                }else{
                    pot=(*segPairwise)[label1-GLOBALnRegLabels][label2-GLOBALnRegLabels][node1-GLOBALnRegNodes][getRelativeNodeIndex(node1-GLOBALnRegNodes,node2-GLOBALnRegNodes,S0,S1)];
                }
            }else if (node1<GLOBALnRegNodes && node2<GLOBALnRegNodes){
                //registration pairwise!
                if (label2>=GLOBALnRegLabels || label1>=GLOBALnRegLabels){
                    pot=0.0;
                }else{
                    pot=(*regPairwise)[label1][label2][node1][node2];

                }
            }else{
                //srs pairwise
                if (label2<GLOBALnRegLabels || label1>=GLOBALnRegLabels){
                    //impossible labelling, either regnode getting assigned a seglabel, or vice versa
                    pot=100000;
                }else {
                    pot=(*srsPairwise)[label2-GLOBALnRegLabels][label1][node2-GLOBALnRegNodes];
                }
            }

        }else{
            if (node1>=GLOBALnRegNodes && node2>=GLOBALnRegNodes){
                //segmentation pairwise!
                if ( (label1<GLOBALnRegLabels) || (label2<GLOBALnRegLabels) ){
                    pot=0.0;
                }else{

                    pot = m_pairwiseSegmentationWeight*m_GraphModel->getPairwiseSegmentationPotential(node1-GLOBALnRegNodes,node2-GLOBALnRegNodes,label1-GLOBALnRegLabels,label2-GLOBALnRegLabels);
                }
            }else if (node1<GLOBALnRegNodes && node2<GLOBALnRegNodes){
                //registration pairwise!
                if (label2>=GLOBALnRegLabels || label1>=GLOBALnRegLabels){
                    pot=0.0;
                }else{

                    pot=m_pairwiseRegistrationWeight*m_GraphModel->getPairwiseRegistrationPotential(node1,node2,label1,label2);
                  
                }
            }else{
                //srs pairwise
                if (label2<GLOBALnRegLabels || label1>=GLOBALnRegLabels){
                    //impossible labelling, either regnode getting assigned a seglabel, or vice versa
                    pot=100000;
                }else {
                    pot=m_pairwiseSegmentationRegistrationWeight*m_GraphModel->getPairwiseRegSegPotential(node2-GLOBALnRegNodes,label1,label2-GLOBALnRegLabels);
                
                }
            }
        }
        //LOGV(10)<<VAR(EnergyType(MULTIPLIER*pot))<<endl;
        return EnergyType(pot);
    }

public:
    GCO_SRSMRFSolver(GraphModelPointerType  graphModel,
                     double unaryRegWeight=1.0, 
                     double pairwiseRegWeight=1.0, 
                     double unarySegWeight=1.0, 
                     double pairwiseSegWeight=1.0, 
                     double pairwiseSegRegWeight=1.0,
                     int vverbose=false)
        
	{
        m_GraphModel=graphModel;
		verbose=vverbose;
		m_unarySegmentationWeight=unarySegWeight;
		m_pairwiseSegmentationWeight=pairwiseSegWeight;
        m_unaryRegistrationWeight=unaryRegWeight;
        m_pairwiseRegistrationWeight=pairwiseRegWeight;
        m_pairwiseSegmentationRegistrationWeight=pairwiseSegRegWeight;
        //createGraph();
        //init();
        m_optimizer=NULL;
        regPairwise=NULL;
        segPairwise=NULL;
        srsPairwise=NULL;
        m_labelOrder=std::vector<int>(this->m_GraphModel->nRegLabels());
        //order registration labels such that they start with zero displacement
        m_zeroDisplacementLabel=this->m_GraphModel->getLabelMapper()->getZeroDisplacementIndex();
        m_labelOrder[0]=m_zeroDisplacementLabel;
        for (int l=0;l<(this->m_GraphModel->nRegLabels());++l){
            if (l < m_labelOrder[0])
                m_labelOrder[l+1]=l;
            else if (l>m_labelOrder[0])
                m_labelOrder[l]=l;
        }
        srand ( time(NULL) );
        m_cachePotentials=false;
        m_deleteRegNeighb=false;
      
    }
    GCO_SRSMRFSolver()  {
        
    }
	~GCO_SRSMRFSolver()
    { 

        LOGV(1)<<"Deleting GCO_MRF Sovler " << std::endl;
        if (m_register) delete regPairwise;
        if (m_segment) delete segPairwise;
        if (m_coherence) delete srsPairwise;

        if (m_numberOfNeighborsofEachNode){
            delete [] m_numberOfNeighborsofEachNode;
#ifdef ALLOCINDIVIDUAL
            for (int i = 0;i< GLOBALnRegNodes+GLOBALnSegNodes; ++i){
                if ( m_neighbourArray[i]!=NULL ) delete [] m_neighbourArray[i];
                if ( m_weights[i] !=NULL)delete [] m_weights[i];
            }
#else
            if (m_register){
                if (m_deleteRegNeighb){
                    for (int i = 0;i< GLOBALnRegNodes; ++i){
                        delete [] m_neighbourArray[i];
                        delete [] m_weights[i];
                    }
                }
                else{
                    delete [] m_regNeighbors;
                    delete [] m_regWeights;

                }
            }
            if (m_segment){
                delete [] m_segNeighbors;
                delete [] m_segWeights;
            }

#endif
            delete [] m_neighbourArray;
            delete [] m_weights;
        }
        delete m_optimizer;
        
    }

    virtual void setPotentialCaching(bool enableCaching){m_cachePotentials=enableCaching;}

    virtual void createGraph(){
        clock_t start = clock();
        {
            m_segment=false; 
            m_register=false;
            m_segment=0;
            m_register=0;
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

        m_start=start;

        int edgeCount=0;
        
        nRegLabels=this->m_GraphModel->nRegLabels();
        nSegLabels=this->m_GraphModel->nSegLabels();
        m_register=((m_pairwiseSegmentationRegistrationWeight>0 || m_unaryRegistrationWeight>0 || m_pairwiseRegistrationWeight>0) && nRegLabels>1);
        m_segment=((m_pairwiseSegmentationRegistrationWeight>0 || m_unarySegmentationWeight>0 || m_pairwiseSegmentationWeight)  && nSegLabels>1);
        m_coherence=m_pairwiseSegmentationRegistrationWeight>0;
        GLOBALnRegNodes= m_register*nRegNodes;
        GLOBALnSegNodes= m_segment*nSegNodes;
        GLOBALnRegLabels=m_register*nRegLabels;
        GLOBALnSegLabels=m_segment*nSegLabels;
        LOGV(5)<<VAR(GLOBALnRegNodes)<<" "<<VAR(GLOBALnRegLabels)<<" "<<VAR(GLOBALnSegNodes)<<" "<<VAR(GLOBALnSegLabels)<<std::endl;
        
        if (m_optimizer) delete m_optimizer;
        m_optimizer= new MRFType(GLOBALnSegNodes+GLOBALnRegNodes,GLOBALnRegLabels+GLOBALnSegLabels);
		
        //set global size variables :(
        {
            S0=this->m_GraphModel->getImageSize()[0];
            S1=this->m_GraphModel->getImageSize()[1];
        }

        //allocate neighbor structs
        m_numberOfNeighborsofEachNode = new int[GLOBALnRegNodes+GLOBALnSegNodes];
        memset(m_numberOfNeighborsofEachNode,0,(GLOBALnRegNodes+GLOBALnSegNodes)*sizeof(int));
#ifdef ALLOCINDIVIDUAL
        m_neighbourArray = new int *[GLOBALnRegNodes+GLOBALnSegNodes];
        memset(m_neighbourArray,NULL,(GLOBALnRegNodes+GLOBALnSegNodes)*(sizeof(int*)));
        m_weights= new EnergyType *[GLOBALnRegNodes+GLOBALnSegNodes];
        memset(m_weights,NULL,(GLOBALnRegNodes+GLOBALnSegNodes)*(sizeof(EnergyType*)));
#else
        m_neighbourArray = new int *[GLOBALnRegNodes+GLOBALnSegNodes];
        m_weights= new EnergyType *[GLOBALnRegNodes+GLOBALnSegNodes];
        
        if (m_register){
            int nNeighbors=2*D+m_coherence*this->m_GraphModel->getMaxRegSegNeighbors();
            LOGV(6)<<VAR(nNeighbors)<<" max neighbors per registration node" <<std::endl;
            if (nNeighbors<10){
                m_regNeighbors = new int[GLOBALnRegNodes*nNeighbors];      
                memset(m_regNeighbors,0,GLOBALnRegNodes*nNeighbors*(sizeof(int)));
                m_regWeights = new EnergyType[GLOBALnRegNodes*nNeighbors];
                memset(m_regWeights,1,GLOBALnRegNodes*nNeighbors*(sizeof(EnergyType)));
                
                for (int i=0;i<GLOBALnRegNodes;++i){
                    m_neighbourArray[i]=&m_regNeighbors[i*nNeighbors];
                    m_weights[i]=&m_regWeights[i*nNeighbors];
                }
                LOGV(6)<<"memory for reg neighbors primary structure : "<<1.0*GLOBALnRegNodes*nNeighbors*sizeof(int*)/1024/1024 <<"MB"<<std::endl;
            }else{
                LOGV(6)<<"allocating memory for registration node adjacency matrix individually.."<<std::endl;
                for (int i=0;i<GLOBALnRegNodes;++i){
                    int nLocalNeighbors=2*D+this->m_GraphModel->getRegSegNeighbors(i).size();
                    m_neighbourArray[i]=new int[nLocalNeighbors];
                    m_weights[i]=new EnergyType[nLocalNeighbors];
                }
                m_deleteRegNeighb=true;
            }

        }
        if (m_segment){
            int nNeighbors=2*D+m_coherence*1;
            m_segNeighbors = new int[GLOBALnSegNodes*nNeighbors];
            m_segWeights = new EnergyType[GLOBALnSegNodes*nNeighbors];
            memset(m_segNeighbors,0,GLOBALnSegNodes*nNeighbors*(sizeof(int)));
            memset(m_segWeights,1,GLOBALnSegNodes*nNeighbors*(sizeof(EnergyType)));

            for (int i=0;i<GLOBALnSegNodes;++i){
                m_neighbourArray[i+GLOBALnRegNodes]=&m_segNeighbors[i*nNeighbors];
                m_weights[i+GLOBALnRegNodes]=&m_segWeights[i*nNeighbors];
            }
            LOGV(6)<<"memory for seg neighbors primary structure : "<<1.0*GLOBALnSegNodes*nNeighbors*sizeof(int*)/1024/1024 <<"MB"<<std::endl;
        }

#endif

        logSetStage("Potential Functions");
        //		traverse grid
        if ( m_register){
            //RegUnaries
            clock_t startUnary = clock();
            
            //now compute&set all potentials
            if (m_unaryRegistrationWeight>0){

                //#pragma omp parallel for 
                //theoretically, this computation could be parallelized
                //however, the computation/caching of the registration potentials is not thread safe at the moment
                for (int l1=0;l1<nRegLabels;++l1)
                    {
                        int regLabel=m_labelOrder[l1];
                        std::vector<GCoptimization::SparseDataCost> costs(nRegNodes);
                        this->m_GraphModel->cacheRegistrationPotentials(regLabel);
                        for (int d=0;d<nRegNodes;++d){
                            costs[d].site=d;
                            costs[d].cost=m_unaryRegistrationWeight*this->m_GraphModel->getUnaryRegistrationPotential(d,regLabel);
                            if (m_coherence && !m_segment){
                                //pretty inefficient as the reg neighbors are recomputed #registrationLabels times for each registration node.
                                std::vector<int> regSegNeighbors=this->m_GraphModel->getRegSegNeighbors(d);
                                int nNeighbours=regSegNeighbors.size();
                                if (nNeighbours==0) {LOG<<"ERROR: node "<<d<<" seems to have no neighbors."<<std::endl;}
                                for (int i=0;i<nNeighbours;++i){
                                    double coherencePot=m_pairwiseSegmentationRegistrationWeight*this->m_GraphModel->getPairwiseRegSegPotential(d,regSegNeighbors[i],regLabel,0);
                                    LOGV(8)<<VAR(d)<<" "<<VAR(i)<<" "<<VAR(coherencePot)<<" "<<VAR(m_pairwiseSegmentationRegistrationWeight)<<std::endl;
                                    costs[d].cost+=coherencePot;
                                }
                            }
                        }
                        m_optimizer->setDataCost(regLabel,&costs[0],nRegNodes);
                    }
            }

         
            clock_t endUnary = clock();
            double t = (float) ((double)(endUnary - startUnary) / CLOCKS_PER_SEC);
            LOGV(1)<<"Registration Unaries took "<<t<<" seconds."<<std::endl;
            tUnary+=t;
            // Pairwise potentials
            if (m_cachePotentials)
                regPairwise= new std::vector<std::vector<std::vector<std::map<int,float> > > > (nRegLabels,std::vector<std::vector<std::map<int,float> > >(nRegLabels,std::vector<std::map<int,float> > (nRegNodes) ) );
            
            for (int d=0;d<nRegNodes;++d){
                m_optimizer->setLabel(d,m_zeroDisplacementLabel);

                {//pure Registration
                    std::vector<int> neighbours= this->m_GraphModel->getForwardRegistrationNeighbours(d);
                    int nNeighbours=neighbours.size();
                    for (int i=0;i<nNeighbours;++i){
                        //LOG<<d<<" "<<regNodes[d]<<" "<<i<<" "<<neighbours[i]<<std::endl;
                        //m_optimizer->setNeighbors(d,neighbours[i],1);
                        addNeighbor(d,neighbours[i],m_numberOfNeighborsofEachNode,m_neighbourArray,m_weights);
                        if (m_cachePotentials){
                            for (int l1=0;l1<nRegLabels;++l1){
                                for (int l2=0;l2<nRegLabels;++l2){                                
                                    if (m_pairwiseRegistrationWeight>0)
                                        (*regPairwise)[l1][l2][d][neighbours[i]] = m_pairwiseRegistrationWeight*this->m_GraphModel->getPairwiseRegistrationPotential(d,neighbours[i],l1,l2);
                                    else
                                        (*regPairwise)[l1][l2][d][neighbours[i]] = 0.0;
                                }
                            }
                        }

                        edgeCount++;
                    }
                
                }
            }
            clock_t endPairwise = clock();
         
            t = (float) ((double)(endPairwise-endUnary ) / CLOCKS_PER_SEC);
            LOGV(1)<<"Registration pairwise took "<<t<<" seconds."<<std::endl;
            LOGV(1)<<"Approximate size of reg pairwise: "<<1.0/(1024*1024)*nRegNodes*nRegLabels*nRegLabels*sizeof(double)*m_cachePotentials<<" mb."<<std::endl;

            tPairwise+=t;
        }
        if (m_segment){
            //SegUnaries
            clock_t startUnary = clock();
            for (int l1=0;l1<nSegLabels;++l1)
                {

                    //LOGV(4)<<"Allocating seg unaries for label "<<l1<<", using "<<1.0*nSegNodes*sizeof( GCoptimization::SparseDataCost ) /(1024*1024)<<" mb memory"<<std::std::endl;
                    std::vector<GCoptimization::SparseDataCost> costas(nSegNodes);
                    int c=0;
                    for (int d=0;d<nSegNodes;++d){
                        double unarySegCost=this->m_GraphModel->getUnarySegmentationPotential(d,l1);
                        if ( unarySegCost<10000){
                            costas[c].cost=m_unarySegmentationWeight*unarySegCost;
                            LOGV(10)<<"node "<<d<<"; seg unary label: "<<l1<<" "<<m_unarySegmentationWeight*this->m_GraphModel->getUnarySegmentationPotential(d,l1)<<std::endl;
                            costas[c].site=d+GLOBALnRegNodes;
                            if (m_coherence && !m_register){
                                double coherenceCost=m_pairwiseSegmentationRegistrationWeight*this->m_GraphModel->getPairwiseRegSegPotential(d,0,l1);
                                LOGV(8)<<VAR(d)<<" "<<VAR(coherenceCost)<<" "<<VAR(m_pairwiseSegmentationRegistrationWeight)<<std::endl;
                                costas[c].cost+=coherenceCost;
                            }
                            ++c;
                        }
                    }
                    costas.resize(c);
                    LOGV(2)<<"Number of nodes with segmentation label "<<l1<<": :"<<c<<std::endl;
                    m_optimizer->setDataCost(l1+GLOBALnRegLabels,&costas[0],c);
                }
          
            clock_t endUnary = clock();
            double t = (float) ((double)(endUnary - startUnary) / CLOCKS_PER_SEC);
            LOGV(1)<<"Segmentation Unaries took "<<t<<" seconds."<<std::endl;
            LOGV(1)<<"Approximate size of seg unaries: "<<1.0/(1024*1024)*nSegNodes*nSegLabels*sizeof(double)<<" mb."<<std::endl;

            int nSegEdges=0,nSegRegEdges=0;
            //Segmentation smoothness cache
            if (m_cachePotentials){
                segPairwise= new std::vector<std::vector<std::vector<std::vector<float> > > > (GLOBALnSegLabels,std::vector<std::vector<std::vector<float> > >(GLOBALnSegLabels,std::vector< std::vector<float> > (GLOBALnSegNodes,std::vector<float> (D)) ) );
                srsPairwise= new std::vector<std::vector<std::vector<float > > > (GLOBALnSegLabels,std::vector<std::vector<float > >(GLOBALnRegLabels,std::vector<float>(GLOBALnSegNodes) ) );
            }

            
         
            for (int d=0;d<nSegNodes;++d){   
                int initLabel= this->m_GraphModel->GetTargetSegmentationAtIdx(d);
                m_optimizer->setLabel(d+GLOBALnRegNodes,initLabel+GLOBALnRegLabels);
                //pure Segmentation
                std::vector<int> neighbours= this->m_GraphModel->getForwardSegmentationNeighbours(d);
                int nNeighbours=neighbours.size();
                for (int i=0;i<nNeighbours;++i){
                    nSegEdges++;
                    //m_optimizer->setNeighbors(d+GLOBALnRegNodes,neighbours[i]+GLOBALnRegNodes,1);
                    addNeighbor(d+GLOBALnRegNodes,neighbours[i]+GLOBALnRegNodes,m_numberOfNeighborsofEachNode,m_neighbourArray,m_weights);

                    edgeCount++;
                    if (m_cachePotentials){
                        for (int l1=0;l1<nSegLabels;++l1){
                            for (int l2=0;l2<nSegLabels;++l2){
                                LOGV(25)<<VAR(d)<<" "<<VAR(l1)<<" "<<VAR(neighbours[i])<<" "<<l2<<std::endl;
                                if (m_pairwiseSegmentationWeight>0){
                                    (*segPairwise)[l1][l2][d][i] = m_pairwiseSegmentationWeight*this->m_GraphModel->getPairwiseSegmentationPotential(d,neighbours[i],l1,l2);
                                }else{
                                    (*segPairwise)[l1][l2][d][i] = 0.0;
                                }
                            }
                        }
                    }
                    
                }
                if (m_register && m_coherence){
                    std::vector<int> segRegNeighbors=this->m_GraphModel->getSegRegNeighbors(d);
                    nNeighbours=segRegNeighbors.size();
                    if (nNeighbours==0) {LOG<<"ERROR: node "<<d<<" seems to have no neighbors."<<std::endl;}

                    for (int i=0;i<nNeighbours;++i){
                        //m_optimizer->setNeighbors(d+GLOBALnRegNodes,segRegNeighbors[i],1);
                        addNeighbor(d+GLOBALnRegNodes,segRegNeighbors[i],m_numberOfNeighborsofEachNode,m_neighbourArray,m_weights);

                        edgeCount++;
                        if (m_cachePotentials){


                            nSegRegEdges++;
                            for (int l1=0;l1<nSegLabels;++l1){
                                for (int l2=0;l2<nRegLabels;++l2){
                                    //forward
                                    LOGV(25)<<VAR(d)<<" "<<VAR(l1)<<" "<<VAR(segRegNeighbors[i])<<" "<<VAR(l2)<<std::endl;
                                    if (m_pairwiseSegmentationRegistrationWeight>0){
                                        (*srsPairwise)[l1][l2][d]=m_pairwiseSegmentationRegistrationWeight*this->m_GraphModel->getPairwiseRegSegPotential(segRegNeighbors[i],d,l2,l1);
                                    }else{
                                        (*srsPairwise)[l1][l2][d]=0.0;
                                    }
                                }
                            }
                        }

                    }

                }
            }
            clock_t endPairwise = clock();
            t = (float) ((double)(endPairwise-endUnary ) / CLOCKS_PER_SEC);
            LOGV(1)<<"Segmentation + SRS pairwise took "<<t<<" seconds."<<std::endl;
            LOGV(1)<<"Approximate size of seg pairwise: "<<1.0/(1024*1024)*nSegEdges*nSegLabels*nSegLabels*sizeof(double)*m_cachePotentials<<" mb."<<std::endl;
            LOGV(1)<<"Approximate size of SRS pairwise: "<<1.0/(1024*1024)*nSegRegEdges*nSegLabels*nRegLabels*sizeof(double)*m_cachePotentials<<" mb."<<std::endl;
            
        }
        m_optimizer->setSmoothCost(&GLOBALsmoothFunction);
        m_optimizer->setAllNeighbors(m_numberOfNeighborsofEachNode,m_neighbourArray,m_weights);
        clock_t finish = clock();
        double t = (float) ((double)(finish - start) / CLOCKS_PER_SEC);
        //tInterpolation+=t;
        LOGV(1)<<"Finished init after "<<t<<" seconds"<<std::endl;
        nEdges=edgeCount;
        logResetStage;
        std::vector<int> order(GLOBALnRegLabels+GLOBALnSegLabels);
        for (int l=0;l<GLOBALnSegLabels;++l){
            order[l]=GLOBALnRegLabels+l;
        }
        for (int l=0;l<GLOBALnRegLabels;++l){
            order[l+GLOBALnSegLabels]=l;
        }
#if 1
        m_optimizer->setLabelOrder(&order[0],GLOBALnRegLabels+GLOBALnSegLabels);
#else
        bool random = true;
        m_optimizer->setLabelOrder(random);
#endif
        int verbosity=0;
        if (verbose>7)
            verbosity=2;
        else if (verbose>3)
            verbosity=1;
        m_optimizer->setVerbosity(verbosity);
    }
    

    virtual double optimize(int maxIter=20){
        logSetStage("GC-Optimizer");

        clock_t opt_start=clock();
        double energy;//=m_optimizer->compute_energy();
        //LOGV(2)<<VAR(energy)<<std::endl;
        try{
            m_optimizer->expansion(maxIter==0?-1:maxIter);
            //m_optimizer->swap(maxIter);
        }catch (GCException e){
            e.Report();
        }
        energy=m_optimizer->compute_energy();
        clock_t finish = clock();
        tOpt+=((double)(finish-opt_start)/CLOCKS_PER_SEC);
        float t = (float) ((double)(finish - m_start) / CLOCKS_PER_SEC);
        LOG<<"Finished optimization after "<<t<<" , resulting energy is "<<energy<<std::endl;
        logResetStage;         
        return energy;

    }
    virtual double optimizeOneStep(int currentIter , bool & converged){
        clock_t opt_start=clock();
        double energy;//=
        //LOGV(2)<<VAR(energy)<<std::endl;
        try{
            m_optimizer->expansion(1);
            //m_optimizer->swap(maxIter);
        }catch (GCException e){
            e.Report();
        }
        energy=m_optimizer->compute_energy();
        clock_t finish = clock();
        tOpt+=((double)(finish-opt_start)/CLOCKS_PER_SEC);
        float t = (float) ((double)(finish - m_start) / CLOCKS_PER_SEC);
        LOG<<VAR(currentIter)<<" Finished optimization after "<<t<<" , resulting energy is "<<energy<<std::endl;
        if (currentIter>0){
            converged= (converged || (fabs(this->m_lastLowerBound-energy) < 1e-6 * this->m_lastLowerBound ));
        }
        //misuse member variable for storing last energy
        this->m_lastLowerBound=energy;
        return -1;

    }
    virtual std::vector<int> getDeformationLabels(){
        std::vector<int> Labels(nRegNodes,0);
        if (m_register){
            for (int i=0;i<nRegNodes;++i){
                Labels[i]=m_optimizer->whatLabel(i);
                LOGV(20)<<"DEF "<<VAR(i)<<" "<<VAR(Labels[i])<<std::endl;
            }
        }
        return Labels;
    }
    virtual std::vector<int> getSegmentationLabels(){
        std::vector<int> Labels(nSegNodes,0);
        if (m_segment){
            for (int i=0;i<nSegNodes;++i){
                Labels[i]=m_optimizer->whatLabel(i+GLOBALnRegNodes)-GLOBALnRegLabels;
                Labels[i]=Labels[i]>0?Labels[i]:0;
                LOGV(20)<<"SEG "<<VAR(i)<<" "<<VAR(Labels[i])<<std::endl;
            }
        }
        return Labels;
    }

    void evalSolution(){
        LOG<<"NYI"<<std::endl;
    }

    void addNeighbor(int id1, int id2, int * neighbCount, int **neighbors, EnergyType ** weights){
        
        LOGV(15)<<"Adding neighbors "<<id1<<" "<<id2<<" with counts "<<VAR(neighbCount[id1])<< " "<<VAR(neighbCount[id2])<<std::endl;
        //allocate memory if not yet allocated
#ifdef ALLOCINDIVIDUAL
        if (neighbCount[id1] == 0){
            int nNeighbors=2*D;
            if (id1>=GLOBALnRegLabels && m_coherence){
                nNeighbors+=1;
            }else if (id1<GLOBALnRegLabels && m_coherence){
                nNeighbors+=this->m_GraphModel->getRegSegNeighbors(id1).size();
            }
            neighbors[id1]=new int[nNeighbors];
            weights[id1]=new EnergyType[nNeighbors];
            LOGV(15)<<"allocated id1"<<std::endl;

        }
        if (neighbCount[id2] == 0){
            int nNeighbors=2*D;
            if (id2>=GLOBALnRegLabels && m_coherence){
                nNeighbors+=1;
            }else if (id2<GLOBALnRegLabels && m_coherence){
                nNeighbors+=this->m_GraphModel->getRegSegNeighbors(id2).size();
            }
            neighbors[id2]=new int[nNeighbors];
            weights[id2]=new EnergyType[nNeighbors];
            LOGV(15)<<"allocated id2"<<std::endl;
        }
#endif
        neighbors[id1][neighbCount[id1]]=id2;
        weights[id1][neighbCount[id1]]=1;
        neighbCount[id1]++;                       
        LOGV(15)<<"added id1->id2"<<std::endl;
        neighbors[id2][neighbCount[id2]]=id1;
        weights[id2][neighbCount[id2]]=1;
        neighbCount[id2]++;                       
        LOGV(15)<<"added id2->id1"<<std::endl;

    }
};

template<class T> std::vector<std::vector<std::vector<std::map<int,float> > > >  * GCO_SRSMRFSolver<T>::regPairwise = NULL;
template<class T> std::vector<std::vector<std::vector<std::vector<float> > > >   * GCO_SRSMRFSolver<T>::segPairwise = NULL;
template<class T> std::vector<std::vector<std::vector<float > > >  * GCO_SRSMRFSolver<T>::srsPairwise = NULL;
template<class T>  typename GCO_SRSMRFSolver<T>::GraphModelPointerType   GCO_SRSMRFSolver<T>::m_GraphModel=NULL;

template<class T> int  GCO_SRSMRFSolver<T>::S0=0;
template<class T> int  GCO_SRSMRFSolver<T>::S1=0;
template<class T> int  GCO_SRSMRFSolver<T>::GLOBALnRegNodes=0;
template<class T> int  GCO_SRSMRFSolver<T>::GLOBALnSegNodes=0;
template<class T> int  GCO_SRSMRFSolver<T>::GLOBALnRegLabels=0;
template<class T> int  GCO_SRSMRFSolver<T>::GLOBALnSegLabels=0;
template<class T>  double GCO_SRSMRFSolver<T>::m_pairwiseSegmentationRegistrationWeight=0;
template<class T>  double GCO_SRSMRFSolver<T>::m_pairwiseSegmentationWeight=0;
template<class T>  double GCO_SRSMRFSolver<T>::m_pairwiseRegistrationWeight=0;
template<class T>  bool GCO_SRSMRFSolver<T>::m_cachePotentials=false;

}
