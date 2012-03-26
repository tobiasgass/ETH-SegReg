#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "typeGeneral.h"
#include "MRFEnergy.h"
#include "minimize.cpp"
#include "treeProbabilities.cpp"
#include <vector>
#include <fenv.h>
#include <google/heap-profiler.h>
#include "HierarchicalSRSImageToImageFilter.h"
#include <stdlib.h>
using namespace std;

template<class T>
class test{
public:
    typedef TypeGeneral TRWType;
    typedef MRFEnergy<TRWType> MRFType;
    typedef  MRFType::NodeId NodeType;
    typedef  MRFType::EdgeId EdgeType;
private:
    MRFType optimizer;
public:
    test():optimizer(TRWType::GlobalSize()){};
    ~test(){
        sleep(1);
        sleep(1);
    }
    void init(){
        int nLabels=10;
        int nLabels2=3;
        int nNodes=1000000;

        vector<NodeType> nodes(nNodes*2, NULL);
        TRWType::GlobalSize globalSize();
        TRWType::REAL D1[nLabels];
        TRWType::REAL D2[nLabels2];
        for (int n=0;n<nNodes;++n){
            nodes[n]=optimizer.AddNode(TRWType::LocalSize(nLabels), TRWType::NodeData(D1));
            nodes[n+nNodes]=optimizer.AddNode(TRWType::LocalSize(nLabels2), TRWType::NodeData(D2));
        }
        TRWType::REAL V[nLabels*nLabels];
        TRWType::REAL V2[nLabels*nLabels];
        int c=0;
        for (int n=0;n<nNodes-1;++n){
            optimizer.AddEdge(nodes[n], nodes[n+1], TRWType::EdgeData(TRWType::GENERAL,V));
            optimizer.AddEdge(nodes[n], nodes[n+nNodes], TRWType::EdgeData(TRWType::GENERAL,V2));
            c+=2;
        }        
        sleep(1);
        LOG<<" "<<c<<std::endl;
    }

};


template<class T>
class test2{
private:
    std::vector<char *> pointers;
public:
    test2(){};
    ~test2(){
        sleep(1);
        for (unsigned int i=0; i<pointers.size();++i){
        if (pointers[i]){
            delete pointers[i];
            pointers[i]=NULL;
        }
    }
    LOG<<"freed TRW-S "<<std::endl;
        sleep(1);
    }
    void init(){
        int nLabels=10;
        int nNodes=10000000;
        for (unsigned int i=0;i<nNodes;++i){
            pointers.push_back(new char[nLabels*nLabels]);
        }
        sleep(1);

    }

};

int main(int argc, char ** argv)
{

	feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
  

   
    for (int i=0;i<30;++i){
            
        typedef test2<int> testType;
        testType t;//= new testType;
        t.init();
        testType t2;//=new testType;
        t2.init();
#if 0
        testType *t3=new testType;
        t3.init();
        sleep(1);
        delete t3;
        LOG<<"deleted optimizer"<<endl;
#endif     

        LOG<<i<<std::endl;

    }

    
	return 1;
}
