//////////////////////////////////////////////////
//// SRS-MRF
//// Tobias Gass
//// ETH Zurich
//// No license yet!
/////////////////////////////////////////////////

#ifndef MRF_H
#define MRF_H

#include "Graph.h"

//do we need to typedef/template over nodes and edges here??
class MRF{
public:
    typedef Graph TGraph;

    void setGraph(MRF::TGraph * graph){graph_=graph;}
    MRF::TGraph * getGraph(){
    		return graph_;
    	}
    virtual void addNode(Node n)=0;
    virtual void addEdge(Edge E)=0;

private:
    MRF::TGraph * graph_;
    
};//MRF




#endif //MRF_H
