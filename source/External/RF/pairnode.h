#ifndef PAIR_NODE_H_
#define PAIR_NODE_H_

#include "hyperparameters.h"
#include "utilities.h"
#include "data.h"
#include <boost/shared_ptr.hpp>
#include <string>
#include <set>
#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>
#include <libxml/tree.h>

using namespace boost::numeric::ublas;
using namespace std;

class PairNode {
public:
    typedef boost::shared_ptr<PairNode> Ptr;
    PairNode(const HyperParameters &hp, int depth);
    PairNode(const HyperParameters &hp, int depth, int reset);
    virtual ~PairNode();

    void train(const std::vector<Pair>& pairs);

    inline int eval(const std::vector<float>& x) const;

    void getPath(const std::vector<float>& x, std::vector<int>& path);

    inline bool isLeaf() const { return m_isLeaf; };

    inline int depth() const { return m_depth; };
    int getDepth(const int oldDepth);
    inline int numNodes() const { return m_numNodes; };
    inline int numLeafs() const;
    inline int nodeIndex() const { return m_nodeIndex; };
    std::vector<int> getLeafNodeIds();

    inline Ptr leftChildNode() const {return m_leftChildNode; };
    inline Ptr rightChildNode() const {return m_rightChildNode; };

private:

    bool shouldISplit(const std::vector<Pair>& pairs);
    void evalNode(const std::vector<Pair>& pairs,std::vector<Pair>& leftPairs,std::vector<Pair>& rightPairs);
    void findHypotheses(const std::vector<Pair>& pairs, const std::vector<int> randFeatures);
    HyperParameters m_hp;

    bool m_isLeaf;
    Ptr m_leftChildNode;
    Ptr m_rightChildNode;
    int m_depth;
    int m_nodeLabel;
    int m_nodeIndex;
    static int m_numNodes;
    std::vector<float> m_nodeConf;
    float m_totalWeights;
    int m_bestFeature;
    float m_bestThreshold;
};

inline int PairNode::eval(const std::vector<float>& x) const
{
    if (m_isLeaf) {
        return m_nodeIndex;
    }
    else {
        if ( x.at( m_bestFeature ) >= m_bestThreshold ) {
            return m_rightChildNode->eval( x );
        }
        else {
            return m_leftChildNode->eval( x );
        }
    }
}

inline int PairNode::numLeafs() const
{
    if (m_isLeaf) {
        return 1;
    }
    else {
        return m_rightChildNode->numLeafs() + m_leftChildNode->numLeafs();
    }
}
#endif /* PAIR_NODE_H_ */

