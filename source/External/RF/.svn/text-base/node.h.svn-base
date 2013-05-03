#ifndef NODE_H_
#define NODE_H_

#include "hyperparameters.h"
#include "utilities.h"
#include <boost/shared_ptr.hpp>
#include <string>
#include <set>
#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>
#include <libxml/tree.h>

using namespace boost::numeric::ublas;
using namespace std;

const char* const LEFT_CHILD_NODE = "left";
const char* const RIGHT_CHILD_NODE = "right";

// Used for the return values of the node training
typedef enum {IS_NOT_LEAF, CHANGE_TO_SVM, IS_LEAF} NODE_TRAIN_STATUS;

class Node {
public:
    typedef boost::shared_ptr<Node> Ptr;
    Node(const HyperParameters &hp, int depth);
    Node(const HyperParameters &hp, int depth, int reset);
    virtual ~Node();

    virtual xmlNodePtr save() const = 0;

    virtual NODE_TRAIN_STATUS train(const matrix<float>& data, const std::vector<int>& labels, std::vector<int>& inBagSamples,
                                    matrix<float>& confidences, std::vector<int>& predictions) = 0;
    virtual NODE_TRAIN_STATUS train(const matrix<float>& data, const std::vector<int>& labels, const std::vector<double>& weights,
                                    std::vector<int>& inBagSamples, matrix<float>& confidences, std::vector<int>& predictions) = 0;

    virtual NODE_TRAIN_STATUS trainLU(const matrix<float>& data, const std::vector<int>& labels, std::vector<int>& inBagSamples,
                                      matrix<float>& confidences, std::vector<int>& predictions) = 0;


    virtual void refine(const matrix<float>& data, const std::vector<int>& labels,
                                    std::vector<int>& samples, matrix<float>& confidences, std::vector<int>& predictions) = 0;

    virtual void refine(const matrix<float>& data, const std::vector<int>& labels, const std::vector<double>& weights,
                                    std::vector<int>& samples,matrix<float>& confidences, std::vector<int>& predictions) = 0;

    virtual void eval(const matrix<float>& data, const std::vector<int>& sampleIndeces,
                      matrix<float>& confidences, std::vector<int>& predictions) = 0;

    virtual void getPath(const matrix<float>& data, const std::vector<int>& sampleIndeces, std::vector<std::vector<int> >& path) = 0;

    inline bool isLeaf() const { return m_isLeaf; };

    inline int depth() const { return m_depth; };

    inline int numNodes() const { return m_numNodes; };

    inline int nodeIndex() const { return m_nodeIndex; };

    inline std::vector<float> nodeConf() const {return m_nodeConf; };

    inline int nodeLabel() const { return m_nodeLabel; };

    inline Ptr leftChildNode() const {return m_leftChildNode; };
    inline Ptr rightChildNode() const {return m_rightChildNode; };

    int getDepth(const int oldDepth);

    virtual std::vector<int> bestFeature() = 0;
    virtual std::vector<float> bestWeight() = 0;
    virtual float bestThreshold() = 0;

    void getLabels(std::set<int>& nodeLabels);

protected:

    bool shouldISplit(const std::vector<int>& labels, std::vector<int>& inBagSamples);
    bool shouldISplitLU(const std::vector<int>& labels, std::vector<int>& inBagSamples);

    xmlNodePtr saveConfidence(const int idx,const float conf) const;

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
};


#endif /* NODE_H_ */
