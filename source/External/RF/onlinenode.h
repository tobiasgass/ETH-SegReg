#ifndef ONLINE_NODE_H_
#define ONLINE_NODE_H_

#include "hyperparameters.h"
#include "utilities.h"
#include <boost/shared_ptr.hpp>
#include <string>
#include <set>
#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>

using namespace boost::numeric::ublas;
using namespace std;

typedef struct {
    double threshold;
    int lTotal;
    int rTotal;
    std::vector<int> lCount;
    std::vector<int> rCount;
} Statistics;

typedef struct {
    int index; // For the single feature Gini
    std::vector<int> indeces; // of the features value in the feature vector
    std::vector<float> weights;
    int bestStatistics; // Holds index to best statistics
    double bestGini;
    std::vector<Statistics> statistics;
} Feature;

class OnlineNode {
public:
    // Used for the return values of the node training
    typedef enum {IS_NOT_LEAF, CHANGE_TO_SVM, IS_LEAF} NODE_TRAIN_STATUS;

    typedef boost::shared_ptr<OnlineNode> Ptr;

    OnlineNode(const HyperParameters &hp, int depth, int numFeatures);
    OnlineNode(const HyperParameters &hp, int depth, int numFeatures, std::vector<int>& labelStatistics);
    virtual ~OnlineNode();

    virtual void train(const std::vector<float>& sample, const int label) = 0;

    virtual double eval(const std::vector<float>& sample, const int label) = 0;

    virtual void eval(const matrix<float>& data, const std::vector<int>& sampleIndeces,
                      matrix<float>& confidences, std::vector<int>& predictions) = 0;

    inline bool isLeaf() const {
        return m_isLeaf;
    };

    inline int depth() const {
        return m_depth;
    };

    inline int numNodes() const {
        return m_numNodes;
    };

    inline int nodeIndex() const {
        return m_nodeIndex;
    };

    inline std::vector<float> nodeConf() const {
        return m_nodeConf;
    };

    inline int nodeLabel() const {
        return m_nodeLabel;
    };

    inline Ptr leftChildNode() const {
        return m_leftChildNode;
    };
    inline Ptr rightChildNode() const {
        return m_rightChildNode;
    };

    int getDepth(const int oldDepth);

    virtual std::vector<int> bestFeature() = 0;
    virtual std::vector<float> bestWeight() = 0;
    virtual float bestThreshold() = 0;

    void getLabels(std::set<int>& nodeLabels);
protected:

    bool shouldISplit();

    const HyperParameters *m_hp;
    double m_numSamples; // samples seen so far
    bool m_isLeaf;
    Ptr m_leftChildNode;
    Ptr m_rightChildNode;
    int m_depth;
    int m_nodeLabel;
    int m_nodeIndex;
    static int m_numNodes;
    std::vector<float> m_nodeConf;
    std::vector<int> m_labelStatistics;
    int m_numFeatures;
    double m_bestGini;
    double m_secondBestGini;
    std::vector<double> m_deltaGinis;
    std::vector<double> m_epsilons;
};


#endif /* ONLINE_NODE_H_ */

