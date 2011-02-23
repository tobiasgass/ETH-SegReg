#ifndef ONLINE_TREE_H_
#define ONLINE_TREE_H_

#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <boost/numeric/ublas/matrix.hpp>
#include <libxml/tree.h>
#include <libxml/parser.h>

#include "hyperparameters.h"
#include "onlinenode.h"
#include "utilities.h"

using namespace boost::numeric::ublas;
using std::cout;
using std::endl;


class OnlineTree
{
public:
    OnlineTree(const HyperParameters &hp, int numFeatures);
    ~OnlineTree();

    void train(const std::vector<float>& sample, const int label);

    double eval(const std::vector<float>& sample, const int label);

    inline std::vector<int> getPredictions() const { return m_predictions; };
    inline std::vector<int> getInBagSamples() const { return m_inBagSamples; };
    inline std::vector<int> getOutOfBagSamples() const { return m_outOfBagSamples; };
    inline matrix<float> getConfidences() const { return m_confidences; };
    //inline int getNumNodes() const { return m_rootNode->numOnlineNodes(); };

    void setInBagSamples(const std::vector<int>& inBagSamples) { m_inBagSamples = inBagSamples; };
    void setOutOfBagSamples(const std::vector<int>& outOfBagSamples) { m_outOfBagSamples = outOfBagSamples; };

    void getOnlineTreeAsMatrix(matrix<float> *data, const int tree_index);

    //std::vector<int> getNodeLabels();

    void clean();

private:
    const HyperParameters *m_hp;
    OnlineNode::Ptr m_rootNode;

    matrix<float> m_confidences;
    std::vector<int> m_predictions;
    std::vector<int> m_inBagSamples;
    std::vector<int> m_outOfBagSamples;

    void initialize(const int numSamples); // Create the confidence matrices and prediction vectors
    void reInitialize(const int numSamples); // Create the confidence matrices and prediction vectors

    void subSample(const int numSamples);    // Create bags
    void subSample(const int numSamples, const std::vector<double>& weights);    // Create bags

    void finalize(const matrix<float>& data,
                  matrix<float>& forestConfidences, matrix<float>& forestOutOfBagConfidences,
                  std::vector<int>& forestOutOfBagVoteNum);

    double computeError(const std::vector<int>& labels);

    double computeError(const std::vector<int>& labels, const std::vector<int>& sampleIndeces);
    void evalOutOfBagSamples(const matrix<float>& data);
};

#endif /* TREE_H_ */

