#include <iostream>
#include <string>
#include <vector>
#include <set>
#include "hyperparameters.h"
#include <boost/numeric/ublas/matrix.hpp>
#include "utilities.h"
#include <libxml/tree.h>
#include <libxml/parser.h>
using namespace boost::numeric::ublas;
using std::cout;
using std::endl;
#ifndef TREE_H_
#define TREE_H_

#include "node.h"
#include "nodegini.h"
#include "nodeinfogain.h"
#include "nodehyperplane.h"

class Tree
{
public:
    Tree(const HyperParameters &hp, const int n=-1);
    Tree(const HyperParameters &hp, const xmlNodePtr treeNode,const int n=-1);
    ~Tree();

    xmlNodePtr save() const;

    void train(const matrix<float>& data, const std::vector<int>& labels);
    void train(const matrix<float>& data, const std::vector<int>& labels, const std::vector<double>& weights);

    void train(const matrix<float>& data, const std::vector<int>& labels,
               matrix<float>& forestConfidences, matrix<float>& forestOutOfBagConfidences,
               std::vector<int>& forestOutOfBagVoteNum);
    void train(const matrix<float>& data, const std::vector<int>& labels, const std::vector<double>& weights,
               matrix<float>& forestConfidences, matrix<float>& forestOutOfBagConfidences,
               std::vector<int>& forestOutOfBagVoteNum);

    void trainLU(const matrix<float>& data, const std::vector<int>& labels);

    void eval(const matrix<float>& data, const std::vector<int>& labels);

    void eval(const matrix<float>& data, const std::vector<int>& labels,
              matrix<float>& forestConfidences);

    inline std::vector<int> getPredictions() const { return m_predictions; };
    inline std::vector<int> getInBagSamples() const { return m_inBagSamples; };
    inline std::vector<int> getOutOfBagSamples() const { return m_outOfBagSamples; };
    inline matrix<float> getConfidences() const { return m_confidences; };
    inline int getNumNodes() const { return m_rootNode->numNodes(); };

    void setInBagSamples(const std::vector<int>& inBagSamples) { m_inBagSamples = inBagSamples; };
    void setOutOfBagSamples(const std::vector<int>& outOfBagSamples) { m_outOfBagSamples = outOfBagSamples; };

    void getTreeAsMatrix(matrix<float> *data, const int tree_index);

    std::vector<std::vector<int> > getPath(const matrix<float>& data);

    int getDepth();

    std::vector<int> getNodeLabels();

    void clean();

private:
    HyperParameters m_hp;
    Node::Ptr m_rootNode;

    matrix<float> m_confidences;
    std::vector<int> m_predictions;
    std::vector<int> m_inBagSamples;
    std::vector<int> m_outOfBagSamples;

    void initialize(const long int numSamples); // Create the confidence matrices and prediction vectors
    void reInitialize(const long int numSamples); // Create the confidence matrices and prediction vectors

    void subSample(const long int numSamples);    // Create bags
    void subSample(const long int numSamples, const std::vector<double>& weights);    // Create bags

    void finalize(const matrix<float>& data,
                  matrix<float>& forestConfidences, matrix<float>& forestOutOfBagConfidences,
                  std::vector<int>& forestOutOfBagVoteNum);

    double computeError(const std::vector<int>& labels);
    void getTreeAsMatrixRecursive(Node::Ptr current_node, matrix<float> *data, const int tree_index, const int node_index);

    double computeError(const std::vector<int>& labels, const std::vector<int>& sampleIndeces);
    void evalOutOfBagSamples(const matrix<float>& data);
};

#endif /* TREE_H_ */
