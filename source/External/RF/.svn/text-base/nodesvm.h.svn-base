#include "node.h"
#include "hyperparameters.h"
#include "utilities.h"
#include <boost/shared_ptr.hpp>
#include <string>
#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>
#include "../extern/svm/liblinear-1.33/linear.h"
#include <libxml/tree.h>
#include <libxml/parser.h>

#ifndef NODE_SVM_
#define NODE_SVM_

using namespace boost::numeric::ublas;
using namespace std;

const char* const NODE_SVM = "nodeSVM";

class NodeSVM : public Node {
public:
    typedef boost::shared_ptr<NodeSVM> Ptr;

    NodeSVM(const HyperParameters &hp, int depth);
    NodeSVM(const HyperParameters &hp, int depth, int reset);
    NodeSVM(const HyperParameters &hp, int reset, xmlNodePtr nodeNode);
    ~NodeSVM();

    virtual xmlNodePtr save() const;

    virtual NODE_TRAIN_STATUS train(const matrix<float>& data, const std::vector<int>& labels, std::vector<int>& inBagSamples,
                                    matrix<float>& confidences, std::vector<int>& predictions);
    virtual NODE_TRAIN_STATUS train(const matrix<float>& data, const std::vector<int>& labels, const std::vector<double>& weights,
                                    std::vector<int>& inBagSamples, matrix<float>& confidences, std::vector<int>& predictions);

    virtual NODE_TRAIN_STATUS trainLU(const matrix<float>& data, const std::vector<int>& labels, std::vector<int>& inBagSamples,
                                      matrix<float>& confidences, std::vector<int>& predictions);

    virtual void refine(const matrix<float>& data, const std::vector<int>& labels,
                                    std::vector<int>& samples, matrix<float>& confidences, std::vector<int>& predictions);

    virtual void refine(const matrix<float>& data, const std::vector<int>& labels, const std::vector<double>& weights,
                                    std::vector<int>& samples,matrix<float>& confidences, std::vector<int>& predictions);

    virtual void eval(const matrix<float>& data, const std::vector<int>& sampleIndeces,
                      matrix<float>& confidences, std::vector<int>& predictions);

    virtual void getPath(const matrix<float>& data, const std::vector<int>& sampleIndeces, std::vector<std::vector<int> >& path);

    static int m_svmNodeCount;

    virtual std::vector<int> bestFeature() { return std::vector<int>(1,-1); };
    virtual std::vector<float> bestWeight() { return std::vector<float>(1,1.0f); };
    virtual float bestThreshold() { return -10.f; };

private:
    void evalNode(const matrix<float>& data, const std::vector<int>& inBagSamples,
                  std::vector<int>& leftNodeSamples, std::vector<int>& rightNodeSamples);
    xmlNodePtr saveModel() const;

    feature_node* convertData(const matrix<float>& data, const std::vector<int>& labels, const std::vector<int>& inBagSamples,
                              problem& svmProblem);
    model* m_svmModel;
    int m_svmNodeIdx;
};

#endif // NODE_SVM_
