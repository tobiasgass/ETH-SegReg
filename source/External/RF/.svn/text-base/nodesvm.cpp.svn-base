#include "nodesvm.h"
#include "nodegini.h"
#include "nodeinfogain.h"
#include <boost/foreach.hpp>

#if WIN32
#define snprintf sprintf_s
#endif

int NodeSVM::m_svmNodeCount;

NodeSVM::NodeSVM(const HyperParameters &hp, int depth) : Node(hp, depth)
{
    m_svmNodeIdx = m_svmNodeCount++;
}

NodeSVM::NodeSVM(const HyperParameters &hp, int depth, int reset) : Node(hp, depth, reset)
{
    m_svmNodeIdx = m_svmNodeCount++;
}

NodeSVM::NodeSVM(const HyperParameters &hp, int reset, xmlNodePtr nodeNode) : Node(hp,0,reset)
{
    m_svmNodeIdx = m_svmNodeCount++;
    m_isLeaf = true; // set default to true
    xmlNodePtr cur = nodeNode->xmlChildrenNode;
    std::string modelFileName;
    while ( cur != 0 )
    {
        if ( xmlStrcmp( cur->name, reinterpret_cast<const xmlChar*>( "model" ) ) == 0 )
        {
            modelFileName = readStringProp(cur,"name");
#ifndef WIN32
            m_svmModel  = load_model(modelFileName.c_str());
#else
            printf("\nthere is no libsvm support under windows yet...!");
#endif
        }
        cur = cur->next;
    }
}

NodeSVM::~NodeSVM()
{
#ifndef WIN32
    destroy_model(m_svmModel);
#else
    printf("\nthere is no libsvm support under windows yet...!");
#endif
}

void NodeSVM::evalNode(const matrix<float>& data, const std::vector<int>& inBagSamples,
                       std::vector<int>& leftNodeSamples, std::vector<int>& rightNodeSamples)
{
}

xmlNodePtr NodeSVM::saveModel() const
{
    std::string modelFileName = m_hp.savePath;
    modelFileName += "svmmodels/";
    int digits = 4;
    char* buffer = new char[digits];
    snprintf( buffer, digits+1, "%0*d", digits, m_svmNodeIdx );
    modelFileName += buffer;
    delete [] buffer;
    modelFileName += ".model";
    xmlNodePtr node = xmlNewNode( NULL, reinterpret_cast<const xmlChar*>( "model" ) );
    xmlNewProp( node, reinterpret_cast<const xmlChar*>( "name" ),
        reinterpret_cast<const xmlChar*>( modelFileName.c_str() ) );
#ifndef WIN32
    save_model(modelFileName.c_str(),m_svmModel);
#else
    printf("\nthere is no libsvm support under windows yet...!");
    return false;

#endif
    return node;
}

xmlNodePtr NodeSVM::save() const
{
    xmlNodePtr node = xmlNewNode( NULL, reinterpret_cast<const xmlChar*>( "node" ) );
    xmlNewProp( node, reinterpret_cast<const xmlChar*>( "type" ),
        reinterpret_cast<const xmlChar*>( NODE_SVM ) );
    const char* isLeaf = (m_isLeaf) ? "true" : "false";
    xmlNewProp( node, reinterpret_cast<const xmlChar*>( "isLeaf" ),
        reinterpret_cast<const xmlChar*>( isLeaf ) );
    if (!m_isLeaf)
    {
        xmlAddChild(node, saveModel());
        xmlNodePtr leftChildNode = m_leftChildNode->save();
        xmlNewProp( leftChildNode, reinterpret_cast<const xmlChar*>( "child" ),
            reinterpret_cast<const xmlChar*>( LEFT_CHILD_NODE ) );
        xmlAddChild( node, leftChildNode );

        xmlNodePtr rightChildNode = m_rightChildNode->save();
        xmlNewProp( rightChildNode, reinterpret_cast<const xmlChar*>( "child" ),
            reinterpret_cast<const xmlChar*>( RIGHT_CHILD_NODE ) );
        xmlAddChild( node, rightChildNode );
    }
    else
    {
        xmlAddChild(node, saveModel());
        //addIntProp( node, "label", m_nodeLabel);
        std::vector<float>::const_iterator it(m_nodeConf.begin()),end(m_nodeConf.end());
        int idx = 0;
        for (;it != end;it++,idx++)
        {
            xmlAddChild(node,saveConfidence(idx,*it));
        }
    }
    return node;
}

NODE_TRAIN_STATUS NodeSVM::train(const matrix<float>& data, const std::vector<int>& labels, std::vector<int>& inBagSamples,
                                 matrix<float>& confidences, std::vector<int>& predictions)
{
    NODE_TRAIN_STATUS myTrainingStatus = IS_LEAF;
    m_isLeaf = true;

    problem svmProblem;
    feature_node *svmX = convertData(data, labels, inBagSamples, svmProblem);

    parameter svmParameters;
    svmParameters.solver_type = m_hp.svmSolverType;
    svmParameters.C = m_hp.svmC;
    svmParameters.eps = m_hp.svmEps;
    svmParameters.nr_weight = 0;
    svmParameters.weight_label = NULL;
    svmParameters.weight = NULL;

#ifndef WIN32
    m_svmModel = trainSVM(&svmProblem, &svmParameters);
#else
    printf("\nthere is no libsvm support under windows yet...!");
#endif

    delete [] svmProblem.x;
    delete [] svmProblem.y;
    delete [] svmX;

    eval(data, inBagSamples, confidences, predictions);

    return myTrainingStatus;
}

NODE_TRAIN_STATUS NodeSVM::trainLU(const matrix<float>& data, const std::vector<int>& labels, std::vector<int>& inBagSamples,
                                   matrix<float>& confidences, std::vector<int>& predictions)
{
    return train(data, labels, inBagSamples, confidences, predictions);
}


NODE_TRAIN_STATUS NodeSVM::train(const matrix<float>& data, const std::vector<int>& labels, const std::vector<double>& weights,
                                 std::vector<int>& inBagSamples, matrix<float>& confidences, std::vector<int>& predictions)
{
    NODE_TRAIN_STATUS myTrainingStatus = IS_LEAF;
    m_isLeaf = true;
    problem svmProblem;
    feature_node *svmX = convertData(data, labels, inBagSamples, svmProblem);

    parameter svmParameters;
    svmParameters.solver_type = m_hp.svmSolverType;
    svmParameters.C = m_hp.svmC;
    svmParameters.eps = m_hp.svmEps;
    svmParameters.nr_weight = 0;
    svmParameters.weight_label = NULL;
    svmParameters.weight = NULL;

#ifndef WIN32
    m_svmModel = trainSVM(&svmProblem, &svmParameters);
#else
    printf("\nthere is no libsvm support under windows yet...!");
#endif

    delete [] svmProblem.x;
    delete [] svmProblem.y;
    delete [] svmX;

    eval(data, inBagSamples, confidences, predictions);

    return myTrainingStatus;
}

void NodeSVM::eval(const matrix<float>& data, const std::vector<int>& sampleIndeces,
                   matrix<float>& confidences, std::vector<int>& predictions)
{
    prediction tmpPre;
    std::vector<int> leftNodeSamples, rightNodeSamples;
    feature_node* tmpFeatureNode = new feature_node [(data.size2() + 2)];
    int i;
    BOOST_FOREACH(int n, sampleIndeces) {
        i = 0;
        for (int j = 0; j < (int) data.size2(); j++)
        {
            tmpFeatureNode[i].index = j + 1;
            tmpFeatureNode[i].value = data(n, j);
            i++;
        }
        tmpFeatureNode[i].index = (int) data.size2() + 1;
        tmpFeatureNode[i].value = 1;
        i++;
        tmpFeatureNode[i].index = -1;

        if (m_hp.svmSolverType == 0 && m_hp.useSoftVoting) {
#ifndef WIN32
            evalSVM(m_svmModel, tmpFeatureNode, tmpPre);
#else
            printf("\nthere is no libsvm support under windows yet...!");
#endif
            predictions[n] = tmpPre.label;
            for (int nClass = 0; nClass < m_hp.numClasses; nClass++) {
                confidences(n, nClass) += (float)tmpPre.conf[nClass];
            }
        }
        else {
#ifndef WIN32
            tmpPre.label = predict(m_svmModel, tmpFeatureNode);
#else
            printf("\nthere is no libsvm support under windows yet...!");
#endif
            predictions[n] = tmpPre.label;
            confidences(n, tmpPre.label)++;
        }
    }
    delete [] tmpFeatureNode;
}

feature_node* NodeSVM::convertData(const matrix<float>& data, const std::vector<int>& labels, const std::vector<int>& inBagSamples,
                                   problem& svmProblem)
{
    // Define a problem for the SVM
    svmProblem.bias = true;
    svmProblem.l = (int) inBagSamples.size();
    svmProblem.n = (int) data.size2() + 1;
    svmProblem.y = new int [(int) inBagSamples.size()];
    svmProblem.x = new feature_node* [(int) inBagSamples.size()];

    feature_node* svmX = new feature_node [(int) inBagSamples.size()*(data.size2() + 2)]; // sampNum*(featNum + bias + endSampIndicator)

    int nS = 0, j = 0;
    BOOST_FOREACH(int n, inBagSamples) {
        svmProblem.y[nS] = labels[n];
        svmProblem.x[nS] = &svmX[j];
        for (int nF = 0; nF < (int) data.size2(); nF++)
        {
            svmX[j].index = nF + 1;
            svmX[j].value = data(n, nF);
            j++;
        }
        svmX[j].value = 1; // bias
        svmX[j].index = (int) data.size2() + 1;
        j++;

        svmX[j].index = -1; // sample end indicator
        j++;

        nS++;
    }
    return svmX;
}

void NodeSVM::getPath(const matrix<float>& data, const std::vector<int>& sampleIndeces, std::vector<std::vector<int> >& path) {
    BOOST_FOREACH(int n, sampleIndeces) {
        path[n].push_back(m_nodeIndex);
    }

    if (!m_isLeaf) {
        // split the data
        std::vector<int> leftNodeSamples, rightNodeSamples;
        evalNode(data,sampleIndeces,leftNodeSamples,rightNodeSamples);

        m_leftChildNode->getPath(data,leftNodeSamples,path);
        m_rightChildNode->getPath(data,rightNodeSamples,path);
    }
}

void NodeSVM::refine(const matrix<float>& data, const std::vector<int>& labels,
                  std::vector<int>& samples, matrix<float>& confidences, std::vector<int>& predictions)
{

}

void NodeSVM::refine(const matrix<float>& data, const std::vector<int>& labels, const std::vector<double>& weights,
                  std::vector<int>& samples, matrix<float>& confidences, std::vector<int>& predictions)
{
    // calc confidence, labels, etc
    if (!m_isLeaf)
    {


    }
    else
    {
        m_nodeConf.resize(m_hp.numClasses, 0.0);

        double totalW = 0;
        BOOST_FOREACH(int n, samples)
        {
            m_nodeConf[labels[n]] += weights[n];
            totalW += weights[n];
        }

        int bestClass = 0, tmpN = 0;
        float bestConf = 0;
        std::vector<float>::iterator confItr = m_nodeConf.begin(), confEnd = m_nodeConf.end();
        for (; confItr != confEnd; confItr++)
        {
            *confItr /= (totalW + 1e-10);
            if (*confItr > bestConf)
            {
                bestConf = *confItr;
                bestClass = tmpN;
            }
            tmpN++;
        }
        m_nodeLabel = bestClass;

        BOOST_FOREACH(int n, samples)
        {
            predictions[n] = m_nodeLabel;
            tmpN = 0;
            BOOST_FOREACH(float conf, m_nodeConf)
            {
                confidences(n, tmpN) = conf;
                tmpN++;
            }
        }
    }
}
