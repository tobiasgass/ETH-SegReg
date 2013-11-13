#include "tree.h"
#include <cmath>
#include "utilities.h"
#include <boost/foreach.hpp>

Tree::Tree(const HyperParameters &hp, const int n) : m_hp( hp )
{
    if (n>-1){
        _rand(n);
    }
    if (hp.useRandProj)
    {
        m_rootNode = Node::Ptr(new NodeHyperPlane(hp,0,0));
    }
    else
    {
        if (hp.useInfoGain)
        {
            m_rootNode = Node::Ptr(new NodeInfoGain(hp,0,0));
        }
        else
        {
            m_rootNode = Node::Ptr(new NodeGini(hp,0,0));
        }
    }
}

Tree::Tree(const HyperParameters &hp, const xmlNodePtr treeNode,const int n) : m_hp( hp )
{
    if (n>-1){
        _rand(n);
    }
    xmlNodePtr cur = treeNode->xmlChildrenNode;
    while ( cur != 0 )
    {
        if ( xmlStrcmp( cur->name, reinterpret_cast<const xmlChar*>( "node" ) ) == 0 )
        {
            std::string nodeType = readStringProp( cur, "type" );
            if (nodeType == NODE_GINI)
            {
                m_rootNode = Node::Ptr( new NodeGini( m_hp, 0, cur ) );
            }
            else if (nodeType == NODE_INFO_GAIN)
            {
                m_rootNode = Node::Ptr( new NodeInfoGain( hp, 0, cur ) );
            }
        }
        else if ( xmlStrcmp( cur->name, reinterpret_cast<const xmlChar*>( "constants" ) ) == 0 )
        {
            //Configurator::conf()->loadConstants( cur );
        }
        cur = cur->next;
    }
}

Tree::~Tree()
{
}

xmlNodePtr Tree::save() const
{
    xmlNodePtr node = xmlNewNode( NULL, reinterpret_cast<const xmlChar*>( "tree" ) );
    xmlAddChild( node, m_rootNode->save() );
    return node;
}

void Tree::finalize(const matrix<float>& data,
                    matrix<float>& forestConfidences, matrix<float>& forestOutOfBagConfidences,
                    std::vector<int>& forestOutOfBagVoteNum)
{
    // Fill the confidence of the forest
    for (long int nSamp = 0; nSamp < m_hp.numLabeled; nSamp++)
    {
        if (m_hp.useSoftVoting)
        {
            for (int nClass = 0; nClass < m_hp.numClasses; nClass++)
            {
                forestConfidences(nSamp,nClass) += m_confidences(nSamp,nClass);
            }
        }
        else
        {
            forestConfidences(nSamp, m_predictions[nSamp])++;
        }
    }

    // Fill the out of bag confidences and vote count
    BOOST_FOREACH(long int n, m_outOfBagSamples)
    {
        forestOutOfBagVoteNum[n]++;

        if (m_hp.useSoftVoting)
        {
            for (int nClass = 0; nClass < m_hp.numClasses; nClass++)
            {
                forestOutOfBagConfidences(n,nClass) += m_confidences(n,nClass);
            }
        }
        else
        {
            forestOutOfBagConfidences(n, m_predictions[n])++;
        }
    }

    // Clean up
    clean();
}

void Tree::clean()
{
    m_confidences.clear();
}

// This
void Tree::train(const matrix<float>& data, const std::vector<int>& labels,
                 matrix<float>& forestConfidences, matrix<float>& forestOutOfBagConfidences,
                 std::vector<int>& forestOutOfBagVoteNum)
{
    // Initialize
    initialize(m_hp.numLabeled);

    // Random Subsamples data according to bagratio
    subSample(m_hp.numLabeled);

    // Train the root Node
    m_rootNode->train(data, labels, m_inBagSamples, m_confidences, m_predictions);

    std::vector<int> allSamples;
    allSamples.insert(allSamples.begin(),m_inBagSamples.begin(),m_inBagSamples.end());
    allSamples.insert(allSamples.begin(),m_outOfBagSamples.begin(),m_outOfBagSamples.end());

//    m_rootNode->refine(data,labels,allSamples,m_confidences,m_predictions);

    evalOutOfBagSamples(data);
    finalize(data, forestConfidences, forestOutOfBagConfidences, forestOutOfBagVoteNum);
    if (m_hp.verbose)
    {
        cout << "Trained a tree with " << m_rootNode->numNodes() << " nodes." << endl;
        cout << "Training error = " << computeError(labels) << ", in bag = ";
        cout << computeError(labels, m_inBagSamples) << ", out of bag = ";
        cout << computeError(labels, m_outOfBagSamples) <<  endl;
    }
}

void Tree::train(const matrix<float>& data, const std::vector<int>& labels, const std::vector<double>& weights,
                 matrix<float>& forestConfidences, matrix<float>& forestOutOfBagConfidences,
                 std::vector<int>& forestOutOfBagVoteNum)
{

    // Initialize
    initialize(m_hp.numLabeled);

    // Random Subsamples data according to bagratio
    subSample(m_hp.numLabeled);
    //subSample(m_hp.numLabeled, weights);

    // Train the root Node
    m_rootNode->train(data, labels, weights, m_inBagSamples, m_confidences, m_predictions);

    std::vector<int> allSamples;
    allSamples.insert(allSamples.begin(),m_inBagSamples.begin(),m_inBagSamples.end());
    allSamples.insert(allSamples.begin(),m_outOfBagSamples.begin(),m_outOfBagSamples.end());

//    m_rootNode->refine(data,labels,weights,allSamples,m_confidences,m_predictions);


    evalOutOfBagSamples(data);

    // Fill out the confidences of the forest
    finalize(data,forestConfidences,forestOutOfBagConfidences,forestOutOfBagVoteNum);

    if (m_hp.verbose)
    {
        cout << "Trained a tree with " << m_rootNode->numNodes() << " nodes." << endl;
        cout << "Training error = " << computeError(labels) << ", in bag = ";
        cout << computeError(labels, m_inBagSamples) << ", out of bag = ";
        cout << computeError(labels, m_outOfBagSamples) <<  endl;
    }
}


void Tree::train(const matrix<float>& data, const std::vector<int>& labels)
{
    // Initialize
    initialize(m_hp.numLabeled);

    // Random Subsamples data according to bagratio
    subSample(m_hp.numLabeled);

    // Train the root Node
    m_rootNode->train(data, labels, m_inBagSamples, m_confidences, m_predictions);

    evalOutOfBagSamples(data);

    if (m_hp.verbose)
    {
        cout << "Trained a tree with " << m_rootNode->numNodes() << " nodes." << endl;
        cout << "Training error = " << computeError(labels) << ", in bag = ";
        cout << computeError(labels, m_inBagSamples) << ", out of bag = ";
        cout << computeError(labels, m_outOfBagSamples) <<  endl;
    }
}

void Tree::trainLU(const matrix<float>& data, const std::vector<int>& labels)
{
    // Initialize
    initialize(m_hp.numLabeled);

    // Random Subsamples data according to bagratio
    subSample(m_hp.numLabeled);

    // Train the root Node
    m_rootNode->trainLU(data, labels, m_inBagSamples, m_confidences, m_predictions);
    evalOutOfBagSamples(data);

    if (m_hp.verbose)
    {
        cout << "Trained a tree with " << m_rootNode->numNodes() << " nodes." << endl;
        cout << "Training error = " << computeError(labels) << ", in bag = ";
        cout << computeError(labels, m_inBagSamples) << ", out of bag = ";
        cout << computeError(labels, m_outOfBagSamples) <<  endl;
    }
}

void Tree::train(const matrix<float>& data, const std::vector<int>& labels, const std::vector<double>& weights)
{
    // Initialize
    initialize(m_hp.numLabeled);

    // Random Subsamples data according to bagratio
    subSample(m_hp.numLabeled);

    // Train the root Node
    m_rootNode->train(data, labels, weights, m_inBagSamples, m_confidences, m_predictions);

    bool refine = false;
    if (refine)
    {
        cout << "refine tree... ";
        std::vector<int> samples(m_hp.numLabeled,0);
        std::vector<int>::iterator sampleIt(samples.begin());
        std::vector<int>::iterator sampleEnd(samples.end());
        long int counter = 0;
        while ( sampleIt != sampleEnd )
        {
            *sampleIt = counter;
            ++counter;
            ++sampleIt;
        }
        m_rootNode->refine(data,labels,weights,samples,m_confidences,m_predictions);
        cout << "done." << endl;
    }
    evalOutOfBagSamples(data);

    if (m_hp.verbose)
    {
        cout << "Trained a tree with " << m_rootNode->numNodes() << " nodes." << endl;
        cout << "Training error = " << computeError(labels) << ", in bag = ";
        cout << computeError(labels, m_inBagSamples) << ", out of bag = ";
        cout << computeError(labels, m_outOfBagSamples) <<  endl;
    }
}


void Tree::evalOutOfBagSamples(const matrix<float>& data)
{
    m_rootNode->eval(data, m_outOfBagSamples, m_confidences, m_predictions);
}

void Tree::eval(const matrix<float>& data, const std::vector<int>& labels, matrix<float>& forestConfidences)
{
    // Initialize
    m_confidences.resize(data.size1(), m_hp.numClasses);
    m_predictions.resize(data.size1());

    std::vector<int> sampleIndeces(data.size1());
//    sampleIndeces.reserve(data.size1());
    for (unsigned int i = 0; i < data.size1(); i++)
    {
//        sampleIndeces.push_back(i);
    	sampleIndeces[i]=i;
    }
    m_rootNode->eval(data, sampleIndeces, m_confidences, m_predictions);

    // Fill the forest confidences
    for (unsigned long int nSamp = 0; nSamp < data.size1(); nSamp++)
    {
        if (m_hp.useSoftVoting)
        {
            for (int nClass = 0; nClass < m_hp.numClasses; nClass++)
            {
                forestConfidences(nSamp,nClass) += m_confidences(nSamp,nClass);
            }
        }
        else
        {
            forestConfidences(nSamp, m_predictions[nSamp])++;
        }
    }

    if (m_hp.verbose)
    {
        cout << "Tree test error: " << computeError(labels) << endl;
    }
}

void Tree::eval(const matrix<float>& data, const std::vector<int>& labels)
{
    // Initialize
    m_confidences.resize(data.size1(), m_hp.numClasses);
    m_predictions.resize(data.size1());


    std::vector<int> sampleIndeces(data.size1());
//    std::vector<int> sampleIndeces;
//    sampleIndeces.reserve(data.size1());
    for (unsigned long int i = 0; i < data.size1(); i++)
    {
    	sampleIndeces[i]=i;
//        sampleIndeces.push_back(i);
    }

    m_rootNode->eval(data, sampleIndeces, m_confidences, m_predictions);

    if (m_hp.verbose)
    {
        cout << "Test error: " << computeError(labels) << endl;
    }
}

void Tree::initialize(const long int numSamples)
{
    m_confidences.resize(numSamples, m_hp.numClasses);
    m_predictions.resize(numSamples);

    m_inBagSamples.clear();
    m_outOfBagSamples.clear();
}

void Tree::reInitialize(const long int numSamples)
{
    m_confidences.resize(numSamples, m_hp.numClasses);
    m_predictions.resize(numSamples);
}


void Tree::subSample(const long int numSamples)
{
    if (m_hp.useSubSamplingWithReplacement)
    {
        m_inBagSamples = subSampleWithReplacement(numSamples);
        m_outOfBagSamples = setDiff(m_inBagSamples, numSamples);
    }
    else
    {
        subSampleWithoutReplacement(numSamples, static_cast<int>(floor(numSamples * m_hp.bagRatio)),
                                    m_inBagSamples, m_outOfBagSamples);
    }
}

void Tree::subSample(const long int numSamples, const std::vector<double>& weights)
{
    std::vector<int> tmpInBag;
    if (m_hp.useSubSamplingWithReplacement)
    {
        tmpInBag = subSampleWithReplacement(numSamples);
    }
    else
    {
        subSampleWithoutReplacement(numSamples, static_cast<int>(floor(numSamples * m_hp.bagRatio)),
                                    tmpInBag, m_outOfBagSamples);
    }

    // Weight Trimming
    BOOST_FOREACH(int n, tmpInBag)
    {
        if (weights[n] > 1e-4)
        {
            m_inBagSamples.push_back(n);
        }
    }
    m_outOfBagSamples = setDiff(m_inBagSamples, numSamples);
}

double Tree::computeError(const std::vector<int>& labels, const std::vector<int>& sampleIndeces)
{
    double error = 0.0;
    long int sampleCount = 0;
    BOOST_FOREACH(long int n, sampleIndeces)
    {
        if (sampleCount < (int) labels.size())
        {
            error += (m_predictions[n] != labels[n]) ? 1.0 : 0.0;
        }
        else
        {
            break;
        }

        sampleCount++;
    }
    return error/(double) labels.size();
}

double Tree::computeError(const std::vector<int>& labels)
{
    double error = 0.0;
    std::vector<int>::const_iterator itr(m_predictions.begin());
    std::vector<int>::const_iterator labelItr(labels.begin()), labelEnd(labels.end());
    for (; labelItr != labelEnd; itr++, labelItr++)
    {
        error += (*itr != *labelItr) ? 1.0 : 0.0;
    }
    return error/(double) labels.size();
}

void Tree::getTreeAsMatrix(boost::numeric::ublas::matrix<float> *data, const int tree_index)
{
    getTreeAsMatrixRecursive(m_rootNode, data, tree_index, 0);
}

void Tree::getTreeAsMatrixRecursive(Node::Ptr current_node, boost::numeric::ublas::matrix<float> *data,
                                    const int tree_index, const int node_index)
{

    // fill matrix row with current node data
    long int last_index = 0;

    // treeIndex
    (*data)(node_index, last_index++) = (float)tree_index;
    // nodeIndex
    (*data)(node_index, last_index++) = (float)node_index;
    // isTerminal
    if (current_node->isLeaf())
        (*data)(node_index, last_index++) = 1.0f;
    else
        (*data)(node_index, last_index++) = 0.0f;
    // feature indices and weights
    if (!current_node->isLeaf())
    {
        if (m_hp.useRandProj)
        {
            for (int i = 0; i < m_hp.numProjFeatures; i++)
            {
                (*data)(node_index, last_index++) = (float)current_node->bestFeature()[i];
                (*data)(node_index, last_index++) = (float)current_node->bestWeight()[i];
            }
        }
        else
        {
            (*data)(node_index, last_index++) = (float)current_node->bestFeature()[0];
            (*data)(node_index, last_index++) = (float)current_node->bestWeight()[0];
        }
    }
    else
    {
        if (m_hp.useRandProj)
        {
            for (int i = 0; i < m_hp.numProjFeatures; i++)
            {
                (*data)(node_index, last_index++) = -1.0f;
                (*data)(node_index, last_index++) = -1.0f;
            }
        }
        else
        {
            (*data)(node_index, last_index++) = -1.0f;
            (*data)(node_index, last_index++) = -1.0f;
        }
    }
    // threshold
    (*data)(node_index, last_index++) = current_node->bestThreshold();
    // confidences
    if (current_node->isLeaf())
    {
        std::vector<float> confidences = current_node->nodeConf();
        if (confidences.size() != (unsigned int) m_hp.numClasses)
            throw("the number of confidences stored doesn't equal the number of classes");
        for (int i = 0; i < m_hp.numClasses; i++)
            (*data)(node_index, last_index++) = confidences[i];
        // prediction
        (*data)(node_index, last_index++) = (float) current_node->nodeLabel();
    }
    else
    {
        for (int i = 0; i < m_hp.numClasses; i++)
            (*data)(node_index, last_index++) = -1.0f;
    }


    // if necessary, reinvoke function for the child nodes
    if (!current_node->isLeaf())
    {
        getTreeAsMatrixRecursive(current_node->leftChildNode(), data, tree_index, node_index * 2 + 1);
        getTreeAsMatrixRecursive(current_node->rightChildNode(), data, tree_index, node_index * 2 + 2);
    }
}

std::vector<std::vector<int> > Tree::getPath(const matrix<float>& data)
{
    std::vector<std::vector<int> > path(data.size1());
    std::vector<int> sampleIndeces;
    sampleIndeces.reserve(data.size1());
    for (unsigned long int i = 0; i < data.size1(); i++)
    {
        sampleIndeces.push_back(i);
    }
    m_rootNode->getPath(data, sampleIndeces, path);

    return path;
}

int Tree::getDepth()
{
    int treeDepth = m_rootNode->getDepth(0);
    return ++treeDepth;
}

std::vector<int> Tree::getNodeLabels()
{
    std::set<int> nodeLabels;
    m_rootNode->getLabels(nodeLabels);

    std::vector<int> out;
    std::set<int>::const_iterator itr = nodeLabels.begin(), end = nodeLabels.end();
    for (; itr != end; itr++)
    {
        out.push_back(*itr);
    }

    return out;
}
