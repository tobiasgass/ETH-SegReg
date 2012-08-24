#include "forest.h"
#include <string.h>
#include <fstream>
#include <boost/foreach.hpp>
#include "time.h"
#include <libxml/tree.h>
#include <libxml/parser.h>

Forest::Forest(const HyperParameters &hp)
{
    m_hp = hp;
#ifdef USE_CUDA
    m_forest_d = NULL;
#endif

}

Forest::Forest(const HyperParameters &hp, const std::string& forestFilename ) :
        m_hp( hp )
{
    xmlDocPtr forestDoc = xmlParseFile( forestFilename.c_str() );
    if ( !forestDoc )
    {
        cout << "ERROR: no forest.xml file or wrong filename" << endl;
        return;
    }

    xmlNodePtr root = xmlDocGetRootElement( forestDoc );
    if ( !root )
    {
        xmlFreeDoc( forestDoc );
        cout << "ERROR: no forest.xml file or wrong filename" << endl;
        return;
    }

    if ( xmlStrcmp( root->name, reinterpret_cast<const xmlChar*>( "randomforest" ) ) != 0 )
    {
        cout << "ERROR: no forest.xml file or wrong filename" << endl;
        cerr << "This doesn't seem to be classifier file..." << endl;
        xmlFreeDoc( forestDoc );
        return;
    }

    xmlNodePtr cur = root->xmlChildrenNode;
    while ( cur != 0 )
    {
        if ( xmlStrcmp( cur->name, reinterpret_cast<const xmlChar*>( "tree" ) ) == 0 )
        {
            m_trees.push_back( Tree( m_hp, cur ) );
        }
        cur = cur->next;
    }
    xmlFreeDoc( forestDoc );
}

void Forest::initialize(const long int numSamples)
{

  m_confidences.resize(numSamples, m_hp.numClasses);
  m_predictions.resize(numSamples);

  for (long int n = 0; n < numSamples; n++)
  {
    for (int m = 0; m < m_hp.numClasses; m++)
    {
        m_confidences(n, m) = 0;
    }
  }
}

xmlNodePtr Forest::save() const
{
    xmlNodePtr node = xmlNewNode( NULL, reinterpret_cast<const xmlChar*>( "randomforest" ) );
    addIntProp( node, "numTrees", m_trees.size() );

    //xmlAddChild( node, Configurator::conf()->saveConstants() );

    int i = 0;
    BOOST_FOREACH(Tree t, m_trees) {
      xmlNodePtr stageNode = t.save();
      addIntProp( stageNode, "num", i );
      xmlAddChild( node, stageNode );
      i++;
    }

    return node;
}

void Forest::train(const matrix<float>& data, const std::vector<int>& labels, bool use_gpu)
{
    // Initialize

    initialize(data.size1());

    if (m_hp.useGPU || use_gpu){
        std::vector<double> weights(labels.size(),1);
        trainByGPU(data,labels, weights);
    }
    else {
        trainByCPU(data,labels);
    }
}
void Forest::train(const matrix<float>& data, const std::vector<int>& labels, float weight,bool use_gpu)
{
    // Initialize
    initialize(data.size1());

    if (m_hp.useGPU || use_gpu){
        std::vector<double> weights(labels.size(),weight);
        trainByGPU(data,labels, weights);
    }
    else {
        trainByCPU(data,labels);
    }
}

void Forest::train(const matrix<float>& data, const std::vector<int>& labels, const std::vector<double>& weights, bool use_gpu)
{
    // Initialize
    initialize(data.size1());

    if (m_hp.useGPU || use_gpu)
        trainByGPU(data,labels, weights);
    else
      trainByCPU(data,labels,weights);
}

Forest::~Forest()
{
    m_trees.clear();
#ifdef USE_CUDA
    if (m_forest_d != NULL)
        delete m_forest_d;
#endif
}

void Forest::eval(const matrix<float>& data, const std::vector<int>& labels, bool use_gpu)
{
  if (m_hp.useGPU || use_gpu)
    evalByGPU(data, labels);
  else
    evalByCPU(data, labels);
}

void Forest::save(const std::string &name)
{
    std::string saveName;
    saveName = (name == "default") ? m_hp.saveName : name;

    const xmlNodePtr rootNode = this->save();
    xmlDocPtr doc = xmlNewDoc( reinterpret_cast<const xmlChar*>( "1.0" ) );
    xmlDocSetRootElement( doc, rootNode );
    xmlSaveFormatFileEnc( saveName.c_str(), doc, "UTF-8", 1 );
    xmlFreeDoc( doc );
    // now save that stuff

}

void Forest::load(const std::string &name)
{
    std::string loadName;
    loadName = (name == "default") ? m_hp.loadName : name;

    // now load that stuff
    xmlDocPtr forestDoc = xmlParseFile( loadName.c_str() );
    if ( !forestDoc )
    {
        cout << "ERROR: no forest.xml file or wrong filename" << endl;
        return;
    }

    xmlNodePtr root = xmlDocGetRootElement( forestDoc );
    if ( !root )
    {
        xmlFreeDoc( forestDoc );
        cout << "ERROR: no forest.xml file or wrong filename" << endl;
        return;
    }

    if ( xmlStrcmp( root->name, reinterpret_cast<const xmlChar*>( "randomforest" ) ) != 0 )
    {
        cout << "ERROR: no forest.xml file or wrong filename" << endl;
        cerr << "This doesn't seem to be classifier file..." << endl;
        xmlFreeDoc( forestDoc );
        return;
    }

    xmlNodePtr cur = root->xmlChildrenNode;
    while ( cur != 0 )
    {
        if ( xmlStrcmp( cur->name, reinterpret_cast<const xmlChar*>( "tree" ) ) == 0 )
        {
            m_trees.push_back( Tree( m_hp, cur ) );
        }
        cur = cur->next;
    }
    xmlFreeDoc( forestDoc );
}

#ifdef USE_CUDA
void Forest::createForestMatrix()
{
  // DEPRECATED !!
    // create a forest matrix for evaluation on the gpu
    int num_cols = 0;
    if (m_hp.useRandProj)
        num_cols = 3 + 2 * m_hp.numProjFeatures + 1 + m_hp.numClasses + 1;
    else
        num_cols = 3 + 2 + 1 + m_hp.numClasses + 1;

    int num_nodes_tree = (int)pow((float)2,m_hp.maxTreeDepth + 1) - 1;
    int num_rows = num_nodes_tree;

    float *forest_host = new float[num_cols * m_hp.numTrees * num_rows];

    if (m_hp.numTrees != m_trees.size())
        throw("The number of trees trained is different from the number configured");

    std::vector<Tree>::iterator tree_it = m_trees.begin();

    for (int tree_index = 0; tree_index < m_hp.numTrees; tree_index++) {
        matrix<float> tree_matrix(num_nodes_tree, num_cols, -1.0f);
        tree_it->getTreeAsMatrix(&tree_matrix, tree_index);

        // copy data
        for (int row = 0; row < num_rows; row++) {
            forest_host[row * m_hp.numTrees * num_cols + tree_index * num_cols] = (float) tree_index;
            for (int col = 1; col < num_cols; col++) {
                forest_host[row * m_hp.numTrees * num_cols + tree_index * num_cols + col] = tree_matrix(row, col);
            }
        }
        tree_it++;
    }

    if (m_hp.verbose)
        std::cout << "\ncreateForestMatrix(): forest matrix needs " << num_cols * m_hp.numTrees * num_rows * 4 / 1024 << " KB of memory\n";

    Cuda::HostMemoryReference<float, 2> forest_hmr(Cuda::Size<2>(num_cols * m_hp.numTrees, num_rows), forest_host);
    // DEBUG
    /*for (int row = 0; row < num_rows; row++) {
        for (int col = 0; col < num_cols * m_hp.numTrees; col++) {
            if (col%num_cols == 0)
                printf("\n");
            printf("%1.1f ", forest_host[row * num_cols * m_hp.numTrees + col]);
        }
        printf("\n");
    }*/

    //std::cout << "\n dimensions: " << forest_hmr.size[0] << "x" << forest_hmr.size[1] <<"\n";
    //std::cout << " max texture size: " << (int)pow(2.0f,16) << "x" << (int)pow(2.0f,15) <<"\n";
    if (forest_hmr.size[0] > pow(2.0f,14) || forest_hmr.size[1] > pow(2.0f,15)) {
        std::cout << "\nForest is larger than maximum cuda texture dimensions: \n";
        std::cout << "forest: w: " << forest_hmr.size[0] << ", h: " << forest_hmr.size[1];
        std::cout << " maximum texture size: w: " << pow(2.0f,14) << ", h: " << pow(2.0f,15);

    }
    if (m_forest_d != NULL)
        delete m_forest_d;
    m_forest_d = new Cuda::Array<float,2>(forest_hmr);

    delete[] forest_host;
}
#endif

// CPU penne code only below this line
void Forest::trainByCPU(const matrix<float>& data, const std::vector<int>& labels)
{
    std::vector<int> outOfBagVoteCount(m_hp.numLabeled, 0);
    matrix<float> outOfBagConfidences(m_hp.numLabeled,m_hp.numClasses);
    for (int i = 0; i < m_hp.numLabeled; i++)
    {
        for ( int j = 0; j < m_hp.numClasses; j++)
        {
            outOfBagConfidences(i,j) = 0.0;
        }
    }

    std::vector<int>::iterator preItr, preEnd;
    HyperParameters tmpHP = m_hp;
    tmpHP.verbose = false;

    if (m_hp.verbose)
    {
        cout << "Training a random forest with " << m_hp.numTrees << " , grab a coffee ... " << endl;
        cout << "\tTree #: ";
    }

    m_trees.clear();
    for (int i = 0; i < m_hp.numTrees; i++)
    {
        if (m_hp.verbose && !(10*i%m_hp.numTrees))
        {
            cout << 100*i/m_hp.numTrees << "% " << flush;
        }

        Tree t(tmpHP);
        t.train(data,labels, m_confidences, outOfBagConfidences, outOfBagVoteCount);
        m_trees.push_back(t);
    }

    if (m_hp.verbose)
    {
        cout << " Done!" << endl;
    }

    double error = computeError(labels);
    double outOfBagError = computeError(labels, outOfBagConfidences, outOfBagVoteCount);

    if (m_hp.verbose)
    {
        cout << "\tTraining error = " << error << ", out of bag error = " << outOfBagError << endl;
    }
}

void Forest::trainAndTest1vsAll(const matrix<float>& dataTr, const std::vector<int>& labelsTr,
                                const matrix<float>& dataTs, const std::vector<int>& labelsTs)
{
  int trueNumClasses = m_hp.numClasses;
  matrix<float> testProb(dataTs.size1(), trueNumClasses);
  m_hp.numClasses = 2;
  bool verbose = m_hp.verbose;
  m_hp.verbose = false;

  // 1 vs All loop
  if (verbose) {
    cout << "Training and testing with 1-vs-all for " << trueNumClasses << " classes ..." << endl;
  }

  std::vector<int> tmpLabelsTr(dataTr.size1()), tmpLabelsTs(dataTs.size1());
  matrix<float> tmpConf;
  double error;
  for (int nClass = 0; nClass < trueNumClasses; nClass++) {
    if (verbose) {
      cout << "\tClass #: " << nClass + 1 << "  ... ";
    }
    initialize(dataTr.size1());
    std::vector<double> tmpWeights(dataTr.size1(), 1.0);

    for (int nSamp = 0; nSamp < (int) dataTr.size1(); nSamp++) {
      if (labelsTr[nSamp] == nClass) {
        tmpLabelsTr[nSamp] = 1;
        tmpWeights[nSamp] = trueNumClasses;
      }
      else {
        tmpLabelsTr[nSamp] = 0;
      }
    }
    if (m_hp.useGPU)
        trainByGPU(dataTr, tmpLabelsTr, tmpWeights);
    else
        trainByCPU(dataTr, tmpLabelsTr, tmpWeights);


    for (int nSamp = 0; nSamp < (int) dataTs.size1(); nSamp++) {
      if (labelsTs[nSamp] == nClass) {
        tmpLabelsTs[nSamp] = 1;
      }
      else {
        tmpLabelsTs[nSamp] = 0;
      }
    }
    if (m_hp.useGPU)
      evalByGPU(dataTs, tmpLabelsTs);
    else
      evalByCPU(dataTs, tmpLabelsTs);

    error = computeError(tmpLabelsTs);
    if (verbose) {
      cout << " Classification error = " << error << endl;
    }

    for (int nSamp = 0; nSamp < (int) dataTs.size1(); nSamp++) {
      testProb(nSamp, nClass) = m_confidences(nSamp, 1);
    }
  }

  // Make the decision
  int bestClass;
  double bestConf, testError = 0;
  std::vector<double> preCount(trueNumClasses, 0), trueCount(trueNumClasses, 0);
  for (int nSamp = 0; nSamp < (int) dataTs.size1(); nSamp++) {
    bestClass = 0;
    bestConf = 0;
    for (int nClass = 0; nClass < trueNumClasses; nClass++) {
      if (bestConf < testProb(nSamp, nClass)) {
        bestConf = testProb(nSamp, nClass);
        bestClass = nClass;
      }
    }

    m_predictions[nSamp] = bestClass;

    if (labelsTs[nSamp] != bestClass) {
      testError++;
    }

    preCount[bestClass]++;
    trueCount[labelsTs[nSamp]]++;
  }

  for (int n = 0; n < trueNumClasses; n++) {
    preCount[n] /= (trueCount[n] + 1e-6);
  }

  if (verbose)
    {
      cout << "\tForest Test Error = " << testError/((double) dataTs.size1()) << endl;
      cout << "\tPrediction ratio: ";
      dispVector(preCount);
    }

  m_hp.numClasses = trueNumClasses;
}

void Forest::trainByCPU(const matrix<float>& data, const std::vector<int>& labels, const std::vector<double>& weights)
{

    std::vector<int> outOfBagVoteCount(data.size1(), 0);
    matrix<float> outOfBagConfidences(data.size1(),m_hp.numClasses);
    for ( unsigned int i = 0; i < data.size1(); i++)
    {
        for ( int j = 0; j < m_hp.numClasses; j++)
        {
            outOfBagConfidences(i,j) = 0.0;
        }
    }

    HyperParameters tmpHP = m_hp;
    tmpHP.verbose = false;

    if (m_hp.verbose)
    {
        cout << "Training a random forest with " << m_hp.numTrees << " trees, grab a coffee ... " << endl;
        cout << "\tTree #: ";
    }

    m_trees.clear();
    for (int i = 0; i < m_hp.numTrees; i++)
    {
        if (m_hp.verbose && !(10*i%m_hp.numTrees))
        {
            cout << 100*i/m_hp.numTrees << "% " << flush;
        }

        Tree t(tmpHP);
        t.train(data,labels, weights, m_confidences, outOfBagConfidences, outOfBagVoteCount);
        m_trees.push_back(t);
    }
    if (m_hp.verbose)
    {
        cout << " Done!" << endl;
    }


    double error = computeError(labels);
    double outOfBagError = computeError(labels, outOfBagConfidences, outOfBagVoteCount);

    if (m_hp.verbose)
    {
        cout << "\tTraining error = " << error << ", out of bag error = " << outOfBagError << endl;
    }
}

void Forest::evalByCPU(const matrix<float>& data, const std::vector<int>& labels)
{
//	clock_t start = clock();

    // Initialize
    initialize(data.size1());

    BOOST_FOREACH(Tree t, m_trees) {
      t.eval(data, labels, m_confidences);
    }
//    clock_t trees = clock();

    // divide confidences by number of trees
    m_confidences *= (1.0f / m_hp.numTrees);

    int bestClass;
    double bestConf;

    for (long int nSamp = 0; nSamp < (long int) data.size1(); nSamp++) {
      bestConf = 0;
      bestClass = 0;
      for(int nClass = 0; nClass < m_hp.numClasses; nClass++) {
        if (m_confidences(nSamp, nClass) > bestConf) {
          bestConf = m_confidences(nSamp, nClass);
          bestClass = nClass;
        }
      }

      m_predictions[nSamp] = bestClass;
    }
//	clock_t conf = clock();
    double error = computeError(labels);
//    clock_t errorTime = clock();

//    if (m_hp.verbose)
    {
//    	float total=(float) (((double)(errorTime- start)) / CLOCKS_PER_SEC);
//    	float treeTime=(float) (((double)(trees- start)) / CLOCKS_PER_SEC);
//    	float predict=(float) (((double)(conf- trees)) / CLOCKS_PER_SEC);
//    	float err=(float) (((double)(errorTime- conf)) / CLOCKS_PER_SEC);
//    	cout<<"total:"<<total<<" trees:"<<treeTime<<" prediction"<<predict<<" errors:"<<err<<endl;
        cout << "\tForest test error = " << error << endl;
    }
}

double Forest::computeError(const std::vector<int>& labels)
{
    int bestClass, nSamp = 0;
    float bestConf;
    double error = 0;
    BOOST_FOREACH(int pre, m_predictions) {
      bestClass = 0;
      bestConf = 0;
      for (int nClass = 0; nClass < (int) m_hp.numClasses; nClass++)
        {
          if (m_confidences(nSamp, nClass) > bestConf)
            {
              bestClass = nClass;
              bestConf = m_confidences(nSamp, nClass);
            }
        }

      pre = bestClass;
      if (bestClass != labels[nSamp])
        {
          error++;
        }

      nSamp++;
    }
    error /= (double) m_predictions.size();

    return error;
}

double Forest::computeError(const std::vector<int>& labels, const matrix<float>& confidences,
                            const std::vector<int>& voteNum)
{
    double error = 0, bestConf;
    int sampleNum = 0, bestClass;
    for (int nSamp = 0; nSamp < (int) confidences.size1(); nSamp++)
    {
        bestClass = 0;
        bestConf = 0;
        if (voteNum[nSamp])
        {
            sampleNum++;
            for (int nClass = 0; nClass < (int) m_hp.numClasses; nClass++)
            {
                if (confidences(nSamp, nClass) > bestConf)
                {
                    bestClass = nClass;
                    bestConf = confidences(nSamp, nClass);
                }
            }

            if (bestClass != labels[nSamp])
            {
                error++;
            }
        }
    }
    error /= (double) sampleNum;

    return error;
}


// GPU spaghetti code only below this line
void Forest::trainByGPU(const matrix<float>& data, const std::vector<int>& labels, const std::vector<double>& weights)
{
#ifdef USE_CUDA
    // test parameters
    if (!m_hp.useRandProj)
        m_hp.numProjFeatures = 1;

    // copy input data to device
    size_t cols = data.size2();
    size_t rows = data.size1();
    Cuda::HostMemoryReference<float, 2> data_hmr(Cuda::Size<2>(cols, rows), (float*)&data.data()[0]);
    Cuda::Array<float, 2> data_dmp(data_hmr);

    // copy labels to device
    Cuda::HostMemoryHeap<int, 1> labels_hmr(Cuda::Size<1>(labels.size()));
    Cuda::HostMemoryHeap<float, 1> weights_hmr(Cuda::Size<1>(labels.size()));
    for (int i = 0; i < labels.size(); i++){
        labels_hmr.getBuffer()[i] = labels[i];
        weights_hmr.getBuffer()[i] = weights[i];
    }
    Cuda::DeviceMemoryLinear<int, 1> labels_dml(labels_hmr);
    Cuda::DeviceMemoryLinear<float, 1> weights_dml(weights_hmr);

    // create forest structure
    int num_cols = iGetNumTreeCols(m_hp.numProjFeatures, m_hp.numClasses);
    Cuda::DeviceMemoryPitched<float,2>* forest_dmp;

    float oobe;
    iTrainForest(data_dmp, labels_dml, weights_dml, &forest_dmp, &oobe, data_dmp.size[1],
        m_hp.numClasses, m_hp.numTrees, m_hp.maxTreeDepth, num_cols, m_hp.numRandomFeatures, m_hp.numProjFeatures, m_hp.bagRatio);

    if (m_forest_d != NULL)
        delete m_forest_d;
    //printf("forest needs %3.3f KB of memory", forest_dmp->stride[0] * forest_dmp->size[1] * sizeof(float) / 1024.0f);
    printf(" oobe: %3.3f ",oobe);
    m_forest_d = new Cuda::Array<float,2>(*forest_dmp);
    delete forest_dmp; // FIXXME, one could directly pass the array
#endif
}

void Forest::evalByGPU(const matrix<float>& data, const std::vector<int>& labels)
{
#ifdef USE_CUDA
    if (!m_hp.useRandProj)
        m_hp.numProjFeatures = 1;

    // initialize
    initialize(data.size1());

    // copy input data to device
    size_t cols = data.size2();
    size_t rows = data.size1();
    Cuda::HostMemoryReference<float, 2> data_hmr(Cuda::Size<2>(cols, rows), (float*)&data.data()[0]);
    Cuda::Array<float, 2> data_dmp(Cuda::Size<2>(cols, rows));
    Cuda::copy(data_dmp, data_hmr);

    // create output structures
    size_t num_samples = rows;
    Cuda::DeviceMemoryPitched<float,2> confidences_dmp(Cuda::Size<2>(num_samples, m_hp.numClasses));
    Cuda::DeviceMemoryLinear1D<float> predictions_dmp(num_samples);

    bool soft_voting = false;
    if (m_hp.useSoftVoting) soft_voting = true;

    int num_tree_cols = iGetNumTreeCols(m_hp.numProjFeatures, m_hp.numClasses);

    iEvaluateForest(*m_forest_d, data_dmp, &confidences_dmp, &predictions_dmp,
    num_samples, m_hp.numTrees, m_hp.maxTreeDepth, num_tree_cols, m_hp.numClasses, m_hp.numProjFeatures, soft_voting);

    // copy output to host memory
    Cuda::HostMemoryHeap<float,2> confidences_hmh(confidences_dmp);
    Cuda::HostMemoryHeap<float,1> predictions_hmh(predictions_dmp);
    float* confidences_hmhp = confidences_hmh.getBuffer();
    float* predictions_hmhp = predictions_hmh.getBuffer();

    // assign confidences and predictions
    for (unsigned int nSamp = 0; nSamp < num_samples; nSamp++) {
        for (int nClass = 0; nClass < m_hp.numClasses; nClass++)
            m_confidences(nSamp,nClass) = confidences_hmhp[nClass * num_samples + nSamp];
        m_predictions[nSamp] = (int) predictions_hmhp[nSamp];
    }

    double error = computeError(labels);

    if (m_hp.verbose)
    {
        cout << "\tTest error = " << error << endl;
    }
#endif
}

void Forest::writeError(const std::string& fileName, double error)
{
    std::ofstream myfile;
    myfile.open(fileName.c_str(), std::ios::app);
    myfile << error << endl;
    myfile.flush();
    myfile.close();
}

