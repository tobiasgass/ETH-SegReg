#include <iostream>
#include <fstream>
#include "forest.h"
#include "pairforest.h"
#include "tree.h"
#include "data.h"
#include "utilities.h"
#include "hyperparameters.h"
#include <libconfig.h++>
#include <cstdlib>
#include <libxml/tree.h>
#include <libxml/parser.h>

using namespace std;
using namespace libconfig;
//namespace po = boost::program_options;

void printUsage()
{
    cout << "Usage: randomForest <config.conf> <option>" << endl;
    cout << "Options: " << "-train -test -trainAndTest" << endl;
}

void printTestUsage()
{
    cout << "Usage: randomForest <config.conf> -test <classifier.xml>" << endl;
}

HyperParameters parseHyperParameters(const std::string& confFile)
{
    HyperParameters hp;
    Config configFile;

    configFile.readFile(confFile.c_str());

    // DATA
    hp.trainData = (const char*) configFile.lookup("Data.trainData");
    hp.trainLabels = (const char*) configFile.lookup("Data.trainLabels");
    hp.testData = (const char*) configFile.lookup("Data.testData");
    hp.testLabels = (const char*) configFile.lookup("Data.testLabels");
    hp.numLabeled = configFile.lookup("Data.numLabeled");
    hp.numClasses = configFile.lookup("Data.numClasses");

    // TREE
    hp.maxTreeDepth = configFile.lookup("Tree.maxDepth");
    hp.bagRatio = configFile.lookup("Tree.bagRatio");
    hp.numRandomFeatures = configFile.lookup("Tree.numRandomFeatures");
    hp.numProjFeatures = configFile.lookup("Tree.numProjFeatures");
    hp.useRandProj = configFile.lookup("Tree.useRandProj");
    hp.useGPU = configFile.lookup("Tree.useGPU");
    hp.useSubSamplingWithReplacement = configFile.lookup("Tree.subSampleWR");
    hp.verbose = configFile.lookup("Tree.verbose");
    hp.useInfoGain = configFile.lookup("Tree.useInfoGain");


    // FOREST
    hp.numTrees = configFile.lookup("Forest.numTrees");
    hp.useSoftVoting = configFile.lookup("Forest.useSoftVoting");
    hp.saveForest = configFile.lookup("Forest.saveForest");

    // OUTPUT
    hp.saveName = (const char *) configFile.lookup("Output.saveName");
    hp.savePath = (const char *) configFile.lookup("Output.savePath");
    hp.loadName = (const char *) configFile.lookup("Output.loadName");

    return hp;
}

void saveForest(const Forest& forest, const std::string& filename )
{
    const xmlNodePtr rootNode = forest.save();
    xmlDocPtr doc = xmlNewDoc( reinterpret_cast<const xmlChar*>( "1.0" ) );
    xmlDocSetRootElement( doc, rootNode );
    xmlSaveFormatFileEnc( filename.c_str(), doc, "UTF-8", 1 );
    xmlFreeDoc( doc );
}

int main(int argc, char** argv)
{
    // at least, we need two arguments
    if (argc < 2) {
      cout << "----------------------------------------------------------------------------------------------------" << endl;
      cout << "Illegal arguments specified. Your options:" << endl;
      cout << "----------------------------------------------------------------------------------------------------" << endl;
      printUsage();
      printTestUsage();
      cout << "----------------------------------------------------------------------------------------------------" << endl;
      return -1;
    }

    // Train Random Forest
    if (argc == 3 && (std::string(argv[2]) == "-train"))
    {
        const std::string confFile(argv[1]);
        HyperParameters hp = parseHyperParameters(confFile);
        Forest rf(hp);
        //Tree t(hp);

        FileData fileData(hp.trainData, hp.trainLabels);
        fileData.readData();
        fileData.readLabels();

        rf.train(fileData.getData(),fileData.getLabels());

    }
    // Test Random Forest
    else if (std::string(argv[2]) == "-test")
    {
        if (argc == 4)
        {
            const std::string confFile(argv[1]);
            HyperParameters hp = parseHyperParameters(confFile);
            Forest rf(hp,argv[3]);

            FileData testData(hp.testData, hp.testLabels);
            testData.readData();
            testData.readLabels();

            rf.eval(testData.getData(),testData.getLabels());
        }
        else
        {
            printTestUsage();
        }
    }
    // Train a Tree
    else if (argc == 3 && (std::string(argv[2]) == "-trainAndTestTree"))
    {
        const std::string confFile(argv[1]);
        HyperParameters hp = parseHyperParameters(confFile);

        Tree rt(hp);

        timeIt(1);
        FileData trainData(hp.trainData, hp.trainLabels);
        trainData.readData();
        trainData.readLabels();

        rt.train(trainData.getData(),trainData.getLabels());
        cout << "\tTraining time = " << timeIt(0) << " seconds" << endl;

        std::vector<int> nodeLabels = rt.getNodeLabels();
        cout << "Classes which has at least a leaf node ... " << endl;
        dispVector(nodeLabels);

        timeIt(1);
        FileData testData(hp.testData, hp.testLabels);
        testData.readData();
        testData.readLabels();
        rt.eval(testData.getData(),testData.getLabels());
        cout << "\tTest time = " << timeIt(0) << " seconds" << endl;
    }
    // Train and Test Random Forest
    else if (argc == 3 && (std::string(argv[2]) == "-trainAndTest"))
    {
        const std::string confFile(argv[1]);
        HyperParameters hp = parseHyperParameters(confFile);

        Forest rf(hp);

        timeIt(1);
        FileData trainData(hp.trainData, hp.trainLabels);
        trainData.readData();
        trainData.readLabels();



        rf.train(trainData.getData(),trainData.getLabels());
        cout << "\tTraining time = " << timeIt(0) << " seconds" << endl;

        timeIt(1);
        FileData testData(hp.testData, hp.testLabels);
        testData.readData();
        testData.readLabels();
        rf.eval(testData.getData(),testData.getLabels());
        cout << "\tTest time = " << timeIt(0) << " seconds" << endl;
        if (hp.saveForest) {
            saveForest(rf,"testforest.xml");
        }
    }

    // Train and Test Random Forest
    else if (argc == 3 && (std::string(argv[2]) == "-trainPairsAndTest"))
    {
        const std::string confFile(argv[1]);
        HyperParameters hp = parseHyperParameters(confFile);

        PairForest rf(hp);

        timeIt(1);
        FileData trainData(hp.trainData, hp.trainLabels);
        trainData.readData();
        trainData.readLabels();
        trainData.createPairs();

        rf.train( trainData.getPairs() );
        cout << "\tTraining time = " << timeIt(0) << " seconds" << endl;
        cout << "\tTrain result: ";
        rf.test( trainData.getPairs() );


        timeIt(1);
        FileData testData(hp.testData, hp.testLabels);
        testData.readData();
        testData.readLabels();
        testData.createPairs();
        cout << "\tTest result: ";
        rf.test( testData.getPairs() );

        cout << "\tTest time = " << timeIt(0) << " seconds" << endl;

    }


    return 0;
}

