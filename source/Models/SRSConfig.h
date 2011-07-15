/*
 * config.h
 *
 *  Created on: Apr 12, 2011
 *      Author: gasst
 */

#ifndef SRSCONFIG_H_
#define SRSCONFIG_H_
#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <iterator>
#include "argstream.h"
class SRSConfig{
public:
	std::string targetFilename,movingFilename,fixedGradientFilename, outputDeformedSegmentationFilename,movingSegmentationFilename, outputDeformedFilename,deformableFilename,defFilename, segmentationOutputFilename;
	std::string segmentationProbsFilename, pairWiseProbsFilename;
	double pairwiseRegistrationWeight;
	double pairwiseSegmentationWeight;
	int displacementSampling;
	double unaryWeight;
	int maxDisplacement;
	double simWeight;
	double rfWeight;
	double segWeight;
	int nSegmentations;
	bool verbose;
	int * levels;
	int nLevels;
	int startTiling;
	int iterationsPerLevel;
	bool train;
    double displacementRescalingFactor;
private:
	argstream * as;
public:
	SRSConfig(){
		defFilename="";
		pairwiseRegistrationWeight=1;
		pairwiseSegmentationWeight=1;
		displacementSampling=-1;
		unaryWeight=1;
		maxDisplacement=10;
		simWeight=1;
		rfWeight=1;
		segWeight=1;
		nSegmentations=1;
		verbose=false;
		nLevels=3;
		startTiling=2;
		iterationsPerLevel=4;
		train=false;
		segmentationProbsFilename='segmentation.bin';
		pairWiseProbsFilename='pairwise.bin';
        displacementRescalingFactor=0.5;
	}
    ~SRSConfig(){
		//delete as;
	}
	void parseParams(int argc, char** argv){
		as= new argstream(argc, argv);
		parse();
	}
	void copyFrom(SRSConfig c){
		targetFilename=c.targetFilename;
		movingFilename=c.movingFilename;
		fixedGradientFilename=c.fixedGradientFilename;
		outputDeformedSegmentationFilename=c.outputDeformedSegmentationFilename;
		movingSegmentationFilename=c.movingSegmentationFilename;
		outputDeformedFilename=c.outputDeformedFilename;
		deformableFilename=c.deformableFilename;
		defFilename=c.defFilename;
		segmentationOutputFilename=c.segmentationOutputFilename;
		segmentationProbsFilename=c.segmentationProbsFilename;
		pairWiseProbsFilename=c.pairWiseProbsFilename;

		pairwiseRegistrationWeight=c.pairwiseRegistrationWeight;
		pairwiseSegmentationWeight=c.pairwiseSegmentationWeight;
		displacementSampling=c.displacementSampling;
		unaryWeight=c.unaryWeight;
		maxDisplacement=c.maxDisplacement;
		simWeight=c.maxDisplacement;
		rfWeight=c.rfWeight;
		segWeight=c.rfWeight;
		nSegmentations=c.nSegmentations;
		verbose=c.nSegmentations;
		levels=c.levels;
		nLevels=c.nLevels;
		startTiling=c.startTiling;
		train=c.train;
        displacementRescalingFactor=c.displacementRescalingFactor;
	}
	void parseFile(std::string filename){
		std::ostringstream streamm;
		std::ifstream is(filename.c_str());
		if (!is){
			std::cout<<"could not open "<<filename<<" for reading"<<std::endl;
			exit(10);
		}
		std::string buff;
		is >> buff;
		streamm<<buff;
		while (!is.eof()){
			streamm<<" ";
			is >>buff;
			streamm<<buff;
			//			std::cout<<streamm.str()<<std::endl;
			//			std::cout<<buff<<std::endl;

		}
		as= new argstream(streamm.str().c_str());
		parse();
	}
	void parse(){
		std::string filename="";
		(*as) >> parameter ("configFile", filename, "read config from file, additional command line parameters overwrite config file settings. (filename)", false);
		if (filename!=""){
			SRSConfig fromFile;
			fromFile.parseFile(filename);
			copyFrom(fromFile);
		}

		(*as) >> parameter ("t", targetFilename, "target image (file name)", false);
		(*as) >> parameter ("m", movingFilename, "moving image (file name)", false);
		(*as) >> parameter ("s", movingSegmentationFilename, "moving segmentation image (file name)", false);
		(*as) >> parameter ("g", fixedGradientFilename, "fixed gradient image (file name)", false);

		(*as) >> parameter ("o", outputDeformedFilename, "output image (file name)", false);
		(*as) >> parameter ("S", outputDeformedSegmentationFilename, "output image (file name)", false);
		(*as) >> parameter ("O", segmentationOutputFilename, "output segmentation image (file name)", false);
		(*as) >> parameter ("f", defFilename,"deformation field filename", false);
		(*as) >> parameter ("rp", pairwiseRegistrationWeight,"weight for pairwise registration potentials", false);
		(*as) >> parameter ("sp", pairwiseSegmentationWeight,"weight for pairwise segmentation potentials", false);

		(*as) >> parameter ("u", unaryWeight,"weight for unary potentials", false);
		(*as) >> parameter ("max", maxDisplacement,"maximum displacement in pixels per axis", false);
		(*as) >> parameter ("wi", simWeight,"weight for intensity similarity", false);
		(*as) >> parameter ("wr", rfWeight,"weight for segmentation posterior", false);
		(*as) >> parameter ("ws", segWeight,"weight for segmentation similarity", false);
		(*as) >> parameter ("nLevels", nLevels,"number of multiresolution pyramid levels", false);
		(*as) >> parameter ("startlevel", startTiling,"start tiling", false);
		(*as) >> parameter ("iterationsPerLevel", iterationsPerLevel,"iterationsPerLevel", false);
        (*as) >> parameter ("r",displacementRescalingFactor,"displacementRescalingFactor", false);

		(*as) >> parameter ("segmentationProbs", segmentationProbsFilename,"segmentation probabilities  filename", false);
		(*as) >> parameter ("pairwiseProbs", pairWiseProbsFilename,"pairwise segmentation probabilities filename", false);
		(*as) >> option ("train", train,"train classifier (and save), if not set data will be read from the given files");
		std::vector<int> tmp_levels(6,-1);
		(*as) >> parameter ("l0", tmp_levels[0],"divisor for level 0", false);
		(*as) >> parameter ("l1", tmp_levels[1],"divisor for level 1", false);
		(*as) >> parameter ("l2", tmp_levels[2],"divisor for level 2", false);
		(*as) >> parameter ("l3", tmp_levels[3],"divisor for level 3", false);
		(*as) >> parameter ("l4", tmp_levels[4],"divisor for level 4", false);
		(*as) >> parameter ("l5", tmp_levels[5],"divisor for level 5", false);
		std::list<int> bla;
//		(*as) >> values<int> (back_inserter(bla),"descr",nLevels);
		(*as) >> help();
		as->defaultErrorHandling();
		nSegmentations=2;
		if (segWeight==0 && pairwiseSegmentationWeight==0 && rfWeight==0 ){
			nSegmentations=1;
		}
		if (displacementSampling==-1) displacementSampling=maxDisplacement;
		verbose=false;
		levels=new int[nLevels+1];
		if (tmp_levels[0]==-1){
			levels[0]=startTiling;
		}else{
			levels[0]=tmp_levels[0];
		}
		for (int i=1;i<nLevels+1;++i){
			if (tmp_levels[i]==-1){
				levels[i]=levels[i-1]*2-1;
			}else{
				levels[i]=tmp_levels[i];
			}

		}

	}
};
#endif /* CONFIG_H_ */
