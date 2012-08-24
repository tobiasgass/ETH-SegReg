#include "nodehyperplane.h"
#include "nodegini.h"
#include "nodeinfogain.h"
#include "utilities.h"
#include <boost/foreach.hpp>
#include <algorithm>
#include <boost/lexical_cast.hpp>

#ifdef WIN32
inline double round(double x)
{
	return (x-floor(x))>0.5 ? ceil(x) : floor(x);
}
#endif
using namespace std;

NodeHyperPlane::NodeHyperPlane(const HyperParameters &hp, int depth) : Node(hp, depth), m_bestThreshold( 0.0 )
{
}

NodeHyperPlane::NodeHyperPlane(const HyperParameters &hp, int depth, int reset) : Node(hp, depth, reset), m_bestThreshold( 0.0 )
{
}

NodeHyperPlane::NodeHyperPlane(const HyperParameters &hp, int reset, const xmlNodePtr nodeNode) : Node(hp,0,reset)
{
	m_isLeaf = (readStringProp(nodeNode,"isLeaf") == "true") ? true : false;

	if (m_isLeaf)
	{
		m_nodeLabel = readIntProp( nodeNode, "label", 0 );
		xmlNodePtr cur = nodeNode->xmlChildrenNode;
		while ( cur != 0 )
		{
			if ( xmlStrcmp( cur->name, reinterpret_cast<const xmlChar*>( "confidence" ) ) == 0 )
			{
				m_nodeConf.push_back( static_cast<float>(readDoubleProp( cur, "conf", 0 )) );
			}
			cur = cur->next;
		}
	}
	else
	{
		xmlNodePtr cur = nodeNode->xmlChildrenNode;
		while ( cur != 0 )
		{
			if ( xmlStrcmp( cur->name, reinterpret_cast<const xmlChar*>( "feature" ) ) == 0 )
			{
                for (int i=0;i<m_hp.numRandomFeatures;++i){
                    string s = "feat"+boost::lexical_cast<string>( i );
                    int tmp = readIntProp( cur, s.c_str(), 0 );
                    m_bestFeatures[i]=tmp;
                    string s2 = "weight"+boost::lexical_cast<string>( i );
                    double tmpW=(float)readDoubleProp(cur,s2.c_str(), 0);
                    m_bestWeights[i]=tmpW;
                }
				
				m_bestThreshold = (float)readDoubleProp( cur, "threshold", 0 );
			}
			else if ( xmlStrcmp( cur->name, reinterpret_cast<const xmlChar*>( "node" ) ) == 0 )
			{
				const std::string childNode = readStringProp(cur,"child");
				if ( childNode == "left" )
				{
					const std::string type = readStringProp(cur,"type");
					if (type == NODE_GINI)
					{
						m_leftChildNode = NodeGini::Ptr(new NodeGini(m_hp,-1,cur));
					}
					else if (type == NODE_INFO_GAIN)
					{
						m_leftChildNode = NodeInfoGain::Ptr(new NodeInfoGain(m_hp,-1, cur));
					}
				}
				else
				{
					const std::string type = readStringProp(nodeNode,"type");
					if (type == NODE_GINI)
					{
						m_rightChildNode = NodeGini::Ptr(new NodeGini(m_hp,-1,cur));
					}
					else if (type == NODE_INFO_GAIN)
					{
						m_rightChildNode = NodeInfoGain::Ptr(new NodeInfoGain(m_hp,-1,cur));
					}
				}
			}
			cur = cur->next;
		}
	}
}

xmlNodePtr NodeHyperPlane::saveFeature() const
{
	xmlNodePtr node = xmlNewNode( NULL, reinterpret_cast<const xmlChar*>( "feature" ) );
    //std::cout<<m_hp.numRandomFeatures<<" "<< m_bestFeatures.size()<<std::endl;
    for (int i=0;i<m_hp. numRandomFeatures;++i){
        string s = "feat"+boost::lexical_cast<string>( i );
        addIntProp(node, s.c_str(), m_bestFeatures[i]);
        string s2 = "weight"+boost::lexical_cast<string>( i );
        addDoubleProp(node, s2.c_str(), m_bestWeights[i]);
    }
	addDoubleProp(node, "threshold", m_bestThreshold);

	return node;
}


xmlNodePtr NodeHyperPlane::save() const
{
	xmlNodePtr node = xmlNewNode( NULL, reinterpret_cast<const xmlChar*>( "node" ) );
	xmlNewProp( node, reinterpret_cast<const xmlChar*>( "type" ),
			reinterpret_cast<const xmlChar*>( NODE_GINI ) );
	const char* isLeaf = (m_isLeaf) ? "true" : "false";
	xmlNewProp( node, reinterpret_cast<const xmlChar*>( "isLeaf" ),
			reinterpret_cast<const xmlChar*>( isLeaf ) );
	if (!m_isLeaf)
	{
		xmlAddChild(node, saveFeature());
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
		addIntProp( node, "label", m_nodeLabel);
		std::vector<float>::const_iterator it(m_nodeConf.begin()),end(m_nodeConf.end());
		int idx = 0;
		for (;it != end;it++,idx++)
		{
			xmlAddChild(node,saveConfidence(idx,*it));
		}
	}

	return node;
}

std::pair<float, float> NodeHyperPlane::calcGiniAndThreshold(const std::vector<int>& labels,
		const std::vector<std::pair<float, int> >& responses)
{
	// Initialize the counters: left takes all at the begining
	double DGini, LGini, RGini, LTotal, RTotal, bestW0 = 0, bestDGini = 1e10;
	std::vector<double> LCount(m_hp.numClasses, 0.0), RCount(m_hp.numClasses, 0.0);

	if (m_hp.isExtreme)
	{
		std::vector<std::pair<float, int> >::const_iterator resIt(responses.begin()), resEnd(responses.end());
		bestW0 = 2.0 * randomDouble( 1.0 ) - 1.0;
		RTotal = 0;
		LTotal = 0;
		for (; resIt != resEnd; resIt++)
		{
			if (resIt->first > bestW0)
			{
				RTotal++;
				RCount[labels[resIt->second]]++;
			}
			else
			{
				LTotal++;
				LCount[labels[resIt->second]]++;
			}
		}

		LGini = 0;
		RGini = 0;
		std::vector<double>::iterator LIt = LCount.begin(), RIt = RCount.begin(), end = LCount.end(), REnd = RCount.end();
		for (; LIt != end; LIt++, RIt++)      // Calculate Gini index
				{
			if (LTotal)
			{
				LGini += (*LIt/LTotal)*(1 - *LIt/LTotal);
			}
			if (RTotal)
			{
				RGini += (*RIt/RTotal)*(1 - *RIt/RTotal);
			}
				}

		bestDGini = (LTotal*LGini + RTotal*RGini)/responses.size();
	}
	else
	{
		RTotal = responses.size();
		LTotal = 0;

		// Count the number of samples in each class
		std::vector<std::pair<float, int> >::const_iterator resIt(responses.begin()), resEnd(responses.end()), tmpResIt;
		for (; resIt != resEnd; resIt++)
		{
			RCount[labels[resIt->second]]++;
		}

		// Loop over the sorted values and find the min DGini
		std::vector<double>::iterator LIt = LCount.begin(), RIt = RCount.begin(), end = LCount.end(), REnd = RCount.end();
		resIt = responses.begin();
		++resIt;
		for (; resIt != resEnd; resIt++)
		{
			tmpResIt = resIt;
			--tmpResIt;

			RTotal--;
			LTotal++;
			RCount[labels[tmpResIt->second]]--;
			LCount[labels[tmpResIt->second]]++;

			if (resIt->first != tmpResIt->first)
			{
				LGini = 0;
				RGini = 0;
				LIt = LCount.begin();
				RIt = RCount.begin();
				for (; LIt != end; LIt++, RIt++)      // Calculate Gini index
						{
					LGini += (*LIt/LTotal)*(1 - *LIt/LTotal);
					RGini += (*RIt/RTotal)*(1 - *RIt/RTotal);
						}

				DGini = (LTotal*LGini + RTotal*RGini)/responses.size();
				if (DGini < bestDGini)
				{
					bestDGini = DGini;
					bestW0 = (resIt->first + tmpResIt->first)*0.5;
				}
			}
		}
	}

	return std::pair<float,float>((float)bestDGini,(float)bestW0);
}

std::pair<float, float> NodeHyperPlane::calcInfoGainAndThreshold(const std::vector<int>& labels,
		const std::vector<std::pair<float, int> >& responses)
{
	// Initialize the counters: left takes all at the begining
	double DInfo, LInfo, RInfo, LTotal, RTotal, bestW0 = 0.0, bestDInfo = 1e10;
	std::vector<double> LCount(m_hp.numClasses, 0.0), RCount(m_hp.numClasses, 0.0);

	RTotal = responses.size();
	LTotal = 0.0;
	// Count the number of samples in each class
	std::vector<std::pair<float, int> >::const_iterator resIt(responses.begin()), resEnd(responses.end()), tmpResIt;
	for (; resIt != resEnd; resIt++)
	{
		RCount[labels[resIt->second]]++;
	}

	// Loop over the sorted values and find the max DInfo
	std::vector<double>::iterator LIt = LCount.begin(), RIt = RCount.begin(), end = LCount.end(), REnd = RCount.end();
	resIt = responses.begin();
	++resIt;
	for (; resIt != resEnd; resIt++)
	{
		tmpResIt = resIt;
		--tmpResIt;

		RTotal--;
		LTotal++;
		RCount[labels[tmpResIt->second]]--;
		LCount[labels[tmpResIt->second]]++;

		if (resIt->first != tmpResIt->first)
		{
			LInfo = 0.0;
			RInfo = 0.0;
			LIt = LCount.begin();
			RIt = RCount.begin();
			for (; LIt != end; LIt++, RIt++)      // Calculate Info index
					{
				if (*LIt)
				{
					LInfo -= (*LIt/LTotal)*log(*LIt/LTotal);
				}
				if (*RIt)
				{
					RInfo -= (*RIt/RTotal)*log(*RIt/RTotal);
				}
					}

			DInfo = (LTotal*LInfo + RTotal*RInfo)/responses.size();
			if (DInfo < bestDInfo)
			{
				bestDInfo = DInfo;
				bestW0 = (resIt->first + tmpResIt->first)*0.5;
			}
		}
	}

	return std::pair<float,float>((float)bestDInfo,(float)bestW0);
}

std::pair<float, float> NodeHyperPlane::calcInfoGainAndThreshold(const std::vector<int>& labels, const std::vector<double>& weights,
		const std::vector<std::pair<float, int> >& responses)
{
	// Initialize the counters: left takes all at the begining
	double DInfo, LInfo, RInfo, LTotal, RTotal, bestW0 = 0.0, bestDInfo = 1e10;
	std::vector<double> LCount(m_hp.numClasses, 0.0), RCount(m_hp.numClasses, 0.0);

	RTotal = 0.0;
	LTotal = 0.0;
	// Count the number of samples in each class
	std::vector<std::pair<float, int> >::const_iterator resIt(responses.begin()), resEnd(responses.end()), tmpResIt;
	for (; resIt != resEnd; resIt++)
	{
		RCount[labels[resIt->second]] += weights[resIt->second];
		RTotal += weights[resIt->second];
	}

	// Loop over the sorted values and find the max DInfo
	std::vector<double>::iterator LIt = LCount.begin(), RIt = RCount.begin(), end = LCount.end(), REnd = RCount.end();
	resIt = responses.begin();
	++resIt;
	for (; resIt != resEnd; resIt++)
	{
		tmpResIt = resIt;
		--tmpResIt;

		RTotal -= weights[tmpResIt->second];
		LTotal += weights[tmpResIt->second];
		RCount[labels[tmpResIt->second]] -= weights[tmpResIt->second];
		LCount[labels[tmpResIt->second]] += weights[tmpResIt->second];

		if (resIt->first != tmpResIt->first)
		{
			LInfo = 0.0;
			RInfo = 0.0;
			LIt = LCount.begin();
			RIt = RCount.begin();
			for (; LIt != end; LIt++, RIt++)      // Calculate Info index
					{
				if (*LIt)
				{
					LInfo -= (*LIt/LTotal)*log(*LIt/LTotal);
				}
				if (*RIt)
				{
					RInfo -= (*RIt/RTotal)*log(*RIt/RTotal);
				}
					}

			DInfo = (LTotal*LInfo + RTotal*RInfo)/(LTotal + RTotal);
			if (DInfo < bestDInfo)
			{
				bestDInfo = DInfo;
				bestW0 = (resIt->first + tmpResIt->first)*0.5;
			}
		}
	}

	return std::pair<float,float>((float)bestDInfo,(float)bestW0);
}


std::pair<float, float> NodeHyperPlane::calcGiniAndThreshold(const std::vector<int>& labels, const std::vector<double>& weights,
		const std::vector<std::pair<float, int> >& responses, const bool useUnlabeledData)
{
	// Initialize the counters: left takes all at the begining
	double DGini, LGini, RGini, LTotal, RTotal, bestW0 = 0, bestDGini = 1e10;
	std::vector<double> LCount(m_hp.numClasses, 0.0), RCount(m_hp.numClasses, 0.0);

	RTotal = 0;
	LTotal = 0;

	// Count the number of samples in each class
	std::vector<std::pair<float, int> >::const_iterator resIt(responses.begin()), resEnd(responses.end()), tmpResIt;
	for (; resIt != resEnd; resIt++)
	{
		if (useUnlabeledData || resIt->second < m_hp.numLabeled)
		{
			RCount[labels[resIt->second]] += weights[resIt->second];
			RTotal += weights[resIt->second];
		}
	}

	// Loop over the sorted values and find the min DGini
	std::vector<double>::iterator LIt = LCount.begin(), RIt = RCount.begin(), end = LCount.end(), REnd = RCount.end();
	resIt = responses.begin();
	++resIt;
	for (; resIt != resEnd; resIt++)
	{
		if (useUnlabeledData || resIt->second < m_hp.numLabeled)
		{
			tmpResIt = resIt;
			--tmpResIt;

			RTotal -= weights[tmpResIt->second];
			LTotal += weights[tmpResIt->second];
			RCount[labels[tmpResIt->second]] -= weights[tmpResIt->second];
			LCount[labels[tmpResIt->second]] += weights[tmpResIt->second];

			if (resIt->first != tmpResIt->first)
			{
				LGini = 0;
				RGini = 0;
				LIt = LCount.begin();
				RIt = RCount.begin();
				for (; LIt != end; LIt++, RIt++)      // Calculate Gini index
						{
					LGini += (*LIt/LTotal)*(1 - *LIt/LTotal);
					RGini += (*RIt/RTotal)*(1 - *RIt/RTotal);
						}

				DGini = (LTotal*LGini + RTotal*RGini)/(LTotal + RTotal);
				if (DGini < bestDGini)
				{
					bestDGini = DGini;
					bestW0 = (resIt->first + tmpResIt->first)*0.5;
				}
			}
		}
	}

	return std::pair<float,float>((float)bestDGini,(float)bestW0);
}


void NodeHyperPlane::findHypotheses(const matrix<float>& data, const std::vector<int>& labels,
		const std::vector<int>& inBagSamples, const std::vector<int>& randFeatures, int numTries)
{
	std::vector<double> gini(m_hp.numRandomFeatures), thresholds(m_hp.numRandomFeatures);
	std::vector<int>::const_iterator it(randFeatures.begin());
	std::vector<int>::const_iterator end(randFeatures.end());

	std::vector<int>::const_iterator bagIt;
	std::vector<int>::const_iterator bagEnd(inBagSamples.end());

	double bestDGini = 1e10, bestThreshold = 0;
	std::pair<float,float> curGiniThresh;
	std::vector<std::pair<float, int> > responses;
	std::vector<float> bestWeights(randFeatures.size(),0.0);
	std::vector<float> tmpWeights(randFeatures.size(),0.0);
	float tmp = 0.0;
	for ( int i = 0; i < numTries; i++)
	{
		fillWithRandomNumbers(tmpWeights);
		responses.clear();
		responses.reserve(inBagSamples.size());
		bagIt = inBagSamples.begin();
		while ( bagIt != bagEnd )
		{
			tmp = 0.0;
			int counter = 0;
			BOOST_FOREACH(int feat, randFeatures)
			{
				tmp += data(*bagIt,feat)*tmpWeights[counter];
				counter++;
			}

			responses.push_back(std::pair<float, int>(tmp,*bagIt));
			++bagIt;
		}

		sort(responses.begin(), responses.end());

		if (m_hp.useInfoGain)
		{
			curGiniThresh = calcInfoGainAndThreshold(labels, responses);
		}
		else
		{
			curGiniThresh = calcGiniAndThreshold(labels, responses);
		}

		if (curGiniThresh.first < bestDGini)
		{
			bestDGini = curGiniThresh.first;
			bestThreshold = curGiniThresh.second;
			bestWeights = tmpWeights;
		}
	}

	m_bestWeights = bestWeights;
	m_bestFeatures = randFeatures;
	m_bestThreshold = (float) bestThreshold;
}

void NodeHyperPlane::findHypothesesLU(const matrix<float>& data, const std::vector<int>& labels,
		const std::vector<int>& inBagSamples, const std::vector<int>& randFeatures, int numTries)
{
	std::vector<double> gini(m_hp.numRandomFeatures), thresholds(m_hp.numRandomFeatures);
	std::vector<int>::const_iterator it(randFeatures.begin());
	std::vector<int>::const_iterator end(randFeatures.end());

	std::vector<int>::const_iterator bagIt;
	std::vector<int>::const_iterator bagEnd(inBagSamples.end());

	double bestDGini = 1e10, bestThreshold = 0;
	std::pair<float,float> curGiniThresh;
	std::vector<std::pair<float, int> > responses;
	std::vector<float> bestWeights(randFeatures.size(),0.0);
	std::vector<float> tmpWeights(randFeatures.size(),0.0);
	float tmp = 0.0;
	for ( int i = 0; i < numTries; i++)
	{
		fillWithRandomNumbers(tmpWeights);
		responses.clear();
		responses.reserve(inBagSamples.size());
		bagIt = inBagSamples.begin();
		while ( bagIt != bagEnd )
		{
			tmp = 0.0;
			int counter = 0;
			BOOST_FOREACH(int feat, randFeatures)
			{
				tmp += data(*bagIt,feat)*tmpWeights[counter];
				counter++;
			}

			responses.push_back(std::pair<float, int>(tmp,*bagIt));
			++bagIt;
		}

		sort(responses.begin(), responses.end());

		if (inBagSamples[0] < m_hp.numLabeled)
		{
			curGiniThresh = calcGiniAndThreshold(labels, responses);
		}
		else
		{
			curGiniThresh = calcClusterScoreAndThreshold(data, inBagSamples, responses);
		}
		if (curGiniThresh.first < bestDGini)
		{
			bestDGini = curGiniThresh.first;
			bestThreshold = curGiniThresh.second;
			bestWeights = tmpWeights;
		}
	}

	m_bestWeights = bestWeights;
	m_bestFeatures = randFeatures;
	m_bestThreshold = (float) bestThreshold;
}

std::pair<float, float> NodeHyperPlane::calcClusterScoreAndThreshold(const matrix<float>& data, const std::vector<int>& inBagSamples,
		const std::vector<double>& weights,
		const std::vector<std::pair<float, int> >& responses)
{
	// Find the mid-point using responsess
			int numResponse = responses.size();
	int midPointIndex = (int) round(numResponse/2 - 1);
	float threshold = (responses[midPointIndex].first + responses[midPointIndex + 1].first)/2;

	// Calculate the weighted cluster center for left and right splits
	double LWeight = 0, RWeight = 0;
	std::vector<float> LCenter(data.size2(), 0.0), RCenter(data.size2(), 0.0);
	for (int m = 0; m < (int) data.size2(); m++)
	{
		for (int n = 0; n < midPointIndex; n++)
		{
			LCenter[m] += (float)weights[responses[n].second]*data(responses[n].second, m);
			if (m == 0)
			{
				LWeight += weights[responses[n].second];
			}
		}
		LCenter[m] /= (float)LWeight;

		for (int n = midPointIndex; n < (int) responses.size(); n++)
		{
			RCenter[m] += (float)weights[responses[n].second]*data(responses[n].second, m);
			if (m == 0)
			{
				RWeight += weights[responses[n].second];
			}
		}
		RCenter[m] /= (float)RWeight;
	}

	// Calculate the weighted distance from each point to the centers
	float LScore = 0, RScore = 0;
	for (int m = 0; m < (int) data.size2(); m++)
	{
		for (int n = 0; n < midPointIndex; n++)
		{
			LScore += weights[responses[n].second]*pow((double) (data(responses[n].second, m) - LCenter[m]), 2.0);
		}
		LScore /= (float)LWeight;
		for (int n = midPointIndex; n < (int) responses.size(); n++)
		{
			RScore += (float)weights[responses[n].second]*pow((double) (data(responses[n].second, m) - RCenter[m]), 2.0);
		}
		RScore /= (float)RWeight;
	}

	return std::pair<float,float>(0.5f*(LScore + RScore), threshold);
}

std::pair<float, float> NodeHyperPlane::calcClusterScoreAndThreshold(const matrix<float>& data, const std::vector<int>& inBagSamples,
		const std::vector<std::pair<float, int> >& responses)
{
	// Find the mid-point using responsess
	int numResponse = responses.size();
	int midPointIndex = (int) round(numResponse/2 - 1);
	float threshold = (responses[midPointIndex].first + responses[midPointIndex + 1].first)/2;

	// Calculate the weighted cluster center for left and right splits
	double LWeight = 0, RWeight = 0;
	std::vector<float> LCenter(data.size2(), 0.0), RCenter(data.size2(), 0.0);
	for (int m = 0; m < (int) data.size2(); m++)
	{
		for (int n = 0; n < midPointIndex; n++)
		{
			LCenter[m] += data(responses[n].second, m);
			if (m == 0)
			{
				LWeight++;
			}
		}
		LCenter[m] /= LWeight;

		for (int n = midPointIndex; n < (int) responses.size(); n++)
		{
			RCenter[m] += data(responses[n].second, m);
			if (m == 0)
			{
				RWeight++;
			}
		}
		RCenter[m] /= RWeight;
	}

	// Calculate the weighted distance from each point to the centers
	float LScore = 0, RScore = 0;
	for (int m = 0; m < (int) data.size2(); m++)
	{
		for (int n = 0; n < midPointIndex; n++)
		{
			LScore += pow((double) (data(responses[n].second, m) - LCenter[m]), 2.0);
		}
		LScore /= LWeight;
		for (int n = midPointIndex; n < (int) responses.size(); n++)
		{
			RScore += pow((double) (data(responses[n].second, m) - RCenter[m]), 2.0);
		}
		RScore /= RWeight;
	}

	return std::pair<float,float>(0.5*(LScore + RScore), threshold);
}

void NodeHyperPlane::findHypotheses(const matrix<float>& data, const std::vector<int>& labels,
		const std::vector<double>& weights,
		const std::vector<int>& inBagSamples, const std::vector<int>& randFeatures, int numTries)
{
	std::vector<double> gini(m_hp.numRandomFeatures), thresholds(m_hp.numRandomFeatures);
	std::vector<int>::const_iterator it(randFeatures.begin());
	std::vector<int>::const_iterator end(randFeatures.end());
	std::vector<int>::const_iterator bagIt;
	std::vector<int>::const_iterator bagEnd(inBagSamples.end());

	double bestDGini = 1e10, bestThreshold = 0.0;
	std::pair<float,float> curGiniThresh;
	std::vector<std::pair<float, int> > responses;

	std::vector<float> bestWeights(randFeatures.size(),0.0);
	std::vector<float> tmpWeights(randFeatures.size(),0.0);
	float tmp = 0.0;
	bool doClustering = clusterOrGini(), useUnlabeledData = true;
	for ( int i = 0; i < numTries; i++)
	{
		fillWithRandomNumbers(tmpWeights);
		responses.clear();
		responses.reserve(inBagSamples.size());
		bagIt = inBagSamples.begin();
		while ( bagIt != bagEnd )
		{
			tmp = 0.0;
			int counter = 0;
			BOOST_FOREACH(int feat, randFeatures)
			{
				tmp += data(*bagIt,feat)*tmpWeights[counter];
				counter++;
			}

			responses.push_back(std::pair<float, int>(tmp,*bagIt));
			++bagIt;
		}
		sort(responses.begin(), responses.end());

		if (!doClustering)
		{
			if (m_hp.useInfoGain)
			{
				curGiniThresh = calcInfoGainAndThreshold(labels, weights, responses);
			}
			else
			{
				curGiniThresh = calcGiniAndThreshold(labels, weights, responses, useUnlabeledData);
			}
		}
		else
		{
			curGiniThresh = calcClusterScoreAndThreshold(data, inBagSamples, weights, responses);
		}

		if (curGiniThresh.first < bestDGini)
		{
			bestDGini = curGiniThresh.first;
			bestThreshold = curGiniThresh.second;
			bestWeights = tmpWeights;
		}
	}

	m_bestFeatures = randFeatures;
	m_bestWeights = bestWeights;
	m_bestThreshold = (float) bestThreshold;
}

NODE_TRAIN_STATUS NodeHyperPlane::trainLU(const matrix<float>& data, const std::vector<int>& labels,
		std::vector<int>& inBagSamples, matrix<float>& confidences, std::vector<int>& predictions)
{
	bool doSplit = shouldISplitLU(labels,inBagSamples);
	NODE_TRAIN_STATUS myTrainingStatus = IS_NOT_LEAF;

	if ( doSplit )
	{
		m_isLeaf = false;

		//train here the node: Select random features and evaluate them
		std::vector<int> randFeatures = randPerm(data.size2(), m_hp.numProjFeatures );
		int numTries = m_hp.numRandomFeatures;// * (m_depth+1);
		findHypothesesLU(data, labels, inBagSamples, randFeatures, numTries);
		if (m_hp.verbose)
		{
			cout << "Node #: " << m_nodeIndex;
			cout << " and the threshold is: " << m_bestThreshold << " at depth " << m_depth << endl;
		}

		// split the data
		std::vector<int> leftNodeSamples, rightNodeSamples;
		evalNode(data,inBagSamples,leftNodeSamples,rightNodeSamples);

		// pass them to the left and right child, respectively
		m_leftChildNode = Ptr(new NodeHyperPlane(m_hp,m_depth + 1));
		m_rightChildNode = Ptr(new NodeHyperPlane(m_hp,m_depth + 1));

		m_leftChildNode->train(data,labels,leftNodeSamples,confidences,predictions);
		m_rightChildNode->train(data,labels,rightNodeSamples,confidences,predictions);
	}
	else
	{
		if (m_hp.verbose)
		{
			cout << "Node #: " << m_nodeIndex << " is terminal, at depth " << m_depth << endl;
		}

		// calc confidence, labels, etc
		m_isLeaf = true;
		myTrainingStatus = IS_LEAF;
		m_nodeConf.resize(m_hp.numClasses, 0.0);
		int numNodeLabeled = 0;
		BOOST_FOREACH(int n, inBagSamples)
		{
			if (n < m_hp.numLabeled)
			{
				m_nodeConf[labels[n]]++;
				numNodeLabeled++;
			}
		}

		int bestClass = 0, tmpN = 0;
		float bestConf = 0;
		std::vector<float>::iterator confItr = m_nodeConf.begin(), confEnd = m_nodeConf.end();
		for (; confItr != confEnd; confItr++)
		{
			if (numNodeLabeled)
			{
				*confItr /= numNodeLabeled;
				if (*confItr > bestConf)
				{
					bestConf = *confItr;
					bestClass = tmpN;
				}
				tmpN++;
			}
			else
			{
				*confItr = 1.0/m_hp.numClasses;
			}
		}
		m_nodeLabel = bestClass;

		BOOST_FOREACH(int n, inBagSamples)
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

	return myTrainingStatus;
}

bool NodeHyperPlane::clusterOrGini()
{
	return false;
}

void NodeHyperPlane::evalNode(const matrix<float>& data, const std::vector<int>& inBagSamples,
		std::vector<int>& leftNodeSamples, std::vector<int>& rightNodeSamples)
{
	float tmp;
	BOOST_FOREACH(int n, inBagSamples)
	{
		tmp = 0.0;
		int counter = 0;
		BOOST_FOREACH(int feat, m_bestFeatures)
		{
			tmp += data(n,feat)*m_bestWeights[counter];
			counter++;
		}

		if (tmp > m_bestThreshold)
		{
			rightNodeSamples.push_back(n);
		}
		else
		{
			leftNodeSamples.push_back(n);
		}
	}
}

NODE_TRAIN_STATUS NodeHyperPlane::train(const matrix<float>& data, const std::vector<int>& labels,
		std::vector<int>& inBagSamples, matrix<float>& confidences, std::vector<int>& predictions)
{
	bool doSplit = shouldISplit(labels,inBagSamples);
	NODE_TRAIN_STATUS myTrainingStatus = IS_NOT_LEAF;

	if ( doSplit )
	{
		m_isLeaf = false;

		//train here the node: Select random features and evaluate them
		std::vector<int> randFeatures = randPerm(data.size2(), m_hp.numProjFeatures );
		int numTries = m_hp.numRandomFeatures;// * (m_depth+1);
		findHypotheses(data, labels, inBagSamples, randFeatures, numTries);
		if (m_hp.verbose)
		{
			cout << "Node #: " << m_nodeIndex;
			cout << " and the threshold is: " << m_bestThreshold << " at depth " << m_depth << endl;
		}

		// split the data
		std::vector<int> leftNodeSamples, rightNodeSamples;
		evalNode(data,inBagSamples,leftNodeSamples,rightNodeSamples);

		// pass them to the left and right child, respectively
		m_leftChildNode = Ptr(new NodeHyperPlane(m_hp,m_depth + 1));
		m_rightChildNode = Ptr(new NodeHyperPlane(m_hp,m_depth + 1));

		NODE_TRAIN_STATUS leftChildStatus = m_leftChildNode->train(data,labels,leftNodeSamples,confidences,predictions);
		NODE_TRAIN_STATUS rightChildStatus= m_rightChildNode->train(data,labels,rightNodeSamples,confidences,predictions);

	}
	else
	{
		if (m_hp.verbose)
		{
			cout << "Node #: " << m_nodeIndex << " is terminal, at depth " << m_depth << endl;
		}

		// calc confidence, labels, etc
		m_isLeaf = true;
		myTrainingStatus = IS_LEAF;
		m_nodeConf.resize(m_hp.numClasses, 0.0);

		BOOST_FOREACH(int n, inBagSamples)
		{
			m_nodeConf[labels[n]]++;
		}

		int bestClass = 0, tmpN = 0;
		float bestConf = 0;
		std::vector<float>::iterator confItr = m_nodeConf.begin(), confEnd = m_nodeConf.end();
		for (; confItr != confEnd; confItr++)
		{
			*confItr /= inBagSamples.size();
			if (*confItr > bestConf)
			{
				bestConf = *confItr;
				bestClass = tmpN;
			}
			tmpN++;
		}
		m_nodeLabel = bestClass;

		BOOST_FOREACH(int n, inBagSamples)
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

	return myTrainingStatus;
}

NODE_TRAIN_STATUS NodeHyperPlane::train(const matrix<float>& data, const std::vector<int>& labels, const std::vector<double>& weights,
		std::vector<int>& inBagSamples, matrix<float>& confidences, std::vector<int>& predictions)
{
	bool doSplit = shouldISplit(labels,inBagSamples);
	NODE_TRAIN_STATUS myTrainingStatus = IS_NOT_LEAF;

	if ( doSplit )
	{
		m_isLeaf = false;

		//train here the node: Select random features and evaluate them
		std::vector<int> randFeatures = randPerm(data.size2(),m_hp.numProjFeatures );
		int numTries = m_hp.numRandomFeatures * (m_depth+1);
		findHypotheses(data, labels, weights, inBagSamples, randFeatures,numTries);
		if (m_hp.verbose)
		{
			cout << "Node #: " << m_nodeIndex;
			cout << " and the threshold is: " << m_bestThreshold << " at depth " << m_depth << endl;
		}

		// split the data
		std::vector<int> leftNodeSamples, rightNodeSamples;
		evalNode(data,inBagSamples,leftNodeSamples,rightNodeSamples);

		// pass them to the left and right child, respectively
		m_leftChildNode = Ptr(new NodeHyperPlane(m_hp,m_depth + 1));
		m_rightChildNode = Ptr(new NodeHyperPlane(m_hp,m_depth + 1));

		NODE_TRAIN_STATUS leftChildStatus = m_leftChildNode->train(data,labels,weights,leftNodeSamples,confidences,predictions);
		NODE_TRAIN_STATUS rightChildStatus= m_rightChildNode->train(data,labels,weights,rightNodeSamples,confidences,predictions);

	}
	else
	{
		if (m_hp.verbose)
		{
			cout << "Node #: " << m_nodeIndex << " is terminal, at depth " << m_depth << endl;
		}

		// calc confidence, labels, etc
		m_isLeaf = true;
		myTrainingStatus = IS_LEAF;
		m_nodeConf.resize(m_hp.numClasses, 0.0);

		double totalW = 0;
		BOOST_FOREACH(int n, inBagSamples)
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

		BOOST_FOREACH(int n, inBagSamples)
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

	return myTrainingStatus;
}


void NodeHyperPlane::eval(const matrix<float>& data, const std::vector<int>& sampleIndeces,
		matrix<float>& confidences, std::vector<int>& predictions)
{
	if (m_isLeaf)
	{
		// Make predictions and confidences
		int tmpN;
		BOOST_FOREACH( int n, sampleIndeces)
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
	else
	{
		// split the data
		std::vector<int> leftNodeSamples, rightNodeSamples;
		evalNode(data,sampleIndeces,leftNodeSamples,rightNodeSamples);

		m_leftChildNode->eval(data,leftNodeSamples,confidences,predictions);
		m_rightChildNode->eval(data,rightNodeSamples,confidences,predictions);
	}
}

void NodeHyperPlane::getPath(const matrix<float>& data, const std::vector<int>& sampleIndeces, std::vector<std::vector<int> >& path)
{
	BOOST_FOREACH(int n, sampleIndeces)
    		{
		path[n].push_back(m_nodeIndex);
    		}

	if (!m_isLeaf)
	{
		// split the data
		std::vector<int> leftNodeSamples, rightNodeSamples;
		evalNode(data,sampleIndeces,leftNodeSamples,rightNodeSamples);

		m_leftChildNode->getPath(data,leftNodeSamples,path);
		m_rightChildNode->getPath(data,rightNodeSamples,path);
	}
}

void NodeHyperPlane::refine(const matrix<float>& data, const std::vector<int>& labels,
		std::vector<int>& samples, matrix<float>& confidences, std::vector<int>& predictions)
{
	if ( !m_isLeaf )
	{
		// split the data
		std::vector<int> leftNodeSamples, rightNodeSamples;
		evalNode(data,samples,leftNodeSamples,rightNodeSamples);

		m_leftChildNode->refine(data,labels,leftNodeSamples,confidences,predictions);
		m_rightChildNode->refine(data,labels,rightNodeSamples,confidences,predictions);
	}
	else
	{
		// calc confidence, labels, etc
		m_nodeConf.resize(m_hp.numClasses,0.0);
		BOOST_FOREACH(int n, samples)
		{
			m_nodeConf[labels[n]]++;
		}

		int bestClass = 0, tmpN = 0;
		float bestConf = 0;
		std::vector<float>::iterator confItr = m_nodeConf.begin(), confEnd = m_nodeConf.end();
		for (; confItr != confEnd; confItr++)
		{
			*confItr /= samples.size();
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

void NodeHyperPlane::refine(const matrix<float>& data, const std::vector<int>& labels, const std::vector<double>& weights,
		std::vector<int>& samples, matrix<float>& confidences, std::vector<int>& predictions)
{
	// calc confidence, labels, etc
	if (!m_isLeaf)
	{
		// split the data
		std::vector<int> leftNodeSamples, rightNodeSamples;
		evalNode(data,samples,leftNodeSamples,rightNodeSamples);

		m_leftChildNode->refine(data,labels,weights,leftNodeSamples,confidences,predictions);
		m_rightChildNode->refine(data,labels,weights,rightNodeSamples,confidences,predictions);
	}
	else
	{
		m_nodeConf.resize(m_hp.numClasses,0.0);
		std::vector<float>::iterator confItr2 = m_nodeConf.begin(), confEnd2 = m_nodeConf.end();
		for (; confItr2 != confEnd2; confItr2++)
		{
			*confItr2=0;
//			std::cout<<*confItr2<<std::endl;
		}
		double totalW = 0;
		BOOST_FOREACH(int n, samples)
		{
			m_nodeConf[labels[n]] += weights[n];
			totalW += weights[n];
//			std::cout<<n<<" "<<labels[n]<<" "<<weights[n]<<" "<<m_nodeConf[labels[n]]<<" "<<(totalW + 1e-10)<<" "<<m_nodeConf[labels[n]]/(totalW + 1e-10)<<std::endl;

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
