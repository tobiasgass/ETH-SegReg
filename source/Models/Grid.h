/*
 * Grid.h
 *
 *  Created on: Nov 25, 2010
 *      Author: gasst
 */

#ifndef GRIm_dim_H_
#define GRIm_dim_H_

#include <vector>
#include <assert.h>

/*
 * Isotropic Grid
 * Returns current/next position in a grid based on size and resolution
 */
template<class TImage>
class Grid{
public:
	typedef typename TImage::IndexType IndexType;
	typedef typename TImage::OffsetType LabelType;
	typedef typename TImage::OffsetType OffsetType;
	typedef typename TImage::SizeType SizeType;
	typedef  TImage ImageType;
	typedef typename TImage::Pointer ImagePointer;

private:
	IndexType m_currentGridPosition,m_currentImagePosition,m_beginning;
	int m_currentIndex;
	SizeType m_totalSize,m_gridSize,m_imageLevelDivisors;
	LabelType m_resolution;

	static const unsigned int m_dim=TImage::ImageDimension;
	int m_nNodes,m_nVertices;

public:

	Grid(ImagePointer image, LabelType res){
		assert(m_dim>1);
		assert(m_dim<4);
		m_totalSize=image->GetLargestPossibleRegion().GetSize();
		m_resolution=res;
		m_nNodes=1;
		for (int d=0;d<m_dim;++d){
			m_beginning[d]=0;
			m_currentGridPosition[d]=0;
			m_currentImagePosition[d]=0;
			m_gridSize[d]=m_totalSize[d]/m_resolution[d];
			m_nNodes*=m_gridSize[d];
			if (d>0){
				m_imageLevelDivisors[d]=m_imageLevelDivisors[d-1]*m_gridSize[d-1];
			}else{
				m_imageLevelDivisors[d]=1;
			}
		}
		std::cout<<"GridSize: "<<m_dim<<" ";
		if (m_dim>=2){
			std::cout<<m_gridSize[0]<<" "<<m_gridSize[1];
			m_nVertices=m_gridSize[1]*(m_gridSize[0]-1)+m_gridSize[0]*(m_gridSize[1]-1);
		}
		if (m_dim==3){
			std::cout<<" "<<m_gridSize[0];

			m_nVertices+=(m_gridSize[2]-1)*m_gridSize[1]*m_gridSize[0];
		}
		std::cout<<std::endl;

	}
	IndexType  getGridPosition(){
		return m_currentGridPosition;
	}
	IndexType  getImagePosition(){
		return m_currentImagePosition;
	}
	int getIndex(){
		return m_currentIndex;
	}
	IndexType gridToImageIndex(IndexType gridIndex){
		IndexType imageIndex;
		for (int d=0;d<m_dim;++d){
			imageIndex[d]=gridIndex[d]*m_resolution[d];
		}
		return imageIndex;
	}

	IndexType imageToGridIndex(IndexType imageIndex){
		IndexType gridIndex;
		for (int d=0;d<m_dim;++d){
			gridIndex[d]=imageIndex[d]/m_resolution[d];
		}
		return gridIndex;
	}
	IndexType getGridPositionAtIndex(int idx){
		IndexType position;
		for (int d=m_dim-1;d>=0;--d){
			position[d]=idx/m_imageLevelDivisors[d];
			idx-=position[d]*m_imageLevelDivisors[d];
		}
		return position;
	}
	int  getIntegerIndex(IndexType gridIndex){
		int i=0;
		for (int d=0;d<m_dim;++d){
			i+=gridIndex[d]*m_imageLevelDivisors[d];
		}
		return i;
	}

	int nNodes(){return m_nNodes;}

	void next(){

		if (m_currentIndex<m_nNodes-1){
			m_currentIndex++;
			m_currentGridPosition=getGridPositionAtIndex(m_currentIndex);
			m_currentImagePosition=gridToImageIndex(m_currentGridPosition);
		}

	}
	IndexType & getNextPosition(){
		if (!m_currentIndex>=m_nNodes){
			m_currentIndex++;
			m_currentGridPosition=getGridPositionAtIndex(m_currentIndex);
			m_currentImagePosition=gridToImageIndex(m_currentGridPosition);
		}
		return m_currentGridPosition;
	}

	void gotoBegin(){
		m_currentGridPosition.Fill(0);
		m_currentImagePosition.Fill(0);
		m_currentIndex=0;
	}
	LabelType getResolution(){return m_resolution;}
	int nVertices(){return m_nVertices;}

	std::vector<int> getCurrentForwardNeighbours(){
		std::vector<int> neighbours;
		for (int d=0;d<m_dim;++d){
			OffsetType off;
			off.Fill(0);
			if (m_currentGridPosition[d]<m_gridSize[d]-1){
				off[d]+=1;
//				std::cout<<off<<" a "<<m_currentGridPosition+off<<" "<<getIntegerIndex(m_currentGridPosition+off)<<std::endl;
				neighbours.push_back(getIntegerIndex(m_currentGridPosition+off));
			}
		}
		return neighbours;

	}
	bool atEnd(){return (m_currentIndex==(m_nNodes-1));}

	IndexType getCurrentGridPosition() const
	{
	    return m_currentGridPosition;
	}
	IndexType getCurrentImagePosition() const
		{
		    return m_currentImagePosition;
		}

	int getCurrentIndex() const
	{
	    return m_currentIndex;
	}


	int getNNodes() const
	{
	    return m_nNodes;
	}

	SizeType getTotalSize() const
	{
	    return m_totalSize;
	}

	void setResolution(LabelType m_resolution)
	{
	    this->m_resolution = m_resolution;
	}

};



#endif /* GRIm_dim_H_ */
