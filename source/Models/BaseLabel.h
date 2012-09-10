#include "Log.h"
/*
 * Label.h
 *
 *  Created on: Nov 26, 2010
 *      Author: gasst
 */

#ifndef BASELABEL_H_
#define BASELABEL_H_
#include "itkLinearInterpolateImageFunction.h"
#include <boost/bimap.hpp>
#include <boost/bimap/unconstrained_set_of.hpp>

template<class TImage, class TLabel>
class BaseLabelMapper{
public:
	//	typedef typename TImage::OffsetType OffsetType;
	//	typedef typename itk::LinearInterpolateImageFunction<TImage>::ContinuousIndexType OffsetType;
	typedef typename itk::Vector<float,TImage::ImageDimension> OffsetType;
	typedef TLabel LabelType;
	typedef typename  itk::Image<LabelType,TImage::ImageDimension > LabelImageType;
	typedef typename LabelImageType::Pointer LabelImagePointerType;
	static int nLabels,nDisplacements,nSegmentations,nDisplacementSamples,k;
	static const int Dimension=TImage::ImageDimension+1;
    typedef typename TImage::SpacingType SpacingType;

	//private:
	//	int nLabels,nDisplacements,nSegmentations,nDisplacementSamples,k;
public:
	BaseLabelMapper(){}
	BaseLabelMapper(int NSegmentations, int NDisplacementSamples){
		nSegmentations=NSegmentations;
		nDisplacementSamples=NDisplacementSamples;
		nDisplacements=pow(double(2*nDisplacementSamples+1),TImage::ImageDimension);
		nLabels=nSegmentations*nDisplacements;
		k=TImage::ImageDimension+1;
	}
	void setDisplacementSamples(int nSamples){
		nDisplacementSamples=nSamples;
		nDisplacements=pow(double(2*nDisplacementSamples+1),TImage::ImageDimension);
		nLabels=nSegmentations*nDisplacements;
	}
	void setSegmentationLabels(int labels){
		nSegmentations=labels;
		nLabels=nSegmentations*nDisplacements;
	}
	static inline const LabelType scaleDisplacement(const LabelType & label,const itk::Vector<float,TImage::ImageDimension> & scaling){
		LabelType result(label);
		for (int d=0;d<TImage::ImageDimension;++d){
			result[d]=result[d]*scaling[d];
		}
		return result;
	}
    static inline const LabelType scaleDisplacement(const LabelType & label,const SpacingType & scaling){
		LabelType result(label);
		for (int d=0;d<TImage::ImageDimension;++d){
			result[d]=result[d]*scaling[d];
		}
		return result;
	}
	static inline const LabelType getLabel(int index){
		LabelType result;
		int m_segmentation;
		if (nDisplacements){
			m_segmentation=index/nDisplacements;
			index=index%nDisplacements;
		}
		else{
			m_segmentation=index;
		}
		int divisor=pow(double(2*nDisplacementSamples+1),TImage::ImageDimension-1);
		for (int d=0;d<TImage::ImageDimension;++d){
			result[d]=index/divisor-nDisplacementSamples;
			index-=(result[d]+nDisplacementSamples)*divisor;
			divisor/=2*nDisplacementSamples+1;
		}
		result[k-1]=m_segmentation;
		return result;
	}

	static inline const int getIndex(const LabelType & label){
		int index=0;
		index+=label[k-1]*(nDisplacements>0?nDisplacements:1);
		int factor=1;
		for (int d=TImage::ImageDimension-1;d>=0;--d){
			index+=factor*(label[d]+nDisplacementSamples);
			factor*=2*nDisplacementSamples+1;
		}
		return index;
	}
	static inline const OffsetType getDisplacement(const LabelType & label){
		OffsetType off;
		for (int d=0;d<TImage::ImageDimension;++d){
			off[d]=label[d];
		}
		return off;
	}
	static inline const float getSegmentation(const LabelType & label){
		return label[k-1];
	}
	static inline const void setSegmentation(LabelType & label, int seg){
		label[k-1]=seg;
	}
	static inline const void setDisplacement(LabelType & label, const OffsetType &off){
		for (int d=0;d<TImage::ImageDimension;++d){
			label[d]=off[d];
		}
	}
};

template<class TLabel>
struct comp:  binary_function <TLabel,TLabel,bool>{
    bool operator() (const TLabel &l1, const TLabel &l2) const{
        bool z1=true,z2=true;
        for (int i=0 ; i< TLabel::Dimension; ++i){
            if (l1[i]!=0) z1=false;
            if (l2[i]!=0) z2=false;
        }
        if (z1) return true;
        if (z2) return false;
        for (int i=0 ; i< TLabel::Dimension; ++i){
            if (l1[i]<l2[2]) return true;
        }
        return false;
    }
};

template<class TImage, class TLabel>
class SparseLabelMapper : public BaseLabelMapper<TImage,TLabel>{
public:
	//	typedef typename TImage::OffsetType OffsetType;
	//	typedef typename itk::LinearInterpolateImageFunction<TImage>::ContinuousIndexType OffsetType;
	typedef typename itk::Vector<float,TImage::ImageDimension> OffsetType;
	typedef TLabel LabelType;
	typedef typename  itk::Image<LabelType, TImage::ImageDimension> LabelImageType;
	typedef typename LabelImageType::Pointer LabelImagePointerType;
	static int nLabels,nDisplacements,nSegmentations,nDisplacementSamples,k;
    //	static const int Dimension=TImage::ImageDimension+1;
	//private:
	//	int nLabels,nDisplacements,nSegmentations,nDisplacementSamples,k;
public:
	SparseLabelMapper(){}
	SparseLabelMapper(int NSegmentations, int NDisplacementSamples){
		nSegmentations=NSegmentations;
		nDisplacementSamples=NDisplacementSamples;
		nDisplacements=(double(2*nDisplacementSamples+1)*TImage::ImageDimension);
		nLabels=nSegmentations*nDisplacements;
		LOG<<"nDisplacements :"<<nDisplacements<<" nSegmentations:"<<nSegmentations<<" nLabels"<<nLabels<<std::endl;
		k=TImage::ImageDimension+1;
        //   setupLabelMap();
	}
    void setSegmentationLabels(int labels){
        nSegmentations=labels;
        nLabels=nSegmentations*nDisplacements;
    }
    void setDisplacementSamples(int nSamples){
        nDisplacementSamples=nSamples;
        if (nDisplacementSamples){
            nDisplacements=(double(2*nDisplacementSamples+1)*TImage::ImageDimension);
        }
        else
            nDisplacements=1;
        nLabels=nSegmentations*nDisplacements;
        LOG<<"nDisplacements :"<<nDisplacements<<" nSegmentations:"<<nSegmentations<<" nLabels"<<nLabels<<std::endl;

    }
    static inline const LabelType getLabel(int index){
        //   return indexToLabelMap[index];
        LabelType result;
        result.Fill(0);
        int m_segmentation=0;
        if (nDisplacements){
            m_segmentation=index/nDisplacements;
            index=index%nDisplacements;
        }
        else if(nSegmentations){
            m_segmentation=index;
        }
        int divisor=(double(2*nDisplacementSamples+1));
        result[index/divisor]=index%divisor-nDisplacementSamples;
        result[k-1]=m_segmentation;
        return result;
    }

    static inline const int getIndex(const LabelType & label){
        //    return labelToIndexMap[label];
        int index=0;
        if (nSegmentations){
            index+=label[k-1]*(nDisplacements>0?nDisplacements:1);
        }
        //		LOG<<label<<" "<<label[k-1]<<" "<<index<<std::endl;
        //find out direction
        if (nDisplacements){
            itk::Vector<double,TImage::ImageDimension> sums;
            sums.Fill(0);
            for (int d=0;d<TImage::ImageDimension;++d){
                for (int d2=0;d2<TImage::ImageDimension;++d2){
                    if (d2!=d){
                        //					LOG<<d<<" "<<d2<<" "<<sums[d]<<std::endl;
                        sums[d]+=abs(label[d2]);//+nDisplacementSamples;
                    }
                }
                if (sums[d]==0){
                    //found it!
                    index+=(d)*(2*nDisplacementSamples+1)+label[d]+nDisplacementSamples;
                    break;
                }
            }
        }
        return index;
    }
    static inline const OffsetType getDisplacement(const LabelType & label){
        OffsetType off;
        for (int d=0;d<TImage::ImageDimension;++d){
            off[d]=label[d];
        }
        return off;
    }
    static inline const float getSegmentation(const LabelType & label){
        return label[k-1];
    }
    static inline const void setSegmentation(LabelType & label, int seg){
        label[k-1]=seg;
    }
    static inline const void setDisplacement(LabelType & label, const OffsetType &off){
        for (int d=0;d<TImage::ImageDimension;++d){
            label[d]=off[d];
        }
    }

};
template<class TImage, class TLabel>
class DenseRegistrationLabelMapper{
public:
	//	typedef typename TImage::OffsetType OffsetType;
	//	typedef typename itk::LinearInterpolateImageFunction<TImage>::ContinuousIndexType OffsetType;
	typedef typename itk::Vector<float,TImage::ImageDimension> OffsetType;
	typedef TLabel LabelType;
	typedef typename  itk::Image<LabelType,TImage::ImageDimension > LabelImageType;
	typedef typename LabelImageType::Pointer LabelImagePointerType;
	static int nLabels,nDisplacements,nSegmentations,nDisplacementSamples,k;
	static const int Dimension=TImage::ImageDimension;
    typedef typename TImage::SpacingType SpacingType;

	//private:
	//	int nLabels,nDisplacements,nSegmentations,nDisplacementSamples,k;
public:
	DenseRegistrationLabelMapper(){}
	DenseRegistrationLabelMapper(int NSegmentations, int NDisplacementSamples){
		nSegmentations=NSegmentations;
		nDisplacementSamples=NDisplacementSamples;
		nDisplacements=pow(double(2*nDisplacementSamples+1),TImage::ImageDimension);
		nLabels=nSegmentations*nDisplacements;
		k=TImage::ImageDimension;
	}
	void setDisplacementSamples(int nSamples){
		nDisplacementSamples=nSamples;
		nDisplacements=pow(double(2*nDisplacementSamples+1),TImage::ImageDimension);
		nLabels=nSegmentations*nDisplacements;
	}
	void setSegmentationLabels(int labels){
		nSegmentations=labels;
		nLabels=nSegmentations*nDisplacements;
	}
	static inline const LabelType scaleDisplacement(const LabelType & label,const itk::Vector<float,TImage::ImageDimension> & scaling){
		LabelType result(label);
		for (int d=0;d<TImage::ImageDimension;++d){
			result[d]=result[d]*scaling[d];
		}
		return result;
	}
    static inline const LabelType scaleDisplacement(const LabelType & label,const SpacingType & scaling){
		LabelType result(label);
		for (int d=0;d<TImage::ImageDimension;++d){
			result[d]=result[d]*scaling[d];
		}
		return result;
	}
    static inline int getZeroDisplacementIndex(){return nDisplacements/2;}
	static inline const LabelType getLabel(int index){
		LabelType result;
        int divisor=pow(double(2*nDisplacementSamples+1),TImage::ImageDimension-1);
		for (int d=TImage::ImageDimension-1;d>=0;--d){
			result[d]=index/divisor-nDisplacementSamples;
			index-=(result[d]+nDisplacementSamples)*divisor;
			divisor/=2*nDisplacementSamples+1;
		}
        return result;
	}

	static inline const int getIndex(const LabelType & label){
		int index=0;
        int factor=1;
		for (int d=0;d<TImage::ImageDimension;++d){
			index+=factor*(label[d]+nDisplacementSamples);
			factor*=2*nDisplacementSamples+1;
		}
		return index;
	}
	static inline const OffsetType getDisplacement(const LabelType & label){
		OffsetType off;
		for (int d=0;d<TImage::ImageDimension;++d){
			off[d]=label[d];
		}
		return off;
	}
};

template<class TImage, class TLabel>
class SparseRegistrationLabelMapper{
public:
	//	typedef typename TImage::OffsetType OffsetType;
	//	typedef typename itk::LinearInterpolateImageFunction<TImage>::ContinuousIndexType OffsetType;
	typedef typename itk::Vector<float,TImage::ImageDimension> OffsetType;
	typedef TLabel LabelType;
	typedef typename  itk::Image<LabelType,TImage::ImageDimension > LabelImageType;
	typedef typename LabelImageType::Pointer LabelImagePointerType;
	static int nLabels,nDisplacements,nSegmentations,nDisplacementSamples,k;
	static const int Dimension=TImage::ImageDimension;
    typedef typename TImage::SpacingType SpacingType;

	//private:
	//	int nLabels,nDisplacements,nSegmentations,nDisplacementSamples,k;
public:
	SparseRegistrationLabelMapper(){}
	SparseRegistrationLabelMapper(int NSegmentations, int NDisplacementSamples){
		nSegmentations=NSegmentations;
		nDisplacementSamples=NDisplacementSamples;
		nDisplacements=(double(2*nDisplacementSamples+1)*TImage::ImageDimension);
		nLabels=nSegmentations+nDisplacements;
		k=TImage::ImageDimension;
	}
	void setDisplacementSamples(int nSamples){
		nDisplacementSamples=nSamples;
		nDisplacements=(double(2*nDisplacementSamples+1)*TImage::ImageDimension);

		nLabels=nSegmentations*nDisplacements;
	}
    static inline int getZeroDisplacementIndex(){return nDisplacements/2;}

	void setSegmentationLabels(int labels){
		nSegmentations=labels;
		nLabels=nSegmentations+nDisplacements;
	}
	static inline const LabelType scaleDisplacement(const LabelType & label,const itk::Vector<float,TImage::ImageDimension> & scaling){
		LabelType result(label);
		for (int d=0;d<TImage::ImageDimension;++d){
			result[d]=result[d]*scaling[d];
		}
		return result;
	}
    static inline const LabelType scaleDisplacement(const LabelType & label,const SpacingType & scaling){
		LabelType result(label);
		for (int d=0;d<TImage::ImageDimension;++d){
			result[d]=result[d]*scaling[d];
		}
		return result;
	}

    static inline const LabelType getLabel(int index){
        LabelType result;
        result.Fill(0);
        int divisor=(double(2*nDisplacementSamples+1));
        result[index/divisor]=index%divisor-nDisplacementSamples;
        return result;
    }

    static inline const int getIndex(const LabelType & label){
        int index=0;
        itk::Vector<double,TImage::ImageDimension> sums;
        sums.Fill(0);
        for (int d=0;d<TImage::ImageDimension;++d){
            for (int d2=0;d2<TImage::ImageDimension;++d2){
                if (d2!=d){
                    sums[d]+=abs(label[d2]);
                }
            }
            if (sums[d]==0){
                //found it!
                index+=(d)*(2*nDisplacementSamples+1)+label[d]+nDisplacementSamples;
                break;
            }
        }
        return index;
    }
    
	static inline const OffsetType getDisplacement(const LabelType & label){
		OffsetType off;
		for (int d=0;d<TImage::ImageDimension;++d){
			off[d]=label[d];
		}
		return off;
	}
};
template<class TImage, class TLabel>
class SemiSparseRegistrationLabelMapper{
public:
	//	typedef typename TImage::OffsetType OffsetType;
	//	typedef typename itk::LinearInterpolateImageFunction<TImage>::ContinuousIndexType OffsetType;
	typedef typename itk::Vector<float,TImage::ImageDimension> OffsetType;
	typedef TLabel LabelType;
	typedef typename  itk::Image<LabelType,TImage::ImageDimension > LabelImageType;
	typedef typename LabelImageType::Pointer LabelImagePointerType;
	static int nLabels,nDisplacements,nSegmentations,nDisplacementSamples,k;
	static const int Dimension=TImage::ImageDimension;
    typedef typename TImage::SpacingType SpacingType;
    //typedef boost::bimap<unsigned short,LabelType,boost::bimaps::unconstrained_set_of_relation> BimapType;

    class cmp{
    public:
        bool operator()(const LabelType &l1, const LabelType &l2){
            for (unsigned short d=0;d<Dimension;++d){
                if (l1[d]<l2[d])
                    return true;
                else if (l1[d]>l2[d])
                    return false;
            }
            return false;
        }
    };
    typedef std::map<LabelType,unsigned short,cmp> LabelMapType;
    typedef std::vector<LabelType> LabelListType;

protected:
    //typedef typename MapType::value_type MapValueType;
    static LabelMapType * m_labelMap;
    static LabelListType * m_labelList;
	//private:
	//	int nLabels,nDisplacements,nSegmentations,nDisplacementSamples,k;
public:
	SemiSparseRegistrationLabelMapper(){}
	SemiSparseRegistrationLabelMapper(int NSegmentations, int NDisplacementSamples){
		nSegmentations=NSegmentations;
		nDisplacementSamples=NDisplacementSamples;
		nDisplacements=(pow(3,int(TImage::ImageDimension))-1)*nDisplacementSamples+1;
		nLabels=nSegmentations+nDisplacements;
		k=TImage::ImageDimension;
        populateLabelMap();
	}
protected:
    void populateLabelMap(){                   
        if (m_labelMap!=NULL){
            delete m_labelMap;
            delete m_labelList;
        }
        m_labelMap=new LabelMapType();
        m_labelList=new LabelListType();
       

#if 0
        //1-axis displacements
        LabelType label;
        label.Fill(0.0);
        LabelType zeroLabel=label;
        int increment=0;
        m_labelMap->insert(std::make_pair(label,increment));
        m_labelList->push_back(label);
        for (int d=0;d<Dimension;++d){
            for (int n=-nDisplacementSamples;n<=nDisplacementSamples;++n){
                if (n != 0){
                    label.Fill(0.0);
                    label[d]=n+1;
                    m_labelMap->insert(std::make_pair(label,increment));
                    m_labelList->push_back(label);
                    increment++;
                }
            }
        }
#endif
        LabelType zeroLabel;
        zeroLabel.Fill(0.0);
        LabelType label;
        label.Fill(-1.0);
        int i=0;
        for (;i<pow(3,int(TImage::ImageDimension));++i){
            m_labelMap->insert(std::make_pair(label,i));
            m_labelList->push_back(label);
            label[0]+=1;
            for (unsigned int d=0;d<TImage::ImageDimension-1;++d){
                if (label[d]>1){
                    label[d+1]+=1;
                    label[d]=-1;
                }
            }
        }
        for (int n=2;n<nDisplacementSamples+1;++n){
            for (int d=0;d<pow(3,int(TImage::ImageDimension));++d){
                const LabelType l= (*m_labelList)[d];
                LabelType label= scaleDisplacement( l , n );
                if (label!=zeroLabel){
                    m_labelMap->insert(std::make_pair(label,i));
                    m_labelList->push_back(label);
                    ++i;
                }
            }
        }
    }
public:
    static inline int getZeroDisplacementIndex(){return pow(3,int(TImage::ImageDimension))/2;}

	void setDisplacementSamples(int nSamples){
		nDisplacementSamples=nSamples;
			nDisplacements=(pow(3,int(TImage::ImageDimension))-1)*nDisplacementSamples+1;
		nLabels=nSegmentations*nDisplacements;
        populateLabelMap();

	}

    
	void setSegmentationLabels(int labels){
		nSegmentations=labels;
		nLabels=nSegmentations+nDisplacements;
	}
	static inline const LabelType scaleDisplacement(const LabelType & label,const itk::Vector<float,TImage::ImageDimension> & scaling){
		LabelType result(label);
		for (int d=0;d<TImage::ImageDimension;++d){
			result[d]=result[d]*scaling[d];
		}
		return result;
	}
	static inline const LabelType scaleDisplacement( const LabelType & label,const int & s){
		LabelType result(label);
		for (int d=0;d<TImage::ImageDimension;++d){
			result[d]=result[d]*s;
		}
		return result;
	}
    static inline const LabelType scaleDisplacement(const LabelType & label,const SpacingType & scaling){
		LabelType result(label);
		for (int d=0;d<TImage::ImageDimension;++d){
			result[d]=result[d]*scaling[d];
		}
		return result;
	}

    static inline const LabelType getLabel(int index){
        return (*m_labelList)[index];
    }

    static inline const int getIndex(const LabelType & label){
        return (*m_labelMap)[label];
    }
    
	static inline const OffsetType getDisplacement(const LabelType & label){
		OffsetType off;
		for (int d=0;d<TImage::ImageDimension;++d){
			off[d]=label[d];
		}
		return off;
	}
};
template<class T,class L> int  BaseLabelMapper<T,L>::nLabels=-1;
template<class T,class L> int  BaseLabelMapper<T,L>::nDisplacements=-1;
template<class T,class L> int  BaseLabelMapper<T,L>::nSegmentations=-1;
template<class T,class L> int  BaseLabelMapper<T,L>::nDisplacementSamples=-1;
template<class T,class L> int  BaseLabelMapper<T,L>::k=-1;
template<class T,class L> int  SparseLabelMapper<T,L>::nLabels=-1;
template<class T,class L> int  SparseLabelMapper<T,L>::nDisplacements=-1;
template<class T,class L> int  SparseLabelMapper<T,L>::nSegmentations=-1;
template<class T,class L> int  SparseLabelMapper<T,L>::nDisplacementSamples=-1;
template<class T,class L> int  SparseLabelMapper<T,L>::k=-1;
template<class T,class L> int  SparseRegistrationLabelMapper<T,L>::nLabels=-1;
template<class T,class L> int  SparseRegistrationLabelMapper<T,L>::nDisplacements=-1;
template<class T,class L> int  SparseRegistrationLabelMapper<T,L>::nDisplacementSamples=-1;
template<class T,class L> int  SparseRegistrationLabelMapper<T,L>::nSegmentations=-1;
template<class T,class L> int  SparseRegistrationLabelMapper<T,L>::k=-1;
template<class T,class L> int  DenseRegistrationLabelMapper<T,L>::nLabels=-1;
template<class T,class L> int  DenseRegistrationLabelMapper<T,L>::nDisplacements=-1;
template<class T,class L> int  DenseRegistrationLabelMapper<T,L>::nSegmentations=-1;
template<class T,class L> int  DenseRegistrationLabelMapper<T,L>::nDisplacementSamples=-1;
template<class T,class L> int  DenseRegistrationLabelMapper<T,L>::k=-1;
template<class T,class L> int  SemiSparseRegistrationLabelMapper<T,L>::nLabels=-1;
template<class T,class L> int  SemiSparseRegistrationLabelMapper<T,L>::nDisplacements=-1;
template<class T,class L> int  SemiSparseRegistrationLabelMapper<T,L>::nSegmentations=-1;
template<class T,class L> int  SemiSparseRegistrationLabelMapper<T,L>::nDisplacementSamples=-1;
template<class T,class L> int  SemiSparseRegistrationLabelMapper<T,L>::k=-1;
template<class T,class L> typename SemiSparseRegistrationLabelMapper<T,L>::LabelMapType *  SemiSparseRegistrationLabelMapper<T,L>::m_labelMap=NULL ;
template<class T,class L> typename SemiSparseRegistrationLabelMapper<T,L>::LabelListType *  SemiSparseRegistrationLabelMapper<T,L>::m_labelList=NULL ;



#endif /* LABEL_H_ */
