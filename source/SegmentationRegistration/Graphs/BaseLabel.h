#include "Log.h"
/*
 * Label.h
 *
 *  Created on: Nov 26, 2010
 *      Author: gasst
 */

#ifndef BASELABEL_H_
#define BASELABEL_H_
//#include "itkLinearInterpolateImageFunction.h"
//#include <boost/bimap.hpp>
//#include <boost/bimap/unconstrained_set_of.hpp>

template<class TImage, class TLabel>
class BaseLabelMapper{
public:
	typedef typename itk::Vector<float,TImage::ImageDimension> OffsetType;
	typedef TLabel LabelType;
	typedef typename  itk::Image<LabelType,TImage::ImageDimension > LabelImageType;
	typedef typename LabelImageType::Pointer LabelImagePointerType;
    static const int Dimension=TImage::ImageDimension+1;
    typedef typename TImage::SpacingType SpacingType;
protected:
    int m_nLabels,m_nDisplacements,m_nSegmentations,m_nDisplacementSamplesPerAxis,k;
    string descr;
public:
	BaseLabelMapper(){}
	BaseLabelMapper(int NSegmentations, int NDisplacementSamples){
		this->m_nSegmentations=NSegmentations;
		this->m_nDisplacementSamplesPerAxis=NDisplacementSamples;
		this->m_nDisplacements=pow(double(2*this->m_nDisplacementSamplesPerAxis+1),TImage::ImageDimension);
		this->m_nLabels=this->m_nSegmentations*this->m_nDisplacements;
		k=TImage::ImageDimension+1;
        LOGV(1)<<"Constructed base labelmapper (product label space), using "<<this->m_nLabels<< " total labels"<<endl;
        descr="BLM";
	}
    virtual int getNumberOfDisplacementSamplesPerAxis(){return this->m_nDisplacementSamplesPerAxis;}
	virtual void setNumberOfDisplacementSamplesPerAxis(int nSamples){
		this->m_nDisplacementSamplesPerAxis=nSamples;
		this->m_nDisplacements=pow(double(2*this->m_nDisplacementSamplesPerAxis+1),TImage::ImageDimension);
		this->m_nLabels=this->m_nSegmentations*this->m_nDisplacements;
	}
    virtual int getNumberOfDisplacementLabels(){LOGV(3)<<VAR(this->descr)<<", returning "<<VAR(this->m_nDisplacements)<<endl;return this->m_nDisplacements;}
    virtual inline int getZeroDisplacementIndex(){return this->m_nDisplacements/2;}
	virtual void setNumberOfSegmentationLabels(int labels){
		this->m_nSegmentations=labels;
		this->m_nLabels=this->m_nSegmentations*this->m_nDisplacements;
	}
    virtual int getNumberOfSegmentationLabels(){return this->m_nSegmentations;}
    virtual int getTotalNumberOfLabels(){return this->m_nLabels;}
    inline const LabelType scaleDisplacement(const LabelType & label,const itk::Vector<float,TImage::ImageDimension> & scaling){
		LabelType result(label);
		for (int d=0;d<TImage::ImageDimension;++d){
			result[d]=result[d]*scaling[d];
		}
		return result;
	}
    inline const LabelType scaleDisplacement(const LabelType & label,const SpacingType & scaling){
		LabelType result(label);
		for (int d=0;d<TImage::ImageDimension;++d){
			result[d]=result[d]*scaling[d];
		}
		return result;
	}
    virtual inline const LabelType getLabel(int index){
		LabelType result;
		int m_segmentation;
		if (this->m_nDisplacements){
			m_segmentation=index/this->m_nDisplacements;
			index=index%this->m_nDisplacements;
		}
		else{
			m_segmentation=index;
		}
		int divisor=pow(double(2*this->m_nDisplacementSamplesPerAxis+1),TImage::ImageDimension-1);
		for (int d=0;d<TImage::ImageDimension;++d){
			result[d]=index/divisor-this->m_nDisplacementSamplesPerAxis;
			index-=(result[d]+this->m_nDisplacementSamplesPerAxis)*divisor;
			divisor/=2*this->m_nDisplacementSamplesPerAxis+1;
		}
		result[k-1]=m_segmentation;
		return result;
	}

	 virtual inline const int getIndex(const LabelType & label){
		int index=0;
		index+=label[k-1]*(this->m_nDisplacements>0?this->m_nDisplacements:1);
		int factor=1;
		for (int d=TImage::ImageDimension-1;d>=0;--d){
			index+=factor*(label[d]+this->m_nDisplacementSamplesPerAxis);
			factor*=2*this->m_nDisplacementSamplesPerAxis+1;
		}
		return index;
	}
    
    //why is this offsettype??
    inline const OffsetType getDisplacement(const LabelType & label){
		OffsetType off;
		for (int d=0;d<TImage::ImageDimension;++d){
			off[d]=label[d];
		}
		return off;
	}
    
    
	virtual inline const float getSegmentation(const LabelType & label){
		return label[k-1];
	}
    
	virtual inline const void setSegmentation(LabelType & label, int seg){
		label[k-1]=seg;
	}

    //weird function :o
	virtual inline const void setDisplacement(LabelType & label, const OffsetType &off){
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
	//static int nLabels,this->m_nDisplacements,this->m_nSegmentations,this->m_nDisplacementSamplesPerAxis,k;
  
public:
	SparseLabelMapper(){}
	SparseLabelMapper(int NSegmentations, int NDisplacementSamples){
		this->m_nSegmentations=NSegmentations;
		this->m_nDisplacementSamplesPerAxis=NDisplacementSamples;
		this->m_nDisplacements=(double(2*this->m_nDisplacementSamplesPerAxis+1)*TImage::ImageDimension);
		this->m_nLabels=this->m_nSegmentations*this->m_nDisplacements;
		LOG<<"this->m_nDisplacements :"<<this->m_nDisplacements<<" this->m_nSegmentations:"<<this->m_nSegmentations<<" this->m_nLabels"<<this->m_nLabels<<std::endl;
		this->k=TImage::ImageDimension+1;
        //   setupLabelMap();
	}
    void setSegmentationLabels(int labels){
        this->m_nSegmentations=labels;
        this->m_nLabels=this->m_nSegmentations*this->m_nDisplacements;
    }
    void setDisplacementSamples(int nSamples){
        this->m_nDisplacementSamplesPerAxis=nSamples;
        if (this->m_nDisplacementSamplesPerAxis){
            this->m_nDisplacements=(double(2*this->m_nDisplacementSamplesPerAxis+1)*TImage::ImageDimension);
        }
        else
            this->m_nDisplacements=1;
        this->m_nLabels=this->m_nSegmentations*this->m_nDisplacements;
        LOG<<"this->m_nDisplacements :"<<this->m_nDisplacements<<" this->m_nSegmentations:"<<this->m_nSegmentations<<" nLabels"<<this->m_nLabels<<std::endl;

    }
     inline const LabelType getLabel(int index){
        //   return indexToLabelMap[index];
        LabelType result;
        result.Fill(0);
        int m_segmentation=0;
        if (this->m_nDisplacements){
            m_segmentation=index/this->m_nDisplacements;
            index=index%this->m_nDisplacements;
        }
        else if(this->m_nSegmentations){
            m_segmentation=index;
        }
        int divisor=(double(2*this->m_nDisplacementSamplesPerAxis+1));
        result[index/divisor]=index%divisor-this->m_nDisplacementSamplesPerAxis;
        result[this->k-1]=m_segmentation;
        return result;
    }

     inline const int getIndex(const LabelType & label){
        //    return labelToIndexMap[label];
        int index=0;
        if (this->m_nSegmentations){
            index+=label[this->k-1]*(this->m_nDisplacements>0?this->m_nDisplacements:1);
        }
        //		LOG<<label<<" "<<label[k-1]<<" "<<index<<std::endl;
        //find out direction
        if (this->m_nDisplacements){
            itk::Vector<double,TImage::ImageDimension> sums;
            sums.Fill(0);
            for (int d=0;d<TImage::ImageDimension;++d){
                for (int d2=0;d2<TImage::ImageDimension;++d2){
                    if (d2!=d){
                        //					LOG<<d<<" "<<d2<<" "<<sums[d]<<std::endl;
                        sums[d]+=abs(label[d2]);//+this->m_nDisplacementSamplesPerAxis;
                    }
                }
                if (sums[d]==0){
                    //found it!
                    index+=(d)*(2*this->m_nDisplacementSamplesPerAxis+1)+label[d]+this->m_nDisplacementSamplesPerAxis;
                    break;
                }
            }
        }
        return index;
    }
     inline const OffsetType getDisplacement(const LabelType & label){
        OffsetType off;
        for (int d=0;d<TImage::ImageDimension;++d){
            off[d]=label[d];
        }
        return off;
    }
     inline const float getSegmentation(const LabelType & label){
        return label[this->k-1];
    }
     inline const void setSegmentation(LabelType & label, int seg){
        label[this->k-1]=seg;
    }
     inline const void setDisplacement(LabelType & label, const OffsetType &off){
        for (int d=0;d<TImage::ImageDimension;++d){
            label[d]=off[d];
        }
    }

};
template<class TImage, class TLabel>
class DenseRegistrationLabelMapper : public BaseLabelMapper<TImage,TLabel>{
public:
	//	typedef typename TImage::OffsetType OffsetType;
	//	typedef typename itk::LinearInterpolateImageFunction<TImage>::ContinuousIndexType OffsetType;
	typedef typename itk::Vector<float,TImage::ImageDimension> OffsetType;
	typedef TLabel LabelType;
	typedef typename  itk::Image<LabelType,TImage::ImageDimension > LabelImageType;
	typedef typename LabelImageType::Pointer LabelImagePointerType;
	static const int Dimension=TImage::ImageDimension;
    typedef typename TImage::SpacingType SpacingType;

	//private:
	//	int nLabels,this->m_nDisplacements,this->m_nSegmentations,this->m_nDisplacementSamplesPerAxis,k;
public:
	DenseRegistrationLabelMapper(){}
	DenseRegistrationLabelMapper(int NSegmentations, int NDisplacementSamples){
		this->m_nSegmentations=NSegmentations;
		this->m_nDisplacementSamplesPerAxis=NDisplacementSamples;
		this->m_nDisplacements=pow(double(2*this->m_nDisplacementSamplesPerAxis+1),TImage::ImageDimension);
		this->m_nLabels=this->m_nSegmentations*this->m_nDisplacements;
		this->k=TImage::ImageDimension;
        LOGV(1)<<"Constructed Dense Label Mapper with "<<this->m_nDisplacements<<" registration labels"<<endl;
	}
	void setDisplacementSamples(int nSamples){
		this->m_nDisplacementSamplesPerAxis=nSamples;
		this->m_nDisplacements=pow(double(2*this->m_nDisplacementSamplesPerAxis+1),TImage::ImageDimension);
		this->m_nLabels=this->m_nSegmentations*this->m_nDisplacements;
	}
	void setSegmentationLabels(int labels){
		this->m_nSegmentations=labels;
		this->m_nLabels=this->m_nSegmentations*this->m_nDisplacements;
	}
	 inline const LabelType scaleDisplacement(const LabelType & label,const itk::Vector<float,TImage::ImageDimension> & scaling){
		LabelType result(label);
		for (int d=0;d<TImage::ImageDimension;++d){
			result[d]=result[d]*scaling[d];
		}
		return result;
	}
     inline const LabelType scaleDisplacement(const LabelType & label,const SpacingType & scaling){
		LabelType result(label);
		for (int d=0;d<TImage::ImageDimension;++d){
			result[d]=result[d]*scaling[d];
		}
		return result;
	}
    
	 inline const LabelType getLabel(int index){
		LabelType result;
        int divisor=pow(double(2*this->m_nDisplacementSamplesPerAxis+1),TImage::ImageDimension-1);
		for (int d=TImage::ImageDimension-1;d>=0;--d){
			result[d]=index/divisor-this->m_nDisplacementSamplesPerAxis;
			index-=(result[d]+this->m_nDisplacementSamplesPerAxis)*divisor;
			divisor/=2*this->m_nDisplacementSamplesPerAxis+1;
		}
        return result;
	}

	 inline const int getIndex(const LabelType & label){
		int index=0;
        int factor=1;
		for (int d=0;d<TImage::ImageDimension;++d){
			index+=factor*(label[d]+this->m_nDisplacementSamplesPerAxis);
			factor*=2*this->m_nDisplacementSamplesPerAxis+1;
		}
		return index;
	}
	 inline const OffsetType getDisplacement(const LabelType & label){
		OffsetType off;
		for (int d=0;d<TImage::ImageDimension;++d){
			off[d]=label[d];
		}
		return off;
	}
};

template<class TImage, class TLabel>
class SparseRegistrationLabelMapper : public BaseLabelMapper<TImage,TLabel>{
public:
	//	typedef typename TImage::OffsetType OffsetType;
	//	typedef typename itk::LinearInterpolateImageFunction<TImage>::ContinuousIndexType OffsetType;
	typedef typename itk::Vector<float,TImage::ImageDimension> OffsetType;
	typedef TLabel LabelType;
	typedef typename  itk::Image<LabelType,TImage::ImageDimension > LabelImageType;
	typedef typename LabelImageType::Pointer LabelImagePointerType;
    static const int Dimension=TImage::ImageDimension;
    typedef typename TImage::SpacingType SpacingType;

	//private:
	//	int nLabels,this->m_nDisplacements,this->m_nSegmentations,this->m_nDisplacementSamplesPerAxis,k;
public:
	SparseRegistrationLabelMapper(){}
	SparseRegistrationLabelMapper(int NSegmentations, int NDisplacementSamples){
		this->m_nSegmentations=NSegmentations;
		this->m_nDisplacementSamplesPerAxis=NDisplacementSamples;
		this->m_nDisplacements=(double(2*this->m_nDisplacementSamplesPerAxis+1)*TImage::ImageDimension);
		this->m_nLabels=this->m_nSegmentations+this->m_nDisplacements;
		this->k=TImage::ImageDimension;
        LOGV(1)<<"Constructed Sparse Label Mapper with "<<this->m_nDisplacements<<" registration labels"<<endl;
        this->descr="SLRM";
	}
    
	void setNumberOfDisplacementSamplesPerAxis(int nSamples){
		this->m_nDisplacementSamplesPerAxis=nSamples;
		this->m_nDisplacements=(double(2*this->m_nDisplacementSamplesPerAxis+1)*TImage::ImageDimension);

		this->m_nLabels=this->m_nSegmentations*this->m_nDisplacements;
	}
     inline int getZeroDisplacementIndex(){return this->m_nDisplacements/2;}

	void setSegmentationLabels(int labels){
		this->m_nSegmentations=labels;
		this->m_nLabels=this->m_nSegmentations+this->m_nDisplacements;
	}
	 inline const LabelType scaleDisplacement(const LabelType & label,const itk::Vector<float,TImage::ImageDimension> & scaling){
		LabelType result(label);
		for (int d=0;d<TImage::ImageDimension;++d){
			result[d]=result[d]*scaling[d];
		}
		return result;
	}
     inline const LabelType scaleDisplacement(const LabelType & label,const SpacingType & scaling){
		LabelType result(label);
		for (int d=0;d<TImage::ImageDimension;++d){
			result[d]=result[d]*scaling[d];
		}
		return result;
	}

     inline const LabelType getLabel(int index){
        LabelType result;
        result.Fill(0);
        int divisor=(double(2*this->m_nDisplacementSamplesPerAxis+1));
        result[index/divisor]=index%divisor-this->m_nDisplacementSamplesPerAxis;
        return result;
    }

     inline const int getIndex(const LabelType & label){
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
                index+=(d)*(2*this->m_nDisplacementSamplesPerAxis+1)+label[d]+this->m_nDisplacementSamplesPerAxis;
                break;
            }
        }
        return index;
    }
    
	 inline const OffsetType getDisplacement(const LabelType & label){
		OffsetType off;
		for (int d=0;d<TImage::ImageDimension;++d){
			off[d]=label[d];
		}
		return off;
	}
};
template<class TImage, class TLabel>
class SemiSparseRegistrationLabelMapper : public BaseLabelMapper<TImage,TLabel>{
public:
	//	typedef typename TImage::OffsetType OffsetType;
	//	typedef typename itk::LinearInterpolateImageFunction<TImage>::ContinuousIndexType OffsetType;
	typedef typename itk::Vector<float,TImage::ImageDimension> OffsetType;
	typedef TLabel LabelType;
	typedef typename  itk::Image<LabelType,TImage::ImageDimension > LabelImageType;
	typedef typename LabelImageType::Pointer LabelImagePointerType;

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
     LabelMapType * m_labelMap;
     LabelListType * m_labelList;
	//private:
	//	int nLabels,this->m_nDisplacements,this->m_nSegmentations,this->m_nDisplacementSamplesPerAxis,k;
public:
	SemiSparseRegistrationLabelMapper(){}
	SemiSparseRegistrationLabelMapper(int NSegmentations, int NDisplacementSamples){
		this->m_nSegmentations=NSegmentations;
		this->m_nDisplacementSamplesPerAxis=NDisplacementSamples;
		this->m_nDisplacements=(pow(3,int(TImage::ImageDimension))-1)*this->m_nDisplacementSamplesPerAxis+1;
		this->m_nLabels=this->m_nSegmentations+this->m_nDisplacements;
		this->k=TImage::ImageDimension;
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
            for (int n=-this->m_nDisplacementSamplesPerAxis;n<=this->m_nDisplacementSamplesPerAxis;++n){
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
        for (int n=2;n<this->m_nDisplacementSamplesPerAxis+1;++n){
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
     inline int getZeroDisplacementIndex(){return pow(3,int(TImage::ImageDimension))/2;}

	void setDisplacementSamples(int nSamples){
		this->m_nDisplacementSamplesPerAxis=nSamples;
			this->m_nDisplacements=(pow(3,int(TImage::ImageDimension))-1)*this->m_nDisplacementSamplesPerAxis+1;
		this->m_nLabels=this->m_nSegmentations*this->m_nDisplacements;
        populateLabelMap();

	}

    
	void setSegmentationLabels(int labels){
		this->m_nSegmentations=labels;
		this->m_nLabels=this->m_nSegmentations+this->m_nDisplacements;
	}
	 inline const LabelType scaleDisplacement(const LabelType & label,const itk::Vector<float,TImage::ImageDimension> & scaling){
		LabelType result(label);
		for (int d=0;d<TImage::ImageDimension;++d){
			result[d]=result[d]*scaling[d];
		}
		return result;
	}
	 inline const LabelType scaleDisplacement( const LabelType & label,const int & s){
		LabelType result(label);
		for (int d=0;d<TImage::ImageDimension;++d){
			result[d]=result[d]*s;
		}
		return result;
	}
     inline const LabelType scaleDisplacement(const LabelType & label,const SpacingType & scaling){
		LabelType result(label);
		for (int d=0;d<TImage::ImageDimension;++d){
			result[d]=result[d]*scaling[d];
		}
		return result;
	}

     inline const LabelType getLabel(int index){
        return (*m_labelList)[index];
    }

     inline const int getIndex(const LabelType & label){
        return (*m_labelMap)[label];
    }
    
	 inline const OffsetType getDisplacement(const LabelType & label){
		OffsetType off;
		for (int d=0;d<TImage::ImageDimension;++d){
			off[d]=label[d];
		}
		return off;
	}
};



#endif /* LABEL_H_ */
