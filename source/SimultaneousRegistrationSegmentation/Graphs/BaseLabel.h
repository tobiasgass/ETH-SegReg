/**
 * @file   BaseLabel.h
 * @author Tobias Gass <tobiasgass@gmail.com>
 * @date   Fri Mar  6 10:19:24 2015
 * 
 * @brief  LabelMappers function as a tool to convert discrete integer labels (0...n-1) to semantic labels, eg displacement vectors for registration
 * 
 * 
 */

#include "Log.h"
/*
 * Label.h
 *
 *  Created on: Nov 26, 2010
 *      Author: gasst
 */

#ifndef BASELABEL_H_
#define BASELABEL_H_

namespace SRS{

///\brief Basic function for mapping integer labels to displacement vectors and segmentation labels
///This class operates on the product label space, meaning the total number of labels is #displacement vectors * #segmentation labels
///Labels are vectors of size Imagedimension+1, where the first elements are the entries of the displacementvector, and the last entry is the segmentation label
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
  std::string descr;
 public:
  BaseLabelMapper(){}

  ///Constructor which sets internal parameters about the total number of labels 
  BaseLabelMapper(int NSegmentations, int NDisplacementSamplesPerAxis){
    this->m_nSegmentations=NSegmentations;
    this->m_nDisplacementSamplesPerAxis=NDisplacementSamplesPerAxis;
    this->m_nDisplacements=pow(double(2*this->m_nDisplacementSamplesPerAxis+1),TImage::ImageDimension);
    this->m_nLabels=this->m_nSegmentations*this->m_nDisplacements;
    k=TImage::ImageDimension+1;
    LOGV(1)<<"Constructed base labelmapper (product label space), using "<<this->m_nLabels<< " total labels"<<std::endl;
    descr="BLM";
  }
      virtual ~BaseLabelMapper(){}
  virtual int getNumberOfDisplacementSamplesPerAxis(){return this->m_nDisplacementSamplesPerAxis;}

  virtual void setNumberOfDisplacementSamplesPerAxis(int nSamples){
    this->m_nDisplacementSamplesPerAxis=nSamples;
    this->m_nDisplacements=pow(double(2*this->m_nDisplacementSamplesPerAxis+1),TImage::ImageDimension);
    this->m_nLabels=this->m_nSegmentations*this->m_nDisplacements;
  }

  virtual int getNumberOfDisplacementLabels(){LOGV(3)<<VAR(this->descr)<<", returning "<<VAR(this->m_nDisplacements)<<std::endl;return this->m_nDisplacements;}
  ///return discrete label which corresponds to the (0,0,0) displacement vector
  virtual inline int getZeroDisplacementIndex(){return this->m_nDisplacements/2;}
   inline LabelType getZeroDisplacement(){
     LabelType zeroDisp;
     zeroDisp.Fill(0.0);
     return zeroDisp;
   }

  virtual void setNumberOfSegmentationLabels(int labels){
    this->m_nSegmentations=labels;
    this->m_nLabels=this->m_nSegmentations*this->m_nDisplacements;
  }
  virtual int getNumberOfSegmentationLabels(){return this->m_nSegmentations;}

  virtual int getTotalNumberOfLabels(){return this->m_nLabels;}

  ///method for scaling the first elements of the vector by a set of factors, leaving the segmentation label unchanged
  inline const LabelType scaleDisplacement(const LabelType & label,const itk::Vector<float,TImage::ImageDimension> & scaling){
    LabelType result(label);
    for (int d=0;d<TImage::ImageDimension;++d){
      result[d]=result[d]*scaling[d];
    }
    return result;
  }

  ///method for scaling the first elements of the vector by a set of factors, leaving the segmentation label unchanged
  inline const LabelType scaleDisplacement(const LabelType & label,const SpacingType & scaling){
    LabelType result(label);
    for (int d=0;d<TImage::ImageDimension;++d){
      result[d]=result[d]*scaling[d];
    }
    return result;
  }

  ///Core function of the class, return the semantic label for a given discrete label index 
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

#if 0
template<class TLabel>
    struct comp:  itk::binary_function <TLabel,TLabel,bool>{
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
#endif

///\brief A LabelMapper which implements the sparse sampling scheme for displacement vector labels
///Displacment vectors are only sampled from the 2/3 principal axes of the image space, greatly reducing the number of displacement samples
template<class TImage, class TLabel>
  class SparseLabelMapper : public BaseLabelMapper<TImage,TLabel>{
 public:
    typedef typename itk::Vector<float,TImage::ImageDimension> OffsetType;
    typedef TLabel LabelType;
    typedef typename  itk::Image<LabelType, TImage::ImageDimension> LabelImageType;
    typedef typename LabelImageType::Pointer LabelImagePointerType;
  
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


///\brief Labelmapper that operates on the sumlabel space, eg displacement and segmentation labels are independent. 
///Dense mapping, only for displacment vectors
template<class TImage, class TLabel>
  class DenseRegistrationLabelMapper : public BaseLabelMapper<TImage,TLabel>{
 public:
    typedef typename itk::Vector<float,TImage::ImageDimension> OffsetType;
    typedef TLabel LabelType;
    typedef typename  itk::Image<LabelType,TImage::ImageDimension > LabelImageType;
    typedef typename LabelImageType::Pointer LabelImagePointerType;
    static const int Dimension=TImage::ImageDimension;
    typedef typename TImage::SpacingType SpacingType;
 public:
    DenseRegistrationLabelMapper(){}
    DenseRegistrationLabelMapper(int NSegmentations, int NDisplacementSamples){
      this->m_nSegmentations=NSegmentations;
      this->m_nDisplacementSamplesPerAxis=NDisplacementSamples;
      this->m_nDisplacements=pow(double(2*this->m_nDisplacementSamplesPerAxis+1),TImage::ImageDimension);
      this->m_nLabels=this->m_nSegmentations*this->m_nDisplacements;
      this->k=TImage::ImageDimension;
      LOGV(1)<<"Constructed Dense Label Mapper with "<<this->m_nDisplacements<<" registration labels"<<std::endl;
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

///\brief Labelmapper that operates on the sumlabel space, eg displacement and segmentation labels are independent. 
///Mapping only for displacment vectors
template<class TImage, class TLabel>
  class SparseRegistrationLabelMapper : public BaseLabelMapper<TImage,TLabel>{
 public:
    typedef typename itk::Vector<float,TImage::ImageDimension> OffsetType;
    typedef TLabel LabelType;
    typedef typename  itk::Image<LabelType,TImage::ImageDimension > LabelImageType;
    typedef typename LabelImageType::Pointer LabelImagePointerType;
    static const int Dimension=TImage::ImageDimension;
    typedef typename TImage::SpacingType SpacingType;
 public:
    SparseRegistrationLabelMapper(){}
    SparseRegistrationLabelMapper(int NSegmentations, int NDisplacementSamples){
      this->m_nSegmentations=NSegmentations;
      this->m_nDisplacementSamplesPerAxis=NDisplacementSamples;
      this->m_nDisplacements=(double(2*this->m_nDisplacementSamplesPerAxis+1)*TImage::ImageDimension);
      this->m_nLabels=this->m_nSegmentations+this->m_nDisplacements;
      this->k=TImage::ImageDimension;
      LOGV(1)<<"Constructed Sparse Label Mapper with "<<this->m_nDisplacements<<" registration labels"<<std::endl;
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

///\brief Labelmapper which also adds the diagonals of each quadrant to the sampling of displacement vectors
///UNTESTED
template<class TImage, class TLabel>
  class SemiSparseRegistrationLabelMapper : public BaseLabelMapper<TImage,TLabel>{
 public:
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

}//namespace

#endif /* LABEL_H_ */
