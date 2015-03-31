#pragma once
#include "matrix.h"
#include "TransformationUtils.h"
#include "Log.h"
#include "ImageUtils.h"
#include <vector>
#include <sstream>

#include "SolverAQUIRCGlobal.h"

namespace CBRR{
template<class ImageType>
class SolverAQUIRCLocal: public SolverAQUIRCGlobal< ImageType>{
public:
    typedef typename  TransfUtils<ImageType>::DeformationFieldType DeformationFieldType;
    typedef typename  DeformationFieldType::Pointer DeformationFieldPointerType;
    typedef typename ImageUtils<ImageType>::FloatImagePointerType FloatImagePointerType;
    typedef typename ImageUtils<ImageType>::FloatImageType FloatImageType;
    typedef typename itk::ImageRegionIterator<typename ImageUtils<ImageType>::FloatImageType> FloatImageIterator;
    typedef typename itk::ImageRegionIterator<DeformationFieldType> DeformationFieldIterator;
    typedef typename DeformationFieldType::PixelType DeformationType;
    typedef typename ImageType::Pointer ImagePointerType;
    typedef typename ImageType::RegionType RegionType;
    typedef typename ImageType::IndexType IndexType;
    typedef typename ImageType::PointType PointType;
    typedef typename ImageType::OffsetType OffsetType;
    static const unsigned int D=ImageType::ImageDimension;
public:

    virtual void SetVariables(std::vector<string> * imageIDList, map< string, map <string, DeformationFieldPointerType> > * deformationCache, map< string, map <string, DeformationFieldPointerType> > * trueDeformations,ImagePointerType img){
        LOG<<"LOCAL  ERROR SOLVER"<<endl;
        this->m_imageIDList=imageIDList;
        this->m_deformationCache=deformationCache;
        this->m_numImages=imageIDList->size();
        this->m_nPixels=2*(*deformationCache)[(*imageIDList)[0]][(*imageIDList)[1]]->GetLargestPossibleRegion().GetNumberOfPixels( );
        this->m_nEqs= this->m_numImages*(this->m_numImages-1)*( this->m_numImages-2)*this->m_nPixels;
        this->m_nVars= this->m_numImages*(this->m_numImages-1)*this->m_nPixels;
        this->m_nNonZeroes=3*this->m_nEqs;
        this->m_trueDeformations=trueDeformations;
      
        if (img.IsNotNull()){
            this->m_regionOfInterest.SetSize(img->GetLargestPossibleRegion().GetSize());
            IndexType startIndex,nullIdx;
            nullIdx.Fill(0);
            PointType startPoint;
            img->TransformIndexToPhysicalPoint(nullIdx,startPoint);
            (*this->m_deformationCache)[(*this->m_imageIDList)[0]][(*this->m_imageIDList)[1]]->TransformPhysicalPointToIndex(startPoint,startIndex);
            this->m_regionOfInterest.SetIndex(startIndex);
        }else{
            this->m_regionOfInterest=  (*this->m_deformationCache)[(*this->m_imageIDList)[0]][(*this->m_imageIDList)[1]]->GetLargestPossibleRegion();
        }

        if (trueDeformations!=NULL)
            this->computeError(deformationCache);
        m_regWeight=0.0;

    }
     void computeError(map< string, map <string, DeformationFieldPointerType> > * deformationCache){
        double residual=0.0;
        int c=0;
        for (int s = 0;s<this->m_numImages;++s){
            for (int t=0;t<this->m_numImages;++t){
                if (s!=t){
                    DeformationFieldIterator itTrueDef((*this->m_trueDeformations)[(*this->m_imageIDList)[s]][(*this->m_imageIDList)[t]],m_regionOfInterest);
                    DeformationFieldIterator itOriginalDef((*deformationCache)[(*this->m_imageIDList)[s]][(*this->m_imageIDList)[t]],m_regionOfInterest);
                    itOriginalDef.GoToBegin();
                    itTrueDef.GoToBegin();
                    for (;!itTrueDef.IsAtEnd();++itTrueDef,++itOriginalDef){
                        residual+=(itOriginalDef.Get()-itTrueDef.Get()).GetNorm();
                        ++c;
                    }

                }
            }
        }
        residual/=c;
        LOG<<VAR(residual)<<" "<<endl;

    }
    void SetRegularization(double lambda){
        m_regWeight=lambda;
        if (lambda>0.0){
            this->m_nEqs+= this->m_numImages *(this->m_numImages-1) *this->m_nPixels *D ;
            this->m_nNonZeroes+= this->m_numImages*(this->m_numImages-1) *this->m_nPixels *D*3;
        }
    }
    virtual void createSystem(){
        this->haveInit=false;
        LOG<<"Creating equation system.."<<endl;
        LOG<<VAR(this->m_numImages)<<" "<<VAR(this->m_nPixels)<<" "<<VAR(this->m_nEqs)<<" "<<VAR(this->m_nVars)<<" "<<VAR(this->m_nNonZeroes)<<endl;
        mxArray *mxX=mxCreateDoubleMatrix(this->m_nNonZeroes,1,mxREAL);
        mxArray *mxY=mxCreateDoubleMatrix(this->m_nNonZeroes,1,mxREAL);
        mxArray *mxV=mxCreateDoubleMatrix(this->m_nNonZeroes,1,mxREAL);
        mxArray *mxB=mxCreateDoubleMatrix(this->m_nEqs,1,mxREAL);
        
        double * x=( double *)mxGetData(mxX);
        double * y=( double *)mxGetData(mxY);
        double * v=( double *)mxGetData(mxV);
        double * b=mxGetPr(mxB);
        
        LOG<<"creating"<<endl;
        {
            ostringstream evalstr;
            evalstr<<"A = sparse([],[],[],"<<this->m_nEqs<<","<<this->m_nVars<<","<<this->m_nNonZeroes<<");";
            engEvalString(this->m_ep,evalstr.str().c_str() );
        }

        {
            ostringstream evalstr;
            evalstr<<" b=zeros(1,"<<this->m_nEqs<<");";
            engEvalString(this->m_ep,evalstr.str().c_str() );
        }


        char buffer[256+1];
        buffer[256] = '\0';
        engOutputBuffer(this->m_ep, buffer, 256);
      
        //attention matlab index convention?!?
        int eq = 1;
        //create edge index storage object
        int c=0;
        int maxE=0;
        std::vector<double> weights=this->getCircleWeights(0.0);

        for (int s = 0;s<this->m_numImages;++s){
            int source=s;
            for (int i=0;i<this->m_numImages;++i){
                if (i!=s){
                    int intermediate=i;
                    for (int t=0;t<this->m_numImages;++t){
                        if (t!=i && t!=s){
                            int target=t;

                            //compute circle
                            DeformationFieldPointerType circle=composeDeformations((*this->m_deformationCache)[(*this->m_imageIDList)[source]][(*this->m_imageIDList)[intermediate]],
                                                                                   (*this->m_deformationCache)[(*this->m_imageIDList)[intermediate]][(*this->m_imageIDList)[target]],
                                                                                   (*this->m_deformationCache)[(*this->m_imageIDList)[target]][(*this->m_imageIDList)[source]]);
                            //compute norm
                            //double norm=TransfUtils<ImageType>::computeDeformationNorm(circle,1.0);
                            DeformationFieldIterator it(circle,circle->GetLargestPossibleRegion());
                            it.GoToBegin();
                            for (int p=0;!it.IsAtEnd();++it){
                                DeformationType localDef=it.Get();
                                IndexType idx=it.GetIndex();
                                long int e1=edgeNum(source,intermediate,idx)+1;
                                long int e2=edgeNum(intermediate,target,idx)+1;
                                long int e3=edgeNum(target,source,idx)+1;
                                for (unsigned int d=0;d<D;++d,++p){
                                    double def=localDef[d];
                                    LOGV(7)<<VAR(source)<<" "<<VAR(intermediate)<<" "<<VAR(eq)<<" "<<VAR(c+3)<<" "<<endl;
                                    //set sparse entries
                                    x[c]=eq;
                                    y[c]=e1+d;
                                    //LOG<<VAR(source)<<" "<<VAR(intermediate)<<" "<<VAR(p)<<" "<<VAR(edgeNum(source,intermediate,p))<<endl;
                                    v[c++]=weights[0];
                                    x[c]=eq;
                                    y[c]=e2+d;
                                    v[c++]=weights[1];
                                    x[c]=eq;
                                    y[c]=e3+d;
                                    v[c++]=weights[2];
                                    LOGV(8)<<VAR(def)<<endl;
                                    //set rhs
				    //add small constant to avoid floating point exception.
				    //constant should be insignificant in terms of expected registration quality (in mm)
                                    b[eq-1]=log(fabs(def)+0.0067);
                                    ++eq;

                                   

                                }
                            }//iterator
                        }//if
                    }//target
                    
                    if (m_regWeight>0.0){
                        DeformationFieldPointerType defSourceInterm=(*this->m_deformationCache)[(*this->m_imageIDList)[source]][(*this->m_imageIDList)[intermediate]];
                        DeformationFieldIterator it(defSourceInterm,defSourceInterm->GetLargestPossibleRegion());
                        it.GoToBegin();
                        for (;!it.IsAtEnd();++it){
                            DeformationType localDef=it.Get();
                            IndexType idx=it.GetIndex();
                            long int e=edgeNum(source,intermediate,idx)+1;
                            for (int n=0;n<D;++n){
                                OffsetType off;
                                off.Fill(0);
                                off[n]=1;
                                IndexType neighborIndex=idx+off;
                                if (defSourceInterm->GetLargestPossibleRegion().IsInside(neighborIndex)){
                                    long int eNeighbor=edgeNum(source,intermediate,neighborIndex)+1;
                                    DeformationType neighborDef=defSourceInterm->GetPixel(neighborIndex);
                                    LOGV(6)<<""<<VAR(idx)<<" "<<VAR(neighborIndex)<<" "<<VAR(e)<<" "<<VAR(eNeighbor)<<" "<<endl;
                                    for (unsigned int d=0;d<D;++d){
                                        LOGV(7)<<"regularizing... "<<VAR(source)<<" "<<VAR(intermediate)<<" "<<VAR(eq)<<" "<<VAR(c+3)<<" "<<endl;
                                        double def=localDef[d];
                                        double defNeighbor=neighborDef[d];
                                        x[c]=eq;
                                        y[c]=e+d;
                                        v[c++]=-m_regWeight;
                                        x[c]=eq;
                                        y[c]=eNeighbor+d;
                                        v[c++]=m_regWeight;
                                        b[eq-1]=0;//m_regWeight*(defNeighbor-def);
                                        ++eq;
                                    }
                                }//inside

                            }//neighbors
                        }//for
                    }//regularization
                }//if
            }//intermediate
        }//source
        LOG<<VAR(c)<<" "<<VAR(eq)<<endl;
        this->m_nNonZeroes=c;
        mxSetM(mxX,c);
        mxSetM(mxY,c);
        mxSetM(mxV,c);
        mxSetM(mxB,eq-1);
        
        for (long int i=0;i<this->m_nNonZeroes;++i){
            if (x[i]<1 || y[i]<1){
                LOG<<VAR(i)<<" "<<VAR(x[i])<<" "<<VAR(y[i])<<endl;
            }
        }
        //put variables into workspace and immediately destroy them
        engPutVariable(this->m_ep,"xCord",mxX);
        mxDestroyArray(mxX);
        engPutVariable(this->m_ep,"yCord",mxY);
        mxDestroyArray(mxY);
        engPutVariable(this->m_ep,"val",mxV);
        mxDestroyArray(mxV);
        engPutVariable(this->m_ep,"b",mxB);
        mxDestroyArray(mxB);
        engEvalString(this->m_ep,"A=sparse(xCord,yCord,val);" );
        //clear unnneeded variables from matlab workspace
        engEvalString(this->m_ep,"clear xCord yCord val;" );

    }

    virtual void storeResult(string directory,string method){
        std::vector<double> result(this->m_nVars);
        double * rData=mxGetPr(this->m_result);
        for (int s = 0;s<this->m_numImages;++s){
            for (int t=0;t<this->m_numImages;++t){
                if (s!=t){
                    DeformationFieldPointerType estimatedError=ImageUtils<DeformationFieldType>::createEmpty((*this->m_deformationCache)[(*this->m_imageIDList)[s]][(*this->m_imageIDList)[t]]);
                    DeformationFieldIterator it(estimatedError,estimatedError->GetLargestPossibleRegion());
                    DeformationFieldIterator itOriginalDef((*this->m_deformationCache)[(*this->m_imageIDList)[s]][(*this->m_imageIDList)[t]],(*this->m_deformationCache)[(*this->m_imageIDList)[s]][(*this->m_imageIDList)[t]]->GetLargestPossibleRegion());
                    itOriginalDef.GoToBegin();
                    DeformationFieldIterator itTrueDef;
                    if ((*this->m_trueDeformations)[(*this->m_imageIDList)[s]][(*this->m_imageIDList)[t]].IsNotNull()){
                        itTrueDef=DeformationFieldIterator((*this->m_trueDeformations)[(*this->m_imageIDList)[s]][(*this->m_imageIDList)[t]],(*this->m_deformationCache)[(*this->m_imageIDList)[s]][(*this->m_imageIDList)[t]]->GetLargestPossibleRegion());
                        itTrueDef.GoToBegin();
                    }
                    it.GoToBegin();
                    for (int p=0;!it.IsAtEnd();++it){
                        DeformationType disp;
                        IndexType idx=it.GetIndex();
                        float localErrorNorm=0.0;
                        for (unsigned int d=0;d<D;++d,++p){
                            double tmp=rData[edgeNum(s,t,idx)+d];
                            disp[d]=tmp;
                            tmp=exp(tmp);
                            tmp*=tmp;
                            localErrorNorm+=tmp;
                            
                        }
                        double trueErrorMagnitude=0.0;
                        if ((*this->m_trueDeformations)[(*this->m_imageIDList)[s]][(*this->m_imageIDList)[t]].IsNotNull()){
                            trueErrorMagnitude=(itOriginalDef.Get()-itTrueDef.Get()).GetNorm();
                            ++itOriginalDef;
                            ++itTrueDef;
                        }
                        localErrorNorm=sqrt(localErrorNorm);
                        LOGV(3)<<VAR(localErrorNorm)<<" "<<VAR(trueErrorMagnitude)<<endl;
                        it.Set(disp);
                    }

                    ostringstream outfile;
                    outfile<<directory<<"/estimatedError-"<<method<<"-FROM-"<<(*this->m_imageIDList)[s]<<"-TO-"<<(*this->m_imageIDList)[t]<<".mha";
                    ImageUtils<DeformationFieldType>::writeImage(outfile.str().c_str(),estimatedError);
                }
            }
        }

    }

    virtual map< string, map <string, DeformationFieldPointerType> > * getEstimatedDeformations(){
        map< string, map <string, DeformationFieldPointerType> > * result=new map< string, map <string, DeformationFieldPointerType> >;
        double * rData=mxGetPr(this->m_result);
        for (int s = 0;s<this->m_numImages;++s){
            for (int t=0;t<this->m_numImages;++t){
                if (s!=t){
                    DeformationFieldPointerType estimatedDef=ImageUtils<DeformationFieldType>::createEmpty((*this->m_deformationCache)[(*this->m_imageIDList)[s]][(*this->m_imageIDList)[t]]);
                    DeformationFieldIterator it(estimatedDef,estimatedDef->GetLargestPossibleRegion());
                    DeformationFieldIterator itOriginalDef((*this->m_deformationCache)[(*this->m_imageIDList)[s]][(*this->m_imageIDList)[t]],(*this->m_deformationCache)[(*this->m_imageIDList)[s]][(*this->m_imageIDList)[t]]->GetLargestPossibleRegion());
                    itOriginalDef.GoToBegin();
                    it.GoToBegin();
                    for (int p=0;!it.IsAtEnd();++it,++itOriginalDef){
                        DeformationType disp;
                        IndexType idx=it.GetIndex();

                        for (unsigned int d=0;d<D;++d,++p){
                            disp[d]=rData[edgeNum(s,t,idx)+d];
                        }
                        it.Set(itOriginalDef.Get()-disp);
                    }

                    (*result)[(*this->m_imageIDList)[s]][(*this->m_imageIDList)[t]]=estimatedDef;
                }
            }
        }
        return result;
    }

protected:
    int m_nPixels;
    RegionType m_regionOfInterest;
    double m_regWeight;

protected:

    inline long int edgeNum(int n1,int n2){ return ((n1)*(this->m_numImages-1) + n2 - (n2>n1));}
    //return edgenumber after taking into acount nPixel*2 edges per image pair
    inline long int edgeNum(int n1,int n2,IndexType idx){ 
        long int offset = (*this->m_deformationCache)[(*this->m_imageIDList)[n1]][(*this->m_imageIDList)[n2]]->ComputeOffset(idx);
        return offset*2+edgeNum(n1,n2)*this->m_nPixels ;
    }
    //inline int edgeNum(int n1,int n2,int p){ return ((n1)*(this->m_numImages-1) + n2 - (n2>n1))*(this->m_nPixels) +p +1;}
    inline void edges(int edgeNum, int &n1, int &n2){
        n1 = edgeNum/(this->m_numImages-1);
        n2 =edgeNum%(this->m_numImages-1);
        if (n2 ==0){
            n2=(this->m_numImages-1);
            n1--;
        }
        if (n2>n1) ++n2;
        n2--;
        
    }


    
};

}//namespace
