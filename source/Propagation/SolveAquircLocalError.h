#pragma once
#include "matrix.h"
#include "BaseLinearSolver.h"
#include "TransformationUtils.h"
#include "Log.h"
#include "ImageUtils.h"
#include <vector>
#include <sstream>

#include "SolveAquircGlobalDeformationNormCVariables.h"

template<class ImageType>
class AquircLocalErrorSolver: public AquircGlobalDeformationNormSolverCVariables< ImageType>{
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
   
    virtual void createSystem(){

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
            for (int i=s+1;i<this->m_numImages;++i){
                for (int t=i+1;t<this->m_numImages;++t){
                    //define a set of 3 images
                    //there are 6 possible circles with 3 images
                    int source=s;
                    int intermediate=i;
                    int target=t;
                    //       LOG<<VAR(s)<<" "<<VAR(i)<<" "<<VAR(t)<<endl;
                    for (int dir=0;dir<2;++dir){ //forwar-backward
                        double normSum=0.0;
                        for (int start=0;start<3;++start){
                            if (intermediate<0 || target<0 || intermediate>this->m_numImages || target>this->m_numImages) break;

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

                                for (unsigned int d=0;d<D;++d,++p){
                                    double def=localDef[d];
                                  
                                    //set sparse entries
                                    x[c]=eq;
                                    y[c]=edgeNum(source,intermediate,p);
                                    //LOG<<VAR(source)<<" "<<VAR(intermediate)<<" "<<VAR(p)<<" "<<VAR(edgeNum(source,intermediate,p))<<endl;
                                    v[c++]=weights[0];
                                    x[c]=eq;
                                    y[c]=edgeNum(intermediate,target,p);
                                    v[c++]=weights[1];
                                    x[c]=eq;
                                    y[c]=edgeNum(target,source,p);
                                    v[c++]=weights[2];
                                    maxE=max(maxE,max(edgeNum(source,intermediate,p),max(edgeNum(intermediate,target,p),edgeNum(target,source,p))));

                                    //set rhs
                                    b[eq-1]=def;
                                    ++eq;
                                }
                            }
                            
                            
                            
                            //shift start point
                            int tmpInt=source;
                            source=intermediate;
                            intermediate=target;
                            target=tmpInt;
                        }
                      
                        int tmp=target;
                        target=intermediate;
                        intermediate=tmp;

                    }

                
                }
            }
        }
        LOG<<VAR(maxE)<<endl;

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

    virtual void storeResult(string directory){
        std::vector<double> result(this->m_nVars);
        double * rData=mxGetPr(this->m_result);
        for (int s = 0;s<this->m_numImages;++s){
            for (int t=0;t<this->m_numImages;++t){
                if (s!=t){
                    DeformationFieldPointerType estimatedError=ImageUtils<DeformationFieldType>::createEmpty((*this->m_deformationCache)[(*this->m_imageIDList)[s]][(*this->m_imageIDList)[t]]);
                    DeformationFieldIterator it(estimatedError,estimatedError->GetLargestPossibleRegion());
                    it.GoToBegin();
                    for (int p=0;!it.IsAtEnd();++it){
                        DeformationType disp;
                        for (unsigned int d=0;d<D;++d,++p){
                            disp[d]=rData[edgeNum(s,t,p)-1];
                        }
                        it.Set(disp);
                    }

                    ostringstream outfile;
                    outfile<<directory<<"/estimatedLocalDeformationError-FROM-"<<(*this->m_imageIDList)[s]<<"-TO-"<<(*this->m_imageIDList)[t]<<".mha";
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
                        for (unsigned int d=0;d<D;++d,++p){
                            disp[d]=rData[edgeNum(s,t,p)-1];
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


protected:
    inline int edgeNum(int n1,int n2,int p){ return ((n1)*(this->m_numImages-1) + n2 - (n2>n1))*(this->m_nPixels) +p +1;}
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
