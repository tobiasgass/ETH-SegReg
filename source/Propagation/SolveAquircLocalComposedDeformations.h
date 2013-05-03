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
class AquircLocalComposedDeformationSolver: public AquircGlobalDeformationNormSolverCVariables< ImageType>{
public:
    typedef typename  TransfUtils<ImageType>::DeformationFieldType DeformationFieldType;
    typedef typename  DeformationFieldType::Pointer DeformationFieldPointerType;
    typedef typename ImageUtils<ImageType>::FloatImagePointerType FloatImagePointerType;
    typedef typename ImageUtils<ImageType>::FloatImageType FloatImageType;
    typedef typename itk::ImageRegionIterator<FloatImageType> FloatImageIterator;
    typedef typename itk::ImageRegionIteratorWithIndex<DeformationFieldType> DeformationFieldIterator;
    typedef typename DeformationFieldType::PixelType DeformationType;
    typedef typename DeformationFieldType::IndexType IndexType;
    typedef typename DeformationFieldType::PointType PointType;
    typedef typename ImageType::Pointer ImagePointerType;
    static const unsigned int D=ImageType::ImageDimension;
public:

    virtual void SetVariables(std::vector<string> * imageIDList, map< string, map <string, DeformationFieldPointerType> > * deformationCache, map< string, map <string, DeformationFieldPointerType> > * trueDeformations=NULL){
        m_imageIDList=imageIDList;
        m_deformationCache=deformationCache;
        m_numImages=imageIDList->size();
        m_nPixels=2*(*deformationCache)[(*imageIDList)[0]][(*imageIDList)[1]]->GetLargestPossibleRegion().GetNumberOfPixels( );
        m_nEqs= m_numImages*(m_numImages-1)*( m_numImages-2)*m_nPixels;
        m_nVars= m_numImages*(m_numImages-1)*m_nPixels;
        m_nNonZeroes=3*m_nEqs;
        m_trueDeformations=trueDeformations;
        if (trueDeformations!=NULL)
            this->computeError(deformationCache);
    }
    
    virtual void createSystem(){

        LOG<<"Creating equation system.."<<endl;
        LOG<<VAR(m_numImages)<<" "<<VAR(m_nPixels)<<" "<<VAR(m_nEqs)<<" "<<VAR(m_nVars)<<" "<<VAR(m_nNonZeroes)<<endl;
        mxArray *mxX=mxCreateDoubleMatrix(m_nNonZeroes,1,mxREAL);
        mxArray *mxY=mxCreateDoubleMatrix(m_nNonZeroes,1,mxREAL);
        mxArray *mxV=mxCreateDoubleMatrix(m_nNonZeroes,1,mxREAL);
        mxArray *mxB=mxCreateDoubleMatrix(m_nEqs,1,mxREAL);
        
        double * x=( double *)mxGetData(mxX);
        std::fill(x,x+m_nNonZeroes,1);
        double * y=( double *)mxGetData(mxY);
        std::fill(y,y+m_nNonZeroes,1);
        double * v=( double *)mxGetData(mxV);
        double * b=mxGetPr(mxB);
        
        LOG<<"creating"<<endl;
      

        char buffer[256+1];
        buffer[256] = '\0';
        engOutputBuffer(this->m_ep, buffer, 256);
      
        //attention matlab index convention?!?
        long int eq = 1;
        long int c=0;
        long int maxE=0;
        for (int s = 0;s<m_numImages;++s){
            for (int i=s+1;i<m_numImages;++i){
                for (int t=i+1;t<m_numImages;++t){
                    //define a set of 3 images
                    //there are 6 possible circles with 3 images
                    int source=s;
                    int intermediate=i;
                    int target=t;
                    //   LOG<<VAR(s)<<" "<<VAR(i)<<" "<<VAR(t)<<endl;
                    for (int dir=0;dir<2;++dir){ //forwar-backward
                        double normSum=0.0;
                        for (int start=0;start<3;++start){
                            if (intermediate<0 || target<0 || intermediate>m_numImages || target>m_numImages) break;

                            DeformationFieldPointerType d1=(*m_deformationCache)[(*m_imageIDList)[source]][(*m_imageIDList)[intermediate]];
                            DeformationFieldPointerType d2=(*m_deformationCache)[(*m_imageIDList)[intermediate]][(*m_imageIDList)[target]];
                            DeformationFieldPointerType d3=(*m_deformationCache)[(*m_imageIDList)[target]][(*m_imageIDList)[source]];

                            DeformationFieldPointerType hatd1,hatd2,hatd3;
                            if (m_trueDeformations!=NULL){
                                hatd1=(*m_trueDeformations)[(*m_imageIDList)[source]][(*m_imageIDList)[intermediate]];
                                hatd2=(*m_trueDeformations)[(*m_imageIDList)[intermediate]][(*m_imageIDList)[target]];
                                hatd3=(*m_trueDeformations)[(*m_imageIDList)[target]][(*m_imageIDList)[source]];

                            }
                            
                            //compute circle
                            DeformationFieldPointerType circle=composeDeformations(d1,d2,d3);
                            //compute norm
                            DeformationFieldIterator it(circle,circle->GetLargestPossibleRegion());
                            it.GoToBegin();
                            
                            // LOG<<VAR(dir)<<" "<<VAR(start)<<endl;
                            for (;!it.IsAtEnd();++it){
                                bool valid=true;
                                IndexType idx3=it.GetIndex(),idx2,idx1;
                                PointType pt1,pt2,pt3;
#if 0                                
                                //This is the backward assumption. circle errors are in the domain of d3, and are summed backwards
                                
                                d3->TransformIndexToPhysicalPoint(idx3,pt3);
                                pt2=pt3+d3->GetPixel(idx3);
                                //pt2=pt3+hatd3->GetPixel(idx3);
                                d2->TransformPhysicalPointToIndex(pt2,idx2);
                                // what to do when circle goes outside along the way?
                                // skip it
                                if ( !d2->GetLargestPossibleRegion().IsInside(idx2) ) {
                                    LOGV(6)<<"break at "<<VAR(eq)<<" "<<VAR(c)<<" "<<VAR(idx3)<<" "<<VAR(idx2)<<endl;
                                    eq=eq+D;
                                    c+=3*D;
                                    continue;
                                }
                                pt1=pt2+d2->GetPixel(idx2);
                                //pt1=pt2+hatd2->GetPixel(idx2);
                                d1->TransformPhysicalPointToIndex(pt1,idx1);
                                if ( (!d1->GetLargestPossibleRegion().IsInside(idx1) )) {
                                    LOGV(6)<<"break at "<<VAR(eq)<<" "<<VAR(c)<<" "<<VAR(idx3)<<" "<<VAR(idx2)<<endl;
                                    eq=eq+D;
                                    c+=3*D;
                                    continue;
                                }
#else
                                //now let's try the inverse of that
                                idx1=idx3;
                                idx2=idx3;
                              
#endif

                                double val=1;


                                //LOG<<VAR(idx3)<<" "<<VAR(idx2)<<" "<<VAR(idx1)<<endl;
                                LOGV(5)<<"1"<<endl;
                                long int e1=edgeNum(source,intermediate,idx1);
                                LOGV(5)<<"2"<<endl;
                                long int e2=edgeNum(intermediate,target,idx2);
                                LOGV(5)<<"3"<<endl;
                                long int e3=edgeNum(target,source,idx3);
                                LOGV(5)<<"4"<<endl;
                                if (e1<=0) {LOG<<VAR(e1)<<" ????? "<<endl;}
                                if (e2<=0) {LOG<<VAR(e2)<<endl;}
                                if (e3<=0) {LOG<<VAR(e3)<<endl; }
                                //LOG<<VAR(e1)<<" "<<VAR(e2)<<" "<<VAR(e3)<<endl;
                                
                                DeformationType localDef=it.Get();

                                PointType pt0;
                                pt0=pt1+d1->GetPixel(idx1);

                                LOGV(4)<<"consistency check : "<<VAR(localDef)<<" ?= "<<VAR(pt0-pt3)<<endl;
                                
                                maxE=max(maxE,max(e1,max(e2,e3)));
                                
                                for (unsigned int d=0;d<D;++d){
                                    double def=localDef[d];
                                    LOGV(6)<<VAR(e1)<<" "<<VAR(e2)<<" "<<VAR(e3)<<endl;

                                    //set sparse entries
                                    x[c]=eq;
                                    y[c]=e1+d;
                                    //LOG<<VAR(source)<<" "<<VAR(intermediate)<<" "<<VAR(p)<<" "<<VAR(edgeNum(source,intermediate,p))<<endl;
                                    v[c++]=val;
                                    x[c]=eq;
                                    y[c]=e2+d;
                                    v[c++]=val;
                                    x[c]=eq;
                                    y[c]=e3+d;
                                    v[c++]=val;
                                    
                                    //set rhs
                                    b[eq-1]=def;
                                    ++eq;
                                    LOGV(6)<<"did it"<<endl;
                                }
                                
                            }
                            //   LOG<<"done all pixels"<<endl;

                            
                            
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
        std::vector<double> result(m_nVars);
        double * rData=mxGetPr(this->m_result);
        for (int s = 0;s<m_numImages;++s){
            for (int t=0;t<m_numImages;++t){
                if (s!=t){
                    //slightly(!!!) stupid creation of empty image
                    DeformationFieldPointerType estimatedError=ImageUtils<DeformationFieldType>::createEmpty((*m_deformationCache)[(*m_imageIDList)[s]][(*m_imageIDList)[t]]);
                    DeformationFieldIterator it(estimatedError,estimatedError->GetLargestPossibleRegion());
                    it.GoToBegin();
                    for (int p=0;!it.IsAtEnd();++it){
                        DeformationType disp;
                        int e=edgeNum(s,t,it.GetIndex());
                        for (unsigned int d=0;d<D;++d,++p){
                            disp[d]=rData[e+d];
                        }
                        it.Set(disp);
                    }

                    ostringstream outfile;
                    outfile<<directory<<"/estimatedLocalComposedDeformationError-FROM-"<<(*m_imageIDList)[s]<<"-TO-"<<(*m_imageIDList)[t]<<".mha";
                    ImageUtils<DeformationFieldType>::writeImage(outfile.str().c_str(),estimatedError);
                }
            }
        }

    }

    std::vector<double> getResult(){
        std::vector<double> result(m_nVars);
        double * rData=mxGetPr(this->m_result);
        for (int i=0;i<m_nVars;++i){
            result[i]=rData[i];
            int n1,n2;
            edges(i+1,n1,n2);
            LOG<<VAR(i)<<" "<<VAR(n1)<<" "<<VAR(n2)<<" "<<VAR(result[i])<<endl;
        }
        return result;
        

    }
protected:
    int m_nVars,m_nEqs,m_nNonZeroes,m_nPixels;
    int m_numImages;
    map< string, map <string, DeformationFieldPointerType> > * m_deformationCache,* m_trueDeformations;
    std::vector<string> * m_imageIDList;
    bool m_additive;

protected:
    inline long int edgeNum(int n1,int n2,int p){ return ((n1)*(m_numImages-1) + n2 - (n2>n1))*(m_nPixels) +p +1;}
    inline long int edgeNum(int n1,int n2,IndexType idx){ 
        long int p = (*m_deformationCache)[(*m_imageIDList)[n1]][(*m_imageIDList)[n2]]->ComputeOffset(idx)*2;
        //LOG<<VAR(idx)<<" "<<VAR(p)<<endl;
        return p+1+((n1)*(m_numImages-1) + n2 - (n2>n1))*(m_nPixels) ;
    }
  

    inline void edges(int edgeNum, int &n1, int &n2){
        n1 = edgeNum/(m_numImages-1);
        n2 =edgeNum%(m_numImages-1);
        if (n2 ==0){
            n2=(m_numImages-1);
            n1--;
        }
        if (n2>n1) ++n2;
        n2--;
        
    }

    //compose 3 deformations. order is left-to-right
    DeformationFieldPointerType composeDeformations(DeformationFieldPointerType d1,DeformationFieldPointerType d2,DeformationFieldPointerType d3){
        return TransfUtils<ImageType>::composeDeformations(d3,TransfUtils<ImageType>::composeDeformations(d2,d1));

    }
};
