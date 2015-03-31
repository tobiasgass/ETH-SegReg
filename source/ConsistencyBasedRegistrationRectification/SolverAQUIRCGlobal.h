#pragma once
#include "matrix.h"
#include "SolverLinearBase.h"
#include "TransformationUtils.h"
#include "Log.h"
#include "ImageUtils.h"
#include <vector>
#include <sstream>

namespace CBRR{
template<class ImageType>
class SolverAQUIRCGlobal: public LinearSolver{
public:
    typedef typename  TransfUtils<ImageType>::DeformationFieldType DeformationFieldType;
    typedef typename  DeformationFieldType::Pointer DeformationFieldPointerType;
    typedef typename ImageType::Pointer ImagePointerType;
    typedef typename ImageUtils<ImageType>::FloatImageType FloatImageType;
public:

    virtual void SetVariables(std::vector<string> * imageIDList, map< string, map <string, DeformationFieldPointerType> > * deformationCache, map< string, map <string, DeformationFieldPointerType> > * trueDeformations, ImagePointerType ROI){
        m_imageIDList=imageIDList;
        m_deformationCache=deformationCache;
        m_numImages=imageIDList->size();
        m_nEqs= m_numImages*(m_numImages-1)*( m_numImages-2);
        m_nVars= m_numImages*(m_numImages-1);
        m_nNonZeroes=3*m_nEqs;
        m_ROI=ROI;
    }
    virtual void setCircleWeights(double w1, double w3){
        m_w1=w1;
        m_w3=w3;
    }
    virtual void SetTrueDeformations( map< string, map <string, DeformationFieldPointerType> > * deformationCache){
        m_trueDeformations=deformationCache;
    }
    virtual void createSystem(){

        mxArray *mxX=mxCreateDoubleMatrix(m_nNonZeroes,1,mxREAL);
        mxArray *mxY=mxCreateDoubleMatrix(m_nNonZeroes,1,mxREAL);
        mxArray *mxV=mxCreateDoubleMatrix(m_nNonZeroes,1,mxREAL);
        mxArray *mxB=mxCreateDoubleMatrix(m_nEqs,1,mxREAL);
        
        double * x=( double *)mxGetData(mxX);
        double * y=( double *)mxGetData(mxY);
        double * v=( double *)mxGetData(mxV);
        double * b=mxGetPr(mxB);
        
        

        //m_A = mxCreateSparse(m_nVars, m_nEqs, m_nNonZeroes);
        //engPutVariable(m_ep, "A", m_A);
        LOG<<"creating"<<endl;
        {
            ostringstream evalstr;
            evalstr<<"A = sparse([],[],[],"<<m_nEqs<<","<<m_nVars<<","<<m_nNonZeroes<<");";
            engEvalString(m_ep,evalstr.str().c_str() );
        }

        {
            ostringstream evalstr;
            evalstr<<" b=zeros(1,"<<m_nEqs<<");";
            engEvalString(m_ep,evalstr.str().c_str() );
        }


        char buffer[256+1];
        buffer[256] = '\0';
        engOutputBuffer(m_ep, buffer, 256);
      
        //attention matlab index convention?!?
        int eq = 1;
        //create edge index storage object
        int c=0;
        for (int s = 0;s<m_numImages;++s){
            for (int i=s+1;i<m_numImages;++i){
                for (int t=i+1;t<m_numImages;++t){
                    //define a set of 3 images
                    //there are 6 possible circles with 3 images
                    int source=s;
                    int intermediate=i;
                    int target=t;
                    //       LOG<<VAR(s)<<" "<<VAR(i)<<" "<<VAR(t)<<endl;
                    for (int dir=0;dir<2;++dir){ //forwar-backward
                        double normSum=0.0;
                        for (int start=0;start<3;++start){
                            if (intermediate<0 || target<0 || intermediate>m_numImages || target>m_numImages) break;

                            //compute circle
                            DeformationFieldPointerType circle=composeDeformations((*m_deformationCache)[(*m_imageIDList)[source]][(*m_imageIDList)[intermediate]],
                                                                                   (*m_deformationCache)[(*m_imageIDList)[intermediate]][(*m_imageIDList)[target]],
                                                                                   (*m_deformationCache)[(*m_imageIDList)[target]][(*m_imageIDList)[source]]);
                            //compute norm
                            double norm=TransfUtils<ImageType>::computeDeformationNorm(circle,1.0);

                            if (!m_additive){
                                norm=max(log(norm),-15.0);
                            }
                            //set sparse entries
                            x[c]=eq;
                            y[c]=edgeNum(source,intermediate);
                            v[c++]=1;
                            x[c]=eq;
                            y[c]=edgeNum(intermediate,target);
                            v[c++]=1;
                            x[c]=eq;
                            y[c]=edgeNum(target,source);
                            v[c++]=1;

                            //set rhs
                            b[eq-1]=3*norm;
                            LOG<<VAR(c)<<" "<<VAR(eq)<<endl;
                            ++eq;

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
        
        engPutVariable(m_ep,"xCord",mxX);
        engPutVariable(m_ep,"yCord",mxY);
        engPutVariable(m_ep,"val",mxV);
        engPutVariable(m_ep,"b",mxB);
        engEvalString(m_ep,"A=sparse(xCord,yCord,val);" );

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
    int m_nVars,m_nEqs,m_nNonZeroes;
    int m_numImages;
    map< string, map <string, DeformationFieldPointerType> > * m_deformationCache,* m_trueDeformations;
    std::vector<string> * m_imageIDList;
    bool m_additive;
    ImagePointerType m_ROI;
    double m_w1,m_w3;

protected:
    inline int edgeNum(int n1,int n2){ return (n1)*(m_numImages-1) + n2 - (n2>n1)+1;}
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

    inline std::vector<double> getCircleWeights(double localCircleDef){
        std::vector<double> weights(3,0.0);
        double wSum=0.0;
        LOGV(5)<<VAR(fabs(localCircleDef))<<" "<<VAR(exp(-0.5*localCircleDef*localCircleDef/50))<<endl;
        weights[0]=this->m_w1;//1.0+0.6;//*exp(-0.5*localCircleDef/50);
        weights[1]=1.0;
        weights[2]=this->m_w3;//1.0-0.3;//*exp(-0.5*localCircleDef/50);
        
        for (int i=0;i<3;++i){
            wSum+=1.0/3*weights[i];
        }
        for (int i=0;i<3;++i){
            weights[i]/=wSum;
        }
        return weights;

    }
};
}//namespace
