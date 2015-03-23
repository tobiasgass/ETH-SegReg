#pragma once
#include "matrix.h"
#include "BaseLinearSolver.h"
#include "TransformationUtils.h"
#include "Log.h"
#include "ImageUtils.h"
#include <vector>
#include <sstream>


template<class ImageType>
class AquircGlobalDeformationNormSolver: public LinearSolver{
public:
    typedef typename  TransfUtils<ImageType>::DeformationFieldType DeformationFieldType;
    typedef typename  DeformationFieldType::Pointer DeformationFieldPointerType;
public:

    virtual void SetVariables(std::vector<string> * imageIDList, map< string, map <string, DeformationFieldPointerType> > * deformationCache){
        m_imageIDList=imageIDList;
        m_deformationCache=deformationCache;
        m_numImages=imageIDList->size();
        m_nEqs= m_numImages*(m_numImages-1)*( m_numImages-2);
        m_nVars= m_numImages*(m_numImages-1);
        m_nNonZeroes=3* m_nEqs;
    }
    
    virtual void createSystem(){
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
        mwSize dummy[m_nEqs];
        memset(dummy, 1, m_nEqs);
       
        //engPutVariable(m_ep, "b", m_b);

        //attention matlab index convention?!?
        int eq = 1;
        //create edge index storage object
        
        for (int s = 0;s<m_numImages;++s){
            for (int i=s+1;i<m_numImages;++i){
                for (int t=i+1;t<m_numImages;++t){
                    //define a set of 3 images
                    //there are 6 possible circles with 3 images
                    int source=s;
                    int intermediate=i;
                    int target=t;
                    LOG<<VAR(s)<<" "<<VAR(i)<<" "<<VAR(t)<<endl;
                    for (int dir=0;dir<2;++dir){ //forwar-backward
                        double normSum=0.0;

                        for (int start=0;start<3;++start){
                            if (intermediate<0 || target<0 || intermediate>m_numImages || target>m_numImages) break;
                            LOG<<VAR(source)<<" "<<VAR(intermediate)<<" "<<VAR(target)<<" "<<VAR(eq)<<endl;
                            LOG<<edgeNum(source,intermediate)<<" "<<edgeNum(intermediate,target)<<" "<<edgeNum(target,source)<<endl;
                            int n1,n2;
                            edges(edgeNum(source,intermediate),n1,n2);
                            LOG<<"TEST "<<(n1==source)<<" "<<(n2==intermediate)<<" "<<source<<" "<<n1<<" "<<intermediate<<" "<<n2<<endl;
                            LOG<<VAR((*m_imageIDList)[source])<<" "<<VAR((*m_imageIDList)[intermediate])<<" "<<VAR((*m_imageIDList)[target])<<endl;
                                                

                         

                            //compute circle
#if 1
                            DeformationFieldPointerType circle=composeDeformations((*m_deformationCache)[(*m_imageIDList)[source]][(*m_imageIDList)[intermediate]],
                                                                                   (*m_deformationCache)[(*m_imageIDList)[intermediate]][(*m_imageIDList)[target]],
                                                                                   (*m_deformationCache)[(*m_imageIDList)[target]][(*m_imageIDList)[source]]);
                            //compute norm
                            double norm=TransfUtils<ImageType>::computeDeformationNorm(circle,1.0);
#else
                            double norm=1;
#endif
                            LOG<<VAR(norm)<<endl;
                            if (!m_additive){
                                norm=max(log(norm),-15.0);
                            }
                            normSum+=norm;
                           
                          

                            //shift start point
                            int tmpInt=source;
                            source=intermediate;
                            intermediate=target;
                            target=tmpInt;
                        }
                        {
                            //create edge array in matlab
                            ostringstream evalstr;
                            evalstr<<"edge=["<<edgeNum(source,intermediate)<<" "<<edgeNum(intermediate,target)<<" "<<edgeNum(target,source)<<"]";
                            engEvalString(m_ep,evalstr.str().c_str() );
                            printf("%s", buffer+2);

                        }
                    
                        {
                            //set edge indices in sparse matrix
                            ostringstream evalstr;
                            evalstr<<"eq="<<eq<<";";
                            engEvalString(m_ep,evalstr.str().c_str() );
                            engEvalString(m_ep, "A(eq, edge) = true");
                        }
                        {
                            //set target array
                            ostringstream evalstr;
                            evalstr<<"b(eq)="<<normSum<<";";
                            engEvalString(m_ep,evalstr.str().c_str() );
                        }
                        ++eq;
                        int tmp=target;
                        target=intermediate;
                        intermediate=tmp;

                    }

                
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
    int m_nVars,m_nEqs,m_nNonZeroes;
    int m_numImages;
    map< string, map <string, DeformationFieldPointerType> > * m_deformationCache;
    std::vector<string> * m_imageIDList;
    bool m_additive;

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
};
