#pragma once
#include "matrix.h"
#include "BaseLinearSolver.h"
#include "TransformationUtils.h"
#include "Log.h"
#include "ImageUtils.h"
#include "Metrics.h"

#include <vector>
#include <sstream>
#include "SolveAquircGlobalDeformationNormCVariables.h"

template<class ImageType>
class AquircLocalDeformationAndErrorSolver: public AquircGlobalDeformationNormSolverCVariables< ImageType>{
public:
    typedef typename  TransfUtils<ImageType>::DeformationFieldType DeformationFieldType;
    typedef typename  DeformationFieldType::Pointer DeformationFieldPointerType;
    typedef typename  ImageType::OffsetType OffsetType;
    typedef typename DeformationFieldType::SpacingType SpacingType;

    typedef typename ImageUtils<ImageType>::FloatImagePointerType FloatImagePointerType;
    typedef typename ImageUtils<ImageType>::FloatImageType FloatImageType;
    typedef typename itk::ImageRegionIterator<FloatImageType> FloatImageIterator;
    typedef typename itk::ImageRegionIteratorWithIndex<DeformationFieldType> DeformationFieldIterator;
    typedef typename DeformationFieldType::PixelType DeformationType;
    typedef typename DeformationFieldType::IndexType IndexType;
    typedef typename DeformationFieldType::PointType PointType;
    typedef typename ImageType::Pointer ImagePointerType;
    typedef typename ImageType::ConstPointer ConstImagePointerType;
    typedef typename ImageType::RegionType RegionType;
    static const unsigned int D=ImageType::ImageDimension;
protected:
    int m_nVars,m_nEqs,m_nNonZeroes;
    int m_numImages;
    map< string, map <string, DeformationFieldPointerType> > * m_deformationCache,* m_trueDeformations, *m_updatedDeformationCache;
    std::vector<string> * m_imageIDList;
    bool m_additive;
    RegionType m_regionOfInterest;
    
    map<string,ImagePointerType> * m_imageList;

private:
    int m_nPixels;// number of pixels/voxels

    int m_nEqWd; // number of equations for  observation energy E_d
    int m_nVarWd; // number of variables for each equation of E_d;
    double m_wWd;

    int m_nEqWdelta; // number of equations for error minimizing energy E_delta
    int m_nVarWdelta; // number of variables for each equation of E_delta;
    double m_wWdelta;

    int m_nEqWcirc; // number of equations for energy circular constraint E_circ
    int m_nVarWcirc; // number of variables for each equation of E_circ;
    double m_wWcirc;

    int m_nEqWs; // number of equations for spatial smoothing energy E_s
    int m_nVarWs; // number of variables for each equation of E_d;
    double m_wWs;

    int m_nEqWT; // number of equations for Transformation similarity energy E_T
    int m_nVarWT; // number of variables for each equation of E_T
    double m_wWT;

    double m_wSum;

    double m_sigma;
    
    bool m_linearInterpol;
    bool m_haveDeformationEstimate;
public:
    AquircLocalDeformationAndErrorSolver(){
        m_wWT=1.0;
        m_wWs=1.0;
        m_wWcirc=1.0;
        m_wWdelta=1.0;
        m_wWd=1.0;
        m_wSum=1.0;
        m_sigma = 10.0;
        m_linearInterpol=false;
        m_haveDeformationEstimate=false;
        m_updatedDeformationCache = new  map< string, map <string, DeformationFieldPointerType> > ; 
    }
    virtual void SetVariables(std::vector<string> * imageIDList, map< string, map <string, DeformationFieldPointerType> > * deformationCache, map< string, map <string, DeformationFieldPointerType> > * trueDeformations,ImagePointerType ROI, map<string,ImagePointerType> * imagelist){
        m_imageList=imagelist;
        m_imageIDList=imageIDList;
        m_deformationCache=deformationCache;
        m_numImages=imageIDList->size();
        this->m_ROI=ROI;
        if (!ROI.IsNotNull()){
            this->m_ROI=FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::createEmpty(TransfUtils<ImageType>::computeLocalDeformationNorm((*m_deformationCache)[(*m_imageIDList)[0]][(*m_imageIDList)[1]],1.0)));
        }
        m_nPixels=this->m_ROI->GetLargestPossibleRegion().GetNumberOfPixels( );

        int interpolationFactor;
        if (m_linearInterpol)
            interpolationFactor = pow(2,D); //linear interpolation
        else
            interpolationFactor = 1 ; //NNinterpolation;
        

        m_nEqWd  = (m_wWd>0.0)* D * m_nPixels *  m_numImages*(m_numImages-1)*(m_numImages-2); //there is one equation for each component of every pixel of every triple of images
        m_nVarWd = 2*(interpolationFactor+2); //two variables per uniqe pair in the triple (2*2), plus linear interpolation for the third pair (2*2^D)

        m_nEqWcirc =  (m_wWcirc>0.0)* m_nPixels * D * m_numImages*(m_numImages-1)*(m_numImages-2); //again all components of all triples
        m_nVarWcirc = interpolationFactor+2; // only one/2^D variables per pair

        m_nEqWs =  (m_wWs>0.0)* D* m_nPixels * D * m_numImages*(m_numImages-1); //every pixel in each registration has D neighbors (in one direction), and each component separately
        m_nVarWs = 3; //for piecewise linear regularization, 2 for piecewise constant

      
        m_nEqWdelta = (m_wWdelta>0.0)*  m_nPixels * D * m_numImages*(m_numImages-1); //every error at every location in each image pair
        m_nVarWdelta = 1;

        m_nEqWT =  (m_wWT>0.0)*m_nPixels * D * m_numImages*(m_numImages-1); //same as Wdelta
        m_nVarWT= 1;

        int m_nEqSUM=(m_wSum>0.0)*m_nPixels * D * m_numImages*(m_numImages-1);
        int m_nVarSUM=2;

        m_nEqs=  m_nEqWd + m_nEqWcirc+ m_nEqWs + m_nEqWdelta+ m_nEqWT + m_nEqSUM ; // total number of equations
        
        bool estError= m_nEqWd ||  m_nEqWdelta || m_nEqSUM;
        bool estDef = m_nEqWd || m_nEqWT ||  m_nEqWs  ||  m_nEqWcirc || m_nEqSUM;
        m_nVars= m_numImages*(m_numImages-1)*m_nPixels*D *(estError + estDef); // total number of free variables (error and deformation)
        
        m_nNonZeroes= m_nEqWd *m_nVarWd + m_nEqWcirc * m_nVarWcirc + m_nEqWs*m_nVarWs + m_nEqWdelta*m_nVarWdelta + m_nEqWT*m_nVarWT + m_nEqSUM*m_nVarSUM; //maximum number of non-zeros

        m_trueDeformations=trueDeformations;


        m_regionOfInterest.SetSize(this->m_ROI->GetLargestPossibleRegion().GetSize());
        IndexType startIndex,nullIdx;
        nullIdx.Fill(0);
        PointType startPoint;
        this->m_ROI->TransformIndexToPhysicalPoint(nullIdx,startPoint);
        (*m_deformationCache)[(*m_imageIDList)[0]][(*m_imageIDList)[1]]->TransformPhysicalPointToIndex(startPoint,startIndex);
        m_regionOfInterest.SetIndex(startIndex);

        

    }

    void setWeightWd(double w){m_wWd=w;}
    void setWeightWT(double w){m_wWT=w;}
    void setWeightWs(double w){m_wWs=w;}
    void setWeightWcirc(double w){m_wWcirc=w;}
    void setWeightWdelta(double w){m_wWdelta=w;}
    void setWeightSum(double w){m_wSum=w;}
    void setLinearInterpol(bool i){m_linearInterpol=i;}
    void setSigma(double s){m_sigma=s;}

    virtual void createSystem(){

        LOG<<"Creating equation system.."<<endl;
        LOG<<VAR(m_numImages)<<" "<<VAR(m_nPixels)<<" "<<VAR(m_nEqs)<<" "<<VAR(m_nVars)<<" "<<VAR(m_nNonZeroes)<<endl;
        LOG<<VAR(  m_wWT)<<VAR(        m_wWs)<<" "<<VAR(m_wWcirc)<<" "<<VAR(m_wWdelta)<<" "<<VAR(m_wWd)<<" "<<VAR(m_wSum)<<endl;
        mxArray *mxX=mxCreateDoubleMatrix(m_nNonZeroes,1,mxREAL);
        mxArray *mxY=mxCreateDoubleMatrix(m_nNonZeroes,1,mxREAL);
        mxArray *mxV=mxCreateDoubleMatrix(m_nNonZeroes,1,mxREAL);
        mxArray *mxB=mxCreateDoubleMatrix(m_nEqs,1,mxREAL);
        mxArray *mxInit=mxCreateDoubleMatrix(m_nVars,1,mxREAL);
        if ( !mxX || !mxY || !mxV || !mxB || !mxInit){
            LOG<<"couldn't allocate memory!"<<endl;
            exit(0);
        }
        double * x=( double *)mxGetData(mxX);
        std::fill(x,x+m_nNonZeroes,m_nEqs);
        double * y=( double *)mxGetData(mxY);
        std::fill(y,y+m_nNonZeroes,m_nVars);
        double * v=( double *)mxGetData(mxV);
        double * b=mxGetPr(mxB);
        
        double * init=mxGetPr(mxInit);

        LOG<<"creating"<<endl;
     


        char buffer[256+1];
        buffer[256] = '\0';
        engOutputBuffer(this->m_ep, buffer, 256);
      
        //attention matlab index convention?!?
        long int eq = 1;
        long int c=0;

       

        for (int s = 0;s<m_numImages;++s){                            
            int source=s;
            for (int i=0;i<m_numImages;++i){
                if (i!=s){
                    int intermediate=i;
                    DeformationFieldPointerType d1=(*m_deformationCache)[(*m_imageIDList)[source]][(*m_imageIDList)[intermediate]];

                    for (int t=0;t<m_numImages;++t){
                        if (t!=i && t!=s){
                            //define a set of 3 images
                            int target=t;
                            DeformationFieldPointerType d2=(*m_deformationCache)[(*m_imageIDList)[intermediate]][(*m_imageIDList)[target]];
                            DeformationFieldPointerType d3=(*m_deformationCache)[(*m_imageIDList)[source]][(*m_imageIDList)[target]];
                        
                            
                            //compute indirect deform
                            DeformationFieldPointerType indirectDeform = TransfUtils<ImageType>::composeDeformations(d2,d1);
                            //compute difference of direct and indirect deform
                            DeformationFieldPointerType difference = TransfUtils<ImageType>::subtract(indirectDeform,d3);
                            
                            //compute norm
                            DeformationFieldIterator it(difference,m_regionOfInterest);//this->m_ROI->GetLargestPossibleRegion());
                            it.GoToBegin();
                            
                            DeformationFieldIterator trueIt;
                            if ((*m_trueDeformations)[(*m_imageIDList)[intermediate]][(*m_imageIDList)[target]].IsNotNull()){
                                trueIt=DeformationFieldIterator((*m_trueDeformations)[(*m_imageIDList)[intermediate]][(*m_imageIDList)[target]],(*m_trueDeformations)[(*m_imageIDList)[intermediate]][(*m_imageIDList)[target]]->GetLargestPossibleRegion());
                                trueIt.GoToBegin();
                            }

                            // LOG<<VAR(dir)<<" "<<VAR(start)<<endl;
                            for (;!it.IsAtEnd();++it){

                                bool valid=true;

                                //get index in target domain
                                IndexType targetIndex=it.GetIndex(),intermediateIndex,idx1;
                                LOGV(9)<<VAR(targetIndex)<<endl;
                                PointType ptIntermediate,ptTarget;
                                IndexType roiIntermediateIndex,roiTargetIndex;
                                
                                //get physical point in target domain
                                d3->TransformIndexToPhysicalPoint(targetIndex,ptTarget);
                                //get corresponding point in intermediate deform
                                DeformationType dIntermediate=d2->GetPixel(targetIndex);
                                ptIntermediate= ptTarget + dIntermediate;
                                
                                //get neighbors of that point
                                std::vector<std::pair<IndexType,double> >ptIntermediateNeighbors,ptIntermediateNeighborsCircle;
                                bool inside;
                                if (m_linearInterpol){
                                    inside=getLinearNeighbors(d1,ptIntermediate,ptIntermediateNeighbors);
                                }else{
                                    inside=getNearestNeighbors(d1,ptIntermediate,ptIntermediateNeighbors);
                                }
                                
                                //this can be used to index the circle constraint equation with the true deform if known. cheating!
                                //or with an estimation from the previous iteration
                                if (true && m_haveDeformationEstimate && (*m_updatedDeformationCache)[(*m_imageIDList)[intermediate]][(*m_imageIDList)[target]].IsNotNull()){
                                    DeformationType trueDef=(*m_updatedDeformationCache)[(*m_imageIDList)[intermediate]][(*m_imageIDList)[target]]->GetPixel(targetIndex);
                                    PointType truePtIntermediate=ptTarget + trueDef;
                                    inside= inside && getLinearNeighbors(d1,truePtIntermediate,ptIntermediateNeighborsCircle);
                                    ++trueIt;
                                }else{
                                    ptIntermediateNeighborsCircle=ptIntermediateNeighbors;
                                }

                                this->m_ROI->TransformPhysicalPointToIndex(ptTarget,roiTargetIndex);
                                LOGV(9)<<VAR(targetIndex)<<" "<<VAR(roiTargetIndex)<<endl;
                                
                                if (inside){
                                    double val=1;
                                    DeformationType localDiscrepance=it.Get();
                                
                                    for (unsigned int d=0;d<D;++d){
                                        double disp=localDiscrepance[d];
                                        LOGV(9)<<VAR(source)<<" "<<VAR(intermediate)<<" "<<VAR(target)<<" "<<VAR(roiTargetIndex)<<" "<<VAR(d)<<endl;
                                        LOGV(9)<<VAR(edgeNumError(intermediate,target,roiTargetIndex,d))<<" "<<VAR(edgeNumDeformation(intermediate,target,roiTargetIndex,d))<<endl;
                                        //set w_d ~ 
                                        if (m_wWd>0){
                                            //def and error intermediate->target
                                            x[c]=eq;
                                            y[c]=edgeNumError(intermediate,target,roiTargetIndex,d);
                                            v[c++]=val*m_wWd;
                                            x[c]=eq;
                                            y[c]=edgeNumDeformation(intermediate,target,roiTargetIndex,d);
                                            v[c++]=val*m_wWd;
                                            //interpolated def and error source->intermediate
                                            for (int i=0;i<ptIntermediateNeighbors.size();++i){
                                                x[c]=eq;
                                                y[c]=edgeNumError(source,intermediate,ptIntermediateNeighbors[i].first,d);
                                                v[c++]=ptIntermediateNeighbors[i].second*val*m_wWd;
                                                x[c]=eq;
                                                y[c]=edgeNumDeformation(source,intermediate,ptIntermediateNeighbors[i].first,d);
                                                v[c++]=ptIntermediateNeighbors[i].second*val*m_wWd;
                                            }
                                            //minus def and error source->target
                                            x[c]=eq;
                                            y[c]=edgeNumError(source,target,roiTargetIndex,d);
                                            v[c++]= - val*m_wWd;
                                            x[c]=eq;
                                            y[c]=edgeNumDeformation(source,target,roiTargetIndex,d);
                                            v[c++]= - val*m_wWd;
                                            b[eq-1]= disp*m_wWd;
                                            ++eq;
                                        }
                                    
                                        //set w_circ
                                        if (m_wWcirc>0.0){
                                            //val *=exp(- disp* disp / 25 ) ;
                                            //LOGV(4)<<VAR(disp)<<" "<<VAR(val)<<endl;
                                            x[c]=eq;
                                            y[c]=edgeNumDeformation(intermediate,target,roiTargetIndex,d);
                                            v[c++]=val* m_wWcirc;
                                            for (int i=0;i<ptIntermediateNeighborsCircle.size();++i){
                                                x[c]=eq;
                                                y[c]=edgeNumDeformation(source,intermediate,ptIntermediateNeighborsCircle[i].first,d); // this is a APPROXIMIATION!!! might be bad :o
                                                v[c++]=ptIntermediateNeighborsCircle[i].second*val* m_wWcirc;
                                            }
                                            x[c]=eq;
                                            y[c]=edgeNumDeformation(source,target,roiTargetIndex,d);
                                            v[c++]= - val* m_wWcirc;
                                            b[eq-1]=0;
                                            ++eq;
                                        }
                                    }
                                    
                                   
                                    

                                }// D
                            }//image

                        }//if
                    }//target
                    
                    //pairwise energies!


                    string sourceID=(*this->m_imageIDList)[source];
                    string intermediateID = (*this->m_imageIDList)[intermediate];
                    DeformationFieldPointerType defSourceInterm=(*this->m_deformationCache)[sourceID][intermediateID];
                    DeformationFieldIterator it(defSourceInterm,defSourceInterm->GetLargestPossibleRegion());
                    it.GoToBegin();

                    FloatImagePointerType lncc;
                    FloatImageIterator lnccIt;
                    if (m_sigma>0.0){
                        //upsample deformation -.-, and warp source image
                        DeformationFieldPointerType def = TransfUtils<ImageType>::bSplineInterpolateDeformationField(defSourceInterm,(ConstImagePointerType)(*m_imageList)[sourceID]);
                        ImagePointerType warpedImage= TransfUtils<ImageType>::warpImage((ConstImagePointerType)(*m_imageList)[sourceID],def);
                        //compute lncc
                        lncc= Metrics<ImageType,FloatImageType>::LNCC(warpedImage,(*m_imageList)[intermediateID],10.0);
                        ostringstream oss;
                        oss<<"lncc-"<<sourceID<<"-TO-"<<intermediateID;
                        if (D==2)
                            oss<<".png";
                        else
                            oss<<".nii";
                        LOGI(6,ImageUtils<ImageType>::writeImage(oss.str(),FilterUtils<FloatImageType,ImageType>::cast(ImageUtils<FloatImageType>::multiplyImageOutOfPlace(lncc,255))));
                        //resample lncc result
                        lncc = FilterUtils<FloatImageType>::LinearResample(lncc, FilterUtils<ImageType,FloatImageType>::cast(this->m_ROI),true);
                        lnccIt=FloatImageIterator(lncc,lncc->GetLargestPossibleRegion());
                        lnccIt.GoToBegin();
                    }
                    for (;!it.IsAtEnd();++it){
                        DeformationType localDef=it.Get();
                        IndexType idx=it.GetIndex();
                        LOGV(8)<<VAR(eq)<<" "<<VAR(localDef)<<endl;
                        
                        double weight=1.0;
                        if (lncc.IsNotNull()){
                            weight = lnccIt.Get();
                            LOGV(6)<<VAR(weight)<<endl;
                            ++lnccIt;
                            //weight*=weight;
                        }
                        
                        for (int n=0;n<D;++n){

                            //set w_delta
                            //set eqn for soft constraining the error to be small
                            if (m_wWdelta>0.0){
                                x[c]=eq;
                                y[c]= edgeNumError(source,intermediate,idx,n);
                                LOGV(8)<<VAR(source)<<" "<<VAR(intermediate)<<" "<<VAR(idx)<<" "<<VAR(n)<<" "<<VAR(edgeNumError(source,intermediate,idx,n))<<endl;
                                v[c++]=1.0*m_wWdelta*weight;
                                b[eq-1]=0.0;
                                ++eq;
                            }

                            //set w_T
                            //set eqn for soft constraining the estimated true deformation to be similar to the original deformation
                            if (m_wWT>0.0){
                                x[c]=eq;
                                y[c]=edgeNumDeformation(source,intermediate,idx,n);
                                LOGV(8)<<VAR(source)<<" "<<VAR(intermediate)<<" "<<VAR(idx)<<" "<<VAR(n)<<" "<<VAR(edgeNumDeformation(source,intermediate,idx,n))<<endl;
                                v[c++]=1.0*m_wWT * weight;
                                b[eq-1]=localDef[n]*m_wWT * weight;
                                ++eq;
                            }
                            
                            
                            //constraint that estimated def + estimated error = original def
                            if (m_wSum>0.0){
                                x[c]=eq;
                                y[c]=edgeNumError(source,intermediate,idx,n);
                                v[c++]=m_wSum;
                                x[c]=eq;
                                y[c]=edgeNumDeformation(source,intermediate,idx,n);
                                v[c++]=m_wSum;
                                b[eq-1]=m_wSum*localDef[n];
                                ++eq;
                            }
                          
                            //spatial smootheness of estimated deformations
                            if (m_wWs>0.0){
                                OffsetType off,off2;
                                off.Fill(0);
                                off2=off;
                                off[n]=1;
                                off2[n]=-1;
                                IndexType neighborIndexRight=idx+off;
                                IndexType neighborIndexLeft=idx+off2;
                                if (defSourceInterm->GetLargestPossibleRegion().IsInside(neighborIndexRight) &&defSourceInterm->GetLargestPossibleRegion().IsInside(neighborIndexLeft) ){
                                    DeformationType neighborDefRight=defSourceInterm->GetPixel(neighborIndexRight);
                                    DeformationType neighborDefLeft=defSourceInterm->GetPixel(neighborIndexLeft);
                                    for (unsigned int d=0;d<D;++d){
                                        LOGV(8)<<"regularizing... "<<VAR(source)<<" "<<VAR(intermediate)<<" "<<VAR(eq)<<" "<<VAR(c+3)<<" "<<endl;
                                        double def=localDef[d];
                                        double defNeighborRight=neighborDefRight[d];
                                        double defNeighborLeft=neighborDefLeft[d];
                                        x[c]=eq;
                                        y[c]=edgeNumDeformation(source,intermediate,idx,d);
                                        v[c++]=-2*this->m_wWs;
                                        x[c]=eq;
                                        y[c]=edgeNumDeformation(source,intermediate,neighborIndexRight,d);
                                        v[c++]=this->m_wWs;
                                        x[c]=eq;
                                        y[c]=edgeNumDeformation(source,intermediate,neighborIndexLeft,d);
                                        v[c++]=this->m_wWs;
                                        b[eq-1]=0.0;
                                        ++eq;
                                    }
                                }//inside

                                 //set initialisation values
                                init[edgeNumError(source,intermediate,idx,n)] = 0.0;
                                init[edgeNumDeformation(source,intermediate,idx,n)] = localDef[n] ;
                                
                            }
                            
                        }//neighbors
                    }//for

                }//if
            }//intermediate
        }//source
        LOG<<VAR(eq)<<" "<<VAR(c)<<endl;
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

        engPutVariable(this->m_ep,"init",mxInit);
        mxDestroyArray(mxInit);
        this->haveInit=true;
        engEvalString(this->m_ep,"save('test.mat','A','b');" );

    }

    virtual void storeResult(string directory){
        std::vector<double> result(m_nVars);
        double * rData=mxGetPr(this->m_result);
        double trueResidual=0.0;
        double estimationResidual=0.0;
        double circleResidual=0.0;
        double averageError=0.0;
        int c=0;
        int c2=0;
        for (int s = 0;s<m_numImages;++s){
            for (int t=0;t<m_numImages;++t){
                if (s!=t){
                    //slightly(!!!) stupid creation of empty image
                    DeformationFieldPointerType estimatedError=TransfUtils<ImageType>::createEmpty(this->m_ROI);//ImageUtils<DeformationFieldType>::createEmpty((*m_deformationCache)[(*m_imageIDList)[s]][(*m_imageIDList)[t]]);
                    DeformationFieldIterator itErr(estimatedError,estimatedError->GetLargestPossibleRegion());
                    DeformationFieldPointerType estimatedDeform=TransfUtils<ImageType>::createEmpty(this->m_ROI);//ImageUtils<DeformationFieldType>::createEmpty((*m_deformationCache)[(*m_imageIDList)[s]][(*m_imageIDList)[t]]);
                    DeformationFieldIterator itDef(estimatedDeform,estimatedDeform->GetLargestPossibleRegion());
                    
                    itErr.GoToBegin();
                    itDef.GoToBegin();
                  
                    for (int p=0;!itErr.IsAtEnd();++itErr,++itDef){
                        //get solution of eqn system
                        DeformationType dispErr,dispDef;
                        IndexType idx = itErr.GetIndex();
                        for (unsigned int d=0;d<D;++d,++p){
                            // minus 1 to correct for matlab indexing
                            dispErr[d]=rData[edgeNumError(s,t,idx,d)-1];
                            dispDef[d]=rData[edgeNumDeformation(s,t,idx,d)-1];
                        }
                        itErr.Set(dispErr);
                        itDef.Set(dispDef);
                        LOGV(8)<<VAR(c)<<" "<<VAR(dispDef)<<endl;
                        ++c;
                    }
                    if ((*m_trueDeformations)[(*m_imageIDList)[s]][(*m_imageIDList)[t]].IsNotNull()){
                        double newError=TransfUtils<ImageType>::computeDeformationNorm(TransfUtils<ImageType>::subtract(estimatedDeform,(*m_trueDeformations)[(*m_imageIDList)[s]][(*m_imageIDList)[t]]),1);
                        double oldError=TransfUtils<ImageType>::computeDeformationNorm(TransfUtils<ImageType>::subtract((*m_deformationCache)[(*m_imageIDList)[s]][(*m_imageIDList)[t]],(*m_trueDeformations)[(*m_imageIDList)[s]][(*m_imageIDList)[t]]),1);
                        LOGV(1)<<VAR(oldError)<<" "<<VAR(newError)<<endl;
                        averageError+=newError;
                    }
                    c2++;
                    ostringstream outfile;
                    if (directory != ""){
                        outfile<<directory<<"/estimatedLocalComposedDeformationError-FROM-"<<(*m_imageIDList)[s]<<"-TO-"<<(*m_imageIDList)[t]<<".mha";
                        ImageUtils<DeformationFieldType>::writeImage(outfile.str().c_str(),estimatedError);
                        ostringstream outfile2;
                        outfile2<<directory<<"/estimatedLocalComposedDeformation-FROM-"<<(*m_imageIDList)[s]<<"-TO-"<<(*m_imageIDList)[t]<<".mha";
                        ImageUtils<DeformationFieldType>::writeImage(outfile2.str().c_str(),estimatedDeform);
                    }
                    //(*m_deformationCache)[(*m_imageIDList)[s]][(*m_imageIDList)[t]]= estimatedDeform;
                    (*m_updatedDeformationCache)[(*m_imageIDList)[s]][(*m_imageIDList)[t]] = estimatedDeform;
                    m_haveDeformationEstimate = true;
                }
            }
        }
        estimationResidual=(estimationResidual)/c;
        trueResidual=(trueResidual)/c;
        averageError/=c2;
        LOG<<VAR(averageError)<<" "<<VAR(estimationResidual)<<" "<<VAR(trueResidual)<<" "<<VAR(c)<<endl;
    }

    std::vector<double> getResult(){
        std::vector<double> result(m_nVars);
        return result;
        

    }


protected:
    //return fortlaufende number of pairs n1,n2, 0..(n*(n-1)-1)
    inline long int edgeNum(int n1,int n2){ return ((n1)*(m_numImages-1) + n2 - (n2>n1));}
 
#if 0   
    //alternating edgenumbering : n*def, n*err, n*def, n*err ....
    //return edgenumber after taking into acount nPixel*2 edges per image pair
    inline long int edgeNumDeformation(int n1,int n2,IndexType idx, int d){ 
        long int offset = this->m_ROI->ComputeOffset(idx);
        return offset*D+2*edgeNum(n1,n2)*m_nPixels*D +d+ 1 ;
        //return offset*D+edgeNum(n1,n2)*m_nPixels*D +d+ 1 ;
    }

    inline long int edgeNumError(int n1,int n2,IndexType idx, int d){ 
        long int offset = this->m_ROI->ComputeOffset(idx);
        return offset*D+2*(edgeNum(n1,n2))*m_nPixels*D + m_nPixels*D +d+ 1;
    }
#else
    inline long int edgeNumDeformation(int n1,int n2,IndexType idx, int d){ 
        long int offset = this->m_ROI->ComputeOffset(idx);
        //return offset*D+2*edgeNum(n1,n2)*m_nPixels*D +d+ 1 ;
        return offset*D+edgeNum(n1,n2)*m_nPixels*D +d+ 1 ;
    }

    inline long int edgeNumError(int n1,int n2,IndexType idx, int d){ 
        return  m_nPixels*D*(m_numImages-1)*(m_numImages) + edgeNumDeformation(n1,n2,idx,d);
    }

#endif
    //compose 3 deformations. order is left-to-right
    DeformationFieldPointerType composeDeformations(DeformationFieldPointerType d1,DeformationFieldPointerType d2,DeformationFieldPointerType d3){
        return TransfUtils<ImageType>::composeDeformations(d3,TransfUtils<ImageType>::composeDeformations(d2,d1));

    }


    inline bool getLinearNeighbors(const DeformationFieldPointerType def, const PointType & point, std::vector<std::pair<IndexType,double> > & neighbors){
        bool inside=false;
        neighbors= std::vector<std::pair<IndexType,double> >(pow(2,D));
        int nNeighbors=0;
        IndexType idx1;
        def->TransformPhysicalPointToIndex(point,idx1);
        inside=inside || def->GetLargestPossibleRegion().IsInside(idx1);
        if (!inside) return false;
        PointType pt1;
        def->TransformIndexToPhysicalPoint(idx1,pt1);
        DeformationType dist=point-pt1;
        if (inside){
            neighbors[nNeighbors++]=std::make_pair(idx1,getWeight(dist,def->GetSpacing()));
        }
        OffsetType off;
        off.Fill(0);
        for (int i=1;i<pow(2,D);++i){
            int spill=1;
            for (int d=0;d<D;++d){
                off[d]+=spill*sign(dist[d]);
                if (fabs(off[d])>1){
                    spill=1;off[d]=0;
                }else{
                    break;
                }

            }
            IndexType idx=idx1+off;
            PointType pt;
            def->TransformIndexToPhysicalPoint(idx,pt);
            DeformationType delta=point-pt;
            if (def->GetLargestPossibleRegion().IsInside(idx)){
                neighbors[nNeighbors++]=std::make_pair(idx,getWeight(delta,def->GetSpacing()));
                inside=true;
            }
        }
        neighbors.resize(nNeighbors);
        return inside;
    }
    inline bool getNearestNeighbors(const DeformationFieldPointerType def, const PointType & point, std::vector<std::pair<IndexType,double> > & neighbors){
        bool inside=false;
        neighbors= std::vector<std::pair<IndexType,double> >(1);
        int nNeighbors=0;
        IndexType idx1;
        def->TransformPhysicalPointToIndex(point,idx1);
        inside=inside || def->GetLargestPossibleRegion().IsInside(idx1);
        PointType pt1;
        def->TransformIndexToPhysicalPoint(idx1,pt1);
        DeformationType dist=point-pt1;
        if (inside){
            neighbors[nNeighbors++]=std::make_pair(idx1,1.0);
        }
       
        return inside;
    }

    inline double getWeight(const DeformationType & dist, const SpacingType & space){
        double w=1.0;
        for (int d=0;d<D;++d){
            w*=(1-fabs(dist[d])/space[d]);
        }
        LOGV(9)<<VAR(dist)<<" "<<VAR(space)<<" "<<VAR(w)<<endl;
        return w;
    }
    inline int sign(const double s){

        if (s>=0) return 1;
        if (s<0) return -1;
        return 0;
    }
};
