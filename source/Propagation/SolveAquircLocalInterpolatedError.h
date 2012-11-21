#pragma once
#include "matrix.h"
#include "BaseLinearSolver.h"
#include "TransformationUtils.h"
#include "Log.h"
#include "ImageUtils.h"
#include <vector>
#include <sstream>
#include "SolveAquircLocalComposedError.h"
#include "itkVectorLinearInterpolateImageFunction.h"
#include "itkNeighborhoodIterator.h"

template<class ImageType>
class AquircLocalInterpolatedErrorSolver: public AquircLocalComposedErrorSolver< ImageType>{
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
    typedef typename DeformationFieldType::SpacingType SpacingType;
    typedef typename ImageType::Pointer ImagePointerType;
    typedef typename ImageType::OffsetType OffsetType;
    typedef typename itk::VectorLinearInterpolateImageFunction<DeformationFieldType> DeformationFieldInterpolaterType;
    typedef typename DeformationFieldInterpolaterType::Pointer DeformationFieldInterpolaterPointerType;
    typedef typename DeformationFieldInterpolaterType::ContinuousIndexType ContinuousIndexType;
    static const unsigned int D=ImageType::ImageDimension;
public:

    virtual void SetVariables(std::vector<string> * imageIDList, map< string, map <string, DeformationFieldPointerType> > * deformationCache, map< string, map <string, DeformationFieldPointerType> > * trueDeformations,ImagePointerType img){
        LOG<<"LOCAL INTERPOLATED COMPOSED ERROR SOLVER"<<endl;
        this->m_imageIDList=imageIDList;
        this->m_deformationCache=deformationCache;
        this->m_numImages=imageIDList->size();
        this->m_nPixels=2*(*deformationCache)[(*imageIDList)[0]][(*imageIDList)[1]]->GetLargestPossibleRegion().GetNumberOfPixels( );
        this->m_nEqs= this->m_numImages*(this->m_numImages-1)*( this->m_numImages-2)*this->m_nPixels;
        this->m_nVars= this->m_numImages*(this->m_numImages-1)*this->m_nPixels;
        //NN interpol
        this->m_nNonZeroes=3*this->m_nEqs;
        //Linear interpol
        this->m_nNonZeroes=(3+2*pow(2,D))*this->m_nEqs;
        //gaussian interpol
        int radius=2;
        //this->m_nNonZeroes=(3+2*pow(2*radius+1,D))*this->m_nEqs;
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
    
    virtual void createSystem(){

        LOG<<"Creating equation system.."<<endl;
        LOG<<VAR(this->m_numImages)<<" "<<VAR(this->m_nPixels)<<" "<<VAR(this->m_nEqs)<<" "<<VAR(this->m_nVars)<<" "<<VAR(this->m_nNonZeroes)<<endl;
        mxArray *mxX=mxCreateDoubleMatrix(this->m_nNonZeroes,1,mxREAL);
        mxArray *mxY=mxCreateDoubleMatrix(this->m_nNonZeroes,1,mxREAL);
        mxArray *mxV=mxCreateDoubleMatrix(this->m_nNonZeroes,1,mxREAL);
        mxArray *mxB=mxCreateDoubleMatrix(this->m_nEqs,1,mxREAL);
        if (! (mxX && mxY && mxV &&mxB)){
            LOG<<"not enough memory, aborting"<<endl;
            exit(0);
        }

        double * x=( double *)mxGetData(mxX);
        std::fill(x,x+this->m_nNonZeroes,this->m_nEqs);
        double * y=( double *)mxGetData(mxY);
        std::fill(y,y+this->m_nNonZeroes,this->m_nVars);
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
        for (int s = 0;s<this->m_numImages;++s){                            
            int source=s;
            for (int i=0;i<this->m_numImages;++i){
                if (i!=s){
                    int intermediate=i;
                    DeformationFieldPointerType d1=(*this->m_deformationCache)[(*this->m_imageIDList)[source]][(*this->m_imageIDList)[intermediate]];
                    DeformationFieldInterpolaterPointerType d1Interpolater=DeformationFieldInterpolaterType::New();
                    d1Interpolater->SetInputImage(d1);

                    for (int t=0;t<this->m_numImages;++t){

                        if (t!=i && t!=s){
                            //define a set of 3 images
                            int target=t;
                            DeformationFieldPointerType d2=(*this->m_deformationCache)[(*this->m_imageIDList)[intermediate]][(*this->m_imageIDList)[target]];
                            DeformationFieldPointerType d3=(*this->m_deformationCache)[(*this->m_imageIDList)[target]][(*this->m_imageIDList)[source]];
                            DeformationFieldInterpolaterPointerType d2Interpolater=DeformationFieldInterpolaterType::New();
                            d2Interpolater->SetInputImage(d2);

                            DeformationFieldPointerType hatd1,hatd2,hatd3;
                            if (this->m_trueDeformations!=NULL){
                                hatd1=(*this->m_trueDeformations)[(*this->m_imageIDList)[source]][(*this->m_imageIDList)[intermediate]];
                                hatd2=(*this->m_trueDeformations)[(*this->m_imageIDList)[intermediate]][(*this->m_imageIDList)[target]];
                                hatd3=(*this->m_trueDeformations)[(*this->m_imageIDList)[target]][(*this->m_imageIDList)[source]];
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

                                std::vector<std::pair<IndexType,double> > pt2Neighbors,pt1Neighbors;
#if 1                         
                                //This is the backward assumption. circle errors are in the domain of d3, and are summed backwards
                                
                                d3->TransformIndexToPhysicalPoint(idx3,pt3);
                                pt2=pt3+d3->GetPixel(idx3);
                                d2->TransformPhysicalPointToIndex(pt2,idx2);
                                // what to do when circle goes outside along the way?
                                // skip it
                                //bool inside=getGaussianNeighbors(d2,pt2,pt2Neighbors);
                                bool inside=getLinearNeighbors(d2,pt2,pt2Neighbors);
                                //bool inside=getNearestNeighbors(d2,pt2,pt2Neighbors);
                                if ( !inside ) {
                                    LOGV(6)<<"break at "<<VAR(eq)<<" "<<VAR(c)<<" "<<VAR(idx3)<<" "<<VAR(idx2)<<endl;
                                    continue;
                                }
                                ContinuousIndexType ci2;
                                d2->TransformPhysicalPointToContinuousIndex(pt2,ci2);
                                pt1=pt2+d2Interpolater->EvaluateAtContinuousIndex(ci2);
                                d1->TransformPhysicalPointToIndex(pt1,idx1);
                                //inside=getNearestNeighbors(d1,pt1,pt1Neighbors);
                                inside=getLinearNeighbors(d1,pt1,pt1Neighbors);
                                //inside=getGaussianNeighbors(d1,pt1,pt1Neighbors);
                                if ( (! inside )) {
                                    LOGV(6)<<"break at "<<VAR(eq)<<" "<<VAR(c)<<" "<<VAR(idx3)<<" "<<VAR(idx2)<<endl;
                                    continue;
                                }
#else
                                //fixed point estimation
                                idx1=idx3;
                                idx2=idx3;
                                
#endif
                                
                                double val=1;

                             
                                
                                DeformationType localDef=it.Get();
                                
                                PointType pt0;
                                ContinuousIndexType ci1;
                                d1->TransformPhysicalPointToContinuousIndex(pt1,ci1);
                                pt0=pt1+d1Interpolater->EvaluateAtContinuousIndex(ci1);
                                
                                LOGV(4)<<"consistency check : "<<VAR(localDef)<<" ?= "<<VAR(pt0-pt3)<<endl;
                                
                                
                                for (unsigned int d=0;d<D;++d){
                                    double def=localDef[d];
                                    val=1.0;//exp(-def*def/100);
                                    std::vector<double> weights=this->getCircleWeights(def);
                                    
                                   
                                    //set sparse entries
                                    //neighbors of pt1
                                    for (int i=0;i<pt1Neighbors.size();++i){
                                        LOGV(7)<<VAR(eq)<<" "<< VAR(pt1)<<" "<<VAR(i)<<" "<<VAR(c)<<" "<<VAR(pt1Neighbors[i].first)<<" "<<VAR(pt1Neighbors[i].second)<<endl;
                                        x[c]=eq;
                                        y[c]=edgeNum(source,intermediate,pt1Neighbors[i].first)+d+1;
                                        v[c++]=weights[0]*pt1Neighbors[i].second*val;
                                    }
                                    for (int i=0;i<pt2Neighbors.size();++i){
                                        LOGV(7)<<VAR(eq)<<" "<< VAR(pt2)<<" "<< VAR(i)<<" "<< VAR(c)<<" "<<VAR(pt2Neighbors[i].first)<<" "<<VAR(pt2Neighbors[i].second)<<endl;
                                        x[c]=eq;
                                        y[c]=edgeNum(intermediate,target,pt2Neighbors[i].first)+d+1;
                                        v[c++]=weights[1]*pt2Neighbors[i].second*val;
                                    }
                                    x[c]=eq;
                                    y[c]=edgeNum(target,source,idx3)+d+1;;
                                    v[c++]=weights[2]*val;
                                    
                                    //set rhs
                                    b[eq-1]=def*val;
                                    ++eq;
                                    LOGV(6)<<"did it"<<endl;
                                }// D
                            }//image

                        }//if
                    }//target
                     if (this->m_regWeight>0.0){
                        DeformationFieldPointerType defSourceInterm=(*this->m_deformationCache)[(*this->m_imageIDList)[source]][(*this->m_imageIDList)[intermediate]];
                        DeformationFieldIterator it(defSourceInterm,defSourceInterm->GetLargestPossibleRegion());
                        it.GoToBegin();
                        for (;!it.IsAtEnd();++it){
                            DeformationType localDef=it.Get();
                            IndexType idx=it.GetIndex();
                            long int e=edgeNum(source,intermediate,idx)+1;
                            for (int n=0;n<D;++n){
                                OffsetType off,off2;
                                off.Fill(0);
                                off2=off;
                                off[n]=1;
                                off2[n]=-1;
                                IndexType neighborIndexRight=idx+off;
                                IndexType neighborIndexLeft=idx+off2;
                                if (defSourceInterm->GetLargestPossibleRegion().IsInside(neighborIndexRight) &&defSourceInterm->GetLargestPossibleRegion().IsInside(neighborIndexLeft) ){
                                    long int eNeighborRight=edgeNum(source,intermediate,neighborIndexRight)+1;
                                    DeformationType neighborDefRight=defSourceInterm->GetPixel(neighborIndexRight);
                                    long int eNeighborLeft=edgeNum(source,intermediate,neighborIndexLeft)+1;
                                    DeformationType neighborDefLeft=defSourceInterm->GetPixel(neighborIndexLeft);
                                    LOGV(6)<<""<<VAR(idx)<<" "<<VAR(neighborIndexRight)<<" "<<VAR(e)<<" "<<VAR(eNeighborRight)<<" "<<endl;
                                    for (unsigned int d=0;d<D;++d){
                                        LOGV(7)<<"regularizing... "<<VAR(source)<<" "<<VAR(intermediate)<<" "<<VAR(eq)<<" "<<VAR(c+3)<<" "<<endl;
                                        double def=localDef[d];
                                        double defNeighborRight=neighborDefRight[d];
                                        double defNeighborLeft=neighborDefLeft[d];
                                        x[c]=eq;
                                        y[c]=e+d;
                                        v[c++]=-2*this->m_regWeight;
                                        x[c]=eq;
                                        y[c]=eNeighborRight+d;
                                        v[c++]=this->m_regWeight;
                                        x[c]=eq;
                                        y[c]=eNeighborLeft+d;
                                        v[c++]=this->m_regWeight;
                                        b[eq-1]=this->m_regWeight*(defNeighborRight+defNeighborLeft-2*def);
                                        ++eq;
                                    }
                                }//inside

                            }//neighbors
                        }//for
                    }//regularization
                }//if
            }//intermediate
        }//source
        LOG<<VAR(eq)<<" "<<VAR(c)<<endl;
        this->m_nNonZeroes=c;
        mxSetM(mxX,c);
        mxSetM(mxY,c);
        mxSetM(mxV,c);
        mxSetM(mxB,eq-1);
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


   

protected:
   
   
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

    inline bool getGaussianNeighbors(const DeformationFieldPointerType def, const PointType & point, std::vector<std::pair<IndexType,double> > & neighbors, int radius=2, double sigma=4.0){
        bool inside=false;
        neighbors= std::vector<std::pair<IndexType,double> >(pow(2*radius+1,D));
        int nNeighbors=0;
        IndexType idx1;
        def->TransformPhysicalPointToIndex(point,idx1);
        if (! def->GetLargestPossibleRegion().IsInside(idx1)){
            return false;
        }
        typename itk::NeighborhoodIterator<DeformationFieldType>::RadiusType r;
        r.Fill(radius);
        typename itk::NeighborhoodIterator<DeformationFieldType>  iterator( r, def, def->GetLargestPossibleRegion());
        LOGV(7)<<VAR(idx1)<<endl;
        iterator.SetLocation(idx1);
        double weightSum=0.0;
        for (int i=0;i<iterator.Size();++i){
            LOGV(7)<<VAR(i)<<endl;
            bool insideTMP;
#if 0           
            DeformationType d=iterator.GetPixel(i,insideTMP);
            LOGV(7)<<VAR(insideTMP)<<" "<<VAR(d)<<endl;
#endif
            IndexType idx=iterator.GetIndex(i);
            insideTMP=def->GetLargestPossibleRegion().IsInside(idx);
            if (insideTMP){
                inside=true;
                IndexType idx=iterator.GetIndex(i);
                LOGV(7)<<VAR(i)<<" "<<VAR(idx)<<endl;
                PointType pt;
                def->TransformIndexToPhysicalPoint(idx,pt);
                LOGV(7)<<VAR(point)<<" "<<VAR(pt)<<" "<<VAR(point-pt)<<endl;
                double w=exp(-0.5*(point-pt).GetSquaredNorm()/(sigma*sigma));
                LOGV(7)<<VAR(w)<<endl;
                weightSum+=w;
                neighbors[nNeighbors++]=std::make_pair(idx,w);
            }

        }
        neighbors.resize(nNeighbors);
        if (inside && weightSum>0){
            for (int i=0;i<nNeighbors;++i){
                LOGV(7)<<VAR(i)<<" "<<VAR( neighbors[i].second) << " " << VAR(weightSum)<<endl;
                neighbors[i].second/=weightSum;
            }
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
    inline int sign(double s){

        if (s>=0) return 1;
        if (s<0) return -1;
        return 0;
    }
    
   
};
