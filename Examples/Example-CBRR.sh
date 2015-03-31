#/bin/bash

binDir="echo ~/work/src/ETH-SegReg/build/bin"
dataDir=~/work/src/ETH-SegReg/data

if [ ! -e $dataDir/List.IDs ]
then
    bash $dataDir/prepareData.sh
fi

N=`wc -l $dataDir/List.IDs | awk '{print $1}'`

pairwiseRegistrationsDir=PairwiseRegistrations/

mkdir -p $pairwiseRegistrationsDir

CBRROutputDir=Results-CBRR/

mkdir -p $CBRROutputDir

##generate random pairwise registrations if they don't exist yet
if [ ! -e $pairwiseRegistrationsDir/pairwiseDeformations.List ]
then
    echo "Computing pairwise registrations using MRF-registration"
    echo -n "" >$pairwiseRegistrationsDir/pairwiseDeformations.List
    for n1 in `seq  1 $N`
    do
	for n2 in `seq  1 $N | grep -v $n1`
	do
	    #random, doesnt work well
	    #$binDir/GenerateDeformation2D --target $dataDir/Images/img-$n2.nii --out $pairwiseRegistrationsDir/def-$n1-$n2.mha --linear
	    ##use SRS to register images. The segmentation  &coherece utility of SRS is not used, therefore it is plain MRF-based registration
	    $binDir/SRS2D-Bone --t $dataDir/Images/img-$n2.nii --a $dataDir/Images/img-$n1.nii --sa $dataDir/Segmentations/seg-$n1.nii --ta $pairwiseRegistrationsDir/deformedAtlasImage-$n1-$n2.nii --tsa $pairwiseRegistrationsDir/deformedAtlasSegmentation-$n1-$n2.nii --T $pairwiseRegistrationsDir/def-$n1-$n2.mha --cp 0 --sp 0 --su 0
	    
	    ##store registration result in file list for later access
	    echo "$n1 $n2 `pwd`/$pairwiseRegistrationsDir/def-$n1-$n2.mha" >>$pairwiseRegistrationsDir/pairwiseDeformations.List
	done
    done
fi


##run CBRR
#parameters:
w=0.1 #smoothness of the fused registration, larger value -> smoother result
r=10  #gamma in the paper, exponent for the metric, eg NCC^s
refineSeamIter=10 #number of iterations of folding removal procedure. set to 0 to disable (folding may then occur).
$binDir/CBRR2D --i $dataDir/List.Images\
		       --true $dataDir/List.DeformationFields\
		       --T $pairwiseRegistrationsDir/pairwiseDeformations.List \
	--O $CBRROutputDir/ \
	--groundTruthSegmentations $dataDir/List.Segmentations \
	--g $r --w $w  \
	--maxHops 10  
