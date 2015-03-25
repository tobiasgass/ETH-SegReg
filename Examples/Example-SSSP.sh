#/bin/bash

binDir=/Users/tobi/Arbeit/src/ETH-SegReg/build/bin
dataDir=/Users/tobi/Arbeit/src/ETH-SegReg/data

if [ ! -e $dataDir/List.IDs ]
then
    bash $dataDir/prepareData.sh
fi

N=`wc -l $dataDir/List.IDs | awk '{print $1}'`

pairwiseRegistrationsDir=PairwiseRegistrations/

mkdir -p $pairwiseRegistrationsDir

SSSPOutputDir=Results-SSSP/

mkdir -p $SSSPOutputDir

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
	    #$binDir/GenerateDeformation2D --target $dataDir/Images/img-$n2.png --out $pairwiseRegistrationsDir/def-$n1-$n2.mha --linear
	    ##use SRS to register images. The segmentation  &coherece utility of SRS is not used, therefore it is plain MRF-based registration
	    $binDir/SRS2D-Bone --t $dataDir/Images/img-$n2.png --a $dataDir/Images/img-$n1.png --sa $dataDir/Segmentations/seg-$n1.png --ta $pairwiseRegistrationsDir/deformedAtlasImage-$n1-$n2.png --tsa $pairwiseRegistrationsDir/deformedAtlasSegmentation-$n1-$n2.png --T $pairwiseRegistrationsDir/def-$n1-$n2.mha --cp 0 --sp 0 --su 0
	    
	    ##store registration result in file list for later access
	    echo "$n1 $n2 `pwd`/$pairwiseRegistrationsDir/def-$n1-$n2.mha" >>$pairwiseRegistrationsDir/pairwiseDeformations.List
	done
    done
fi


##run SSSP

##SSSP requires an additional filelist with the IDs and segmentattion file lists for the atlas(es) to be used. Here we create on by just using the first image of the simulated data as atlas.

head -n 1 $dataDir/List.Segmentations > $SSSPOutputDir/atlasSegmentationFilelist

#parameters:
w=0.1 #smoothness of the fused registration, larger value -> smoother result
r=10  #gamma in the paper, exponent for the metric, eg NCC^s
refineSeamIter=10 #number of iterations of folding removal procedure. set to 0 to disable (folding may then occur).
$binDir/SegProp2D --i $dataDir/List.Images\
		  --T $pairwiseRegistrationsDir/pairwiseDeformations.List \
		  --A $SSSPOutputDir/atlasSegmentationFilelist \
		  --O $SSSPOutputDir/ \
		  --s $r --weighting local

##SSSP does not internally evaluate the results, so we do it now: ($n1 is the used atlas as per our creation of the atlas segmentation file list
echo;
echo "RESULTS:"
echo;
 for n1 in 1
    do
	for n2 in `seq  1 $N | grep -v $n1`
	do
	    groundTruthSegmentation=`cat $dataDir/List.Segmentations | grep "$n2 " | awk '{print $2}'`
	    ##Compare Segmentation before SSSP with groundtruth
	    echo -n " $n1 to $n2 : Dice "`$binDir/CompareSegmentations2D --g $groundTruthSegmentation --s $pairwiseRegistrationsDir/deformedAtlasSegmentation-$n1-$n2.nii | awk '{print $NF}'`" "

	    ##CompareSegmentation after SSSP
	    echo `$binDir/CompareSegmentations2D --g $groundTruthSegmentation --s $SSSPOutputDir/segmentation-weightinglocal-metricNCC-target$n2-hop1.nii | awk '{print $NF}'`
	    
	done
 done | awk 'BEGIN{avg1=0;avg2=0;count=0}{print $0; avg1+=$6; avg2+=$7; count+=1}END{print "Average dice before SSSP: "avg1/count", average dice after SSSP: "avg2/count}'

