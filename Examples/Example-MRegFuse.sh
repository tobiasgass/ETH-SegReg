#/bin/bash

binDir=/Users/tobi/Arbeit/src/ETH-SegReg/build/bin
dataDir=/Users/tobi/Arbeit/src/ETH-SegReg/data

if [ ! -e $dataDir/List.IDs ]
then
    bash $dataDir/prepareData.sh
fi

N=`wc -l $dataDir/List.IDs | awk '{print $1}'`

mkdir -p MRegFuseData/inputDeformations
mkdir -p MRegFuseData/outputDeformations

##generate random pairwise registrations
echo -n "" >MRegFuseData/pairwiseDeformations.List
for n1 in `seq  1 $N`
do
    for n2 in `seq  1 $N | grep -v $n1`
    do
	#random, doesnt work well
	#$binDir/GenerateDeformation2D --target $dataDir/Images/img-$n2.png --out MRegFuseData/inputDeformations/def-$n1-$n2.mha --linear
	##use SRS to register images. The segmentation  &coherece utility of SRS is not used, therefore it is plain MRF-based registration
	$binDir/SRS2D-Bone --t $dataDir/Images/img-$n2.png --a $dataDir/Images/img-$n1.png --sa $dataDir/Segmentations/seg-$n1.png --ta MRegFuseData/inputDeformations/deformedAtlasImage-$n1-$n2.png --tsa MRegFuseData/inputDeformations/deformedAtlasSegmentation-$n1-$n2.png --T MRegFuseData/inputDeformations/def-$n1-$n2.mha --cp 0 --sp 0 --su 0

	##store registration result in file list for later access
	echo "$n1 $n2 `pwd`/MRegFuseData/inputDeformations/def-$n1-$n2.mha" >>MRegFuseData/pairwiseDeformations.List
    done
done


##run MRegFuse
#parameters:
w=0.1 #smoothness of the fused registration, larger value -> smoother result
s=10  #gamma in the paper, exponent for the metric, eg NCC^s
refineSeamIter=10 #number of iterations of folding removal procedure. set to 0 to disable (folding may then occur).
$binDir/RegProp2D --i $dataDir/List.Images\
		       --true $dataDir/List.DeformationFields\
		       --T MRegFuseData/pairwiseDeformations.List \
		       --O MRegFuseData/outputDeformations/ \
		       --groundTruthSegmentations $dataDir/List.Segmentations \
		       --MRF \
		       --s $s --w $w --refineSeamIter $refineSeamIter \
		       --maxHops 10  --runEndless | grep error
