#/bin/bash

binDir="echo /Users/tobi/Arbeit/src/ETH-SegReg/build/bin"
dataDir=/Users/tobi/Arbeit/src/ETH-SegReg/data

if [ ! -e $dataDir/List.IDs ]
then
    bash $dataDir/prepareData.sh
fi

N=`wc -l $dataDir/List.IDs | awk '{print $1}'`


SRSOutputDir=Results-SRS/

mkdir -p $SRSOutputDir

##Generate multilabel atlas segmentation, using first image as atlas
atlasID=1

if [ ! -e $dataDir/Segmentations/seg-$atlasID-multilabel.nii ]
then
    $binDir/SRS2D-Bone --t $dataDir/Images/img-$atlasID.nii --ru 0 --rp 0 --cp 0 --sp 1 --su 1 --st $dataDir/Segmentations/seg-$atlasID-BINARY-GC.nii
    $binDir/CreateMultiLabelAtlasSegmentation2D --automatic $dataDir/Segmentations/seg-$atlasID-BINARY-GC.nii --manual  $dataDir/Segmentations/seg-$atlasID.nii --output $dataDir/Segmentations/seg-$atlasID-multilabel.nii
fi



##run SRS
#parameters:


for i in `seq 1 $N | grep -v $atlasID`
do
    $binDir/SRS2D-Bone --t $dataDir/Images/img-$i.nii --a $dataDir/Images/img-$atlasID.nii --sa $dataDir/Segmentations/seg-$atlasID-multilabel.nii \
		       --sp 1 --su 1 --cp 1 --rp 1e-5 --ru 1 --nSegmentations 3 --auxLabel 2 --tsc 1
    
done
