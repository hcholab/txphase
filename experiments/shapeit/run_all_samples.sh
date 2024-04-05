#!/usr/bin/env bash

OUTPUT_DIR=$1
export UKB_REF_PANEL_SIZE=$2

MAIN_DIR=$(git rev-parse --show-toplevel)
SIZE_DIR=$OUTPUT_DIR/$UKB_REF_PANEL_SIZE

source $MAIN_DIR/data/datasets/ukb/env_all.sh

MISSING_SITES_DIR=$MAIN_DIR/data/datasets/ukb/targets/formatted/missing_sites

N_CPUS=16
RED='\033[1;31m'
NC='\033[0m' # No Color

mkdir -p $SIZE_DIR && \
echo -e "${RED}### Phasing ###${NC}" && \
/usr/bin/time -f "%e %M" -o $SIZE_DIR/time_mem.txt $MAIN_DIR/shapeit-bin/phase_common_static \
    --input $INPUT \
    --reference $VCF_REF_PANEL \
    --map $GMAP \
    --region $CHR \
    --filter-maf 0.001 \
    --thread $N_CPUS \
    --output $SIZE_DIR/phased.bcf \
    --log $SIZE_DIR/log.txt && \
mkdir -p $SIZE_DIR/tmp && \
echo -e "${RED}### Spliting ###${NC}" && \
bcftools +split $SIZE_DIR/phased.bcf -o $SIZE_DIR/tmp -O b && \
echo -e "${RED}### Merging ###${NC}" && \
for sample in $SIZE_DIR/tmp/*; do
    sample_renamed=$(basename $sample)
    sample_renamed=$SIZE_DIR/tmp/${sample_renamed:0:15}.bcf
    mv $sample $sample_renamed
done && \
echo -n > $SIZE_DIR/tmp/merge.list && \
for sample in $SIZE_DIR/tmp/*.bcf; do
    bcftools index $sample
    missing_sites_sample=$(basename $sample)
    missing_sites_sample=$MISSING_SITES_DIR/$missing_sites_sample
    sample_dir=${sample%%.*}
    mkdir -p $sample_dir
    bcftools isec -C -p $sample_dir $sample $missing_sites_sample
    bcftools convert $sample_dir/0000.vcf -O b -o $sample_dir/0000.bcf
    bcftools index $sample_dir/0000.bcf
    echo "$sample_dir/0000.bcf" >> $SIZE_DIR/tmp/merge.list
done && \
bcftools merge -l $SIZE_DIR/tmp/merge.list -O b -o $SIZE_DIR/phased_remove_imputed.bcf && \
echo -e "${RED}### Computing SERs ###${NC}" && \
$MAIN_DIR/scripts/switch-error-rate.sh $SIZE_DIR/phased_remove_imputed.bcf $PARENTS $TRIO $SIZE_DIR && \
mv $SIZE_DIR/ser.txt $SIZE_DIR/ser_remove_imputed.txt && \
$MAIN_DIR/scripts/switch-error-rate.sh $SIZE_DIR/phased.bcf $PARENTS $TRIO $SIZE_DIR && \
rm -rf $SIZE_DIR/tmp
