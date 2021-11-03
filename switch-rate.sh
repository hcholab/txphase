#!/usr/bin/env bash

INPUT=$1

bcftools index -t $INPUT
bcftools merge -m none -o merged.vcf.gz -O z $INPUT ~/workspace/genome-data/data/giab/father.vcf.gz ~/workspace/genome-data/data/giab/mother.vcf.gz
bcftools +trio-switch-rate merged.vcf.gz -- -p ~/workspace/genome-data/data/giab/trio.ped
