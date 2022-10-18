#!/bin/bash

root_dir=/mnt/e/DATA/CHRISTIAN/KULIAH/TA/code/dataset/cubicasa5k
#
merge_dir=$root_dir/merge
folder=$root_dir/backup/dataset_used_$(date +"%d%b%Y_%H-%M")
mkdir "$folder"
#
iter=0
#
for file in $merge_dir/*.png ; do
    cp "$file" "$folder/$(basename $file)"
    iter=$((iter + 1))
done
#
echo "Copied $iter images"