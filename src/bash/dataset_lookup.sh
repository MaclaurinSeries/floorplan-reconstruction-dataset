root_dir=/mnt/e/DATA/CHRISTIAN/KULIAH/TA/code/dataset/cubicasa5k
#
merge_dir=$root_dir/merge
textfile=$root_dir/backup/dataset_used_$(date +"%d%b%Y_%H-%M").txt
#
> $textfile
iter=0
#
for file in $merge_dir/*.png ; do
    IFS='_' read -ra ADDR <<< "$(basename $file)"
    echo "$ADDR" >> $textfile
    iter=$((iter + 1))
done
#
echo "Used $iter images"
