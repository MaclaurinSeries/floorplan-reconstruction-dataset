file=$1
if [ -z "$file" ]
then
      file="input.txt"
fi

root_dir=/mnt/e/DATA/CHRISTIAN/KULIAH/TA/code/dataset/cubicasa5k
merge_dir=$root_dir/merge
colorful_dir=$root_dir/colorful
high_quality_dir=$root_dir/high_quality
high_quality_architectural_dir=$root_dir/high_quality_architectural
textfile=$root_dir/backup/$1

function copy_files {
    copy=""
    iter=0
    echo $1
    for file in $1/*scaled.png ; do
        copy="cp $file \"$merge_dir/$(basename $directory)_$(basename $file)\""
        iter=$((iter + 1))
    done
    if ((iter < 3)) ; then
        #$(eval $copy)
        echo "copied $directory"
    fi
}

function search_files {
    echo "12"
}

while read -r line; do

    for directory in $colorful_dir/*/ ; do
        copy_files $directory
    done
done < $file_dir
