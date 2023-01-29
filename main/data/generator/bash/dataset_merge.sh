read -p "Are you sure to repopulate merge files? " ans

if [ "$ans" == "y" ] || [ "$ans" == "Y" ]; then
    root_dir=/mnt/e/DATA/CHRISTIAN/KULIAH/TA/code/dataset/cubicasa5k
    merge_dir=$root_dir/merge
    colorful_dir=$root_dir/colorful
    high_quality_dir=$root_dir/high_quality
    high_quality_architectural_dir=$root_dir/high_quality_architectural

    function copy_files {
        copy="";
        iter=0;
        for file in $1/*scaled.png ; do
            copy="cp $file \"$merge_dir/$(basename $directory)_$(basename $file)\""
            iter=$((iter + 1))
        done
        if ((iter < 3)) ; then
            $(eval $copy)
            echo "copied $directory"
        fi
    }

    for directory in $colorful_dir/*/ ; do
        copy_files $directory
    done

    for directory in $high_quality_dir/*/ ; do
        copy_files $directory
    done
fi