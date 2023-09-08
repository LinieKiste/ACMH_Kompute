#!/bin/sh

help(){
        echo "Usage: pipeline.sh -d [./path/to/dataset]"
        echo "the dataset directory should contain an \"images\" directory"
        echo "use -f to run the full SfM+MVS pipeline. Otherwise, this script only runs SfM+converter"
}

acmh_path=""
DATASET_PATH=""
echo $DATASET_PATH
while getopts "hd:f:" option; do
   case $option in
      h|help) # display Help
         help
         exit;;
      d|dataset) # dataset path
         DATASET_PATH=$OPTARG;;
      f|full) # ACMH executable path
         acmh_path=$OPTARG;;
   esac
done
echo $DATASET_PATH

if [ -z "$DATASET_PATH" ]
then
        echo "no dataset provided!"
        help
else
        ACMH_INPUT_FOLDER=acmh_input

        colmap feature_extractor \
                --database_path $DATASET_PATH/database.db \
                --image_path $DATASET_PATH/images && \
                colmap exhaustive_matcher \
                --database_path $DATASET_PATH/database.db && \
                mkdir $DATASET_PATH/sparse && \
                colmap mapper \
                --database_path $DATASET_PATH/database.db \
                --image_path $DATASET_PATH/images \
                --output_path $DATASET_PATH/sparse

        ./colmap2mvsnet_acm.py \
                --dense_folder $DATASET_PATH \
                --save_folder $DATASET_PATH/$ACMH_INPUT_FOLDER \
                --model_ext .bin

        if [ -n "$acmh_path" ]; then
                $acmh_path $DATASET_PATH/$ACMH_INPUT_FOLDER
        fi
fi

