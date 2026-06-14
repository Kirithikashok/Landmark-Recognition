#!/bin/bash

# Number of parallel downloads
NUM_PROC=1

# Dataset split: train, test, index
SPLIT=$1

# Upper limit:
# train -> 499
# test  -> 19
# index -> 99
N=$2

echo "============================================================"
printf "%-15s %-15s\n" "FILE" "STATUS"
echo "============================================================"

download_check_and_extract() {

    local i=$1

    images_file_name="images_${i}.tar"
    images_md5_file_name="md5.images_${i}.txt"

    images_tar_url="https://s3.amazonaws.com/google-landmark/${SPLIT}/${images_file_name}"
    images_md5_url="https://s3.amazonaws.com/google-landmark/md5sum/${SPLIT}/${images_md5_file_name}"

    # Download silently
    curl -s -L -O "$images_tar_url"
    curl -s -L -O "$images_md5_url"

    # Verify files exist
    if [ ! -f "$images_file_name" ]; then
        printf "%-15s %-15s\n" "$images_file_name" "DOWNLOAD FAIL"
        return
    fi

    if [ ! -f "$images_md5_file_name" ]; then
        printf "%-15s %-15s\n" "$images_file_name" "MD5 MISSING"
        return
    fi

    printf "%-15s %-15s\n" "$images_file_name" "DOWNLOADED"

    # Calculate MD5
    md5_1=$(md5sum "$images_file_name" | awk '{print $1}')

    # Read expected MD5
    md5_2=$(awk '{print $1}' "$images_md5_file_name")

    # Compare
    if [ "$md5_1" = "$md5_2" ]; then

        printf "%-15s %-15s\n" "$images_file_name" "MD5 OK"

        tar -xf "$images_file_name"

        if [ $? -eq 0 ]; then
            printf "%-15s %-15s\n" "$images_file_name" "EXTRACTED"
        else
            printf "%-15s %-15s\n" "$images_file_name" "EXTRACT FAIL"
        fi

    else

        printf "%-15s %-15s\n" "$images_file_name" "MD5 FAIL"

    fi
}

# Download in batches
for i in $(seq 0 $NUM_PROC $N)
do

    upper=$((i + NUM_PROC - 1))

    if [ $upper -gt $N ]; then
        upper=$N
    fi

    for j in $(seq -f "%03g" $i $upper)
    do
        download_check_and_extract "$j" &
    done

    wait

done

echo "============================================================"
echo "ALL TASKS COMPLETED"
echo "============================================================"