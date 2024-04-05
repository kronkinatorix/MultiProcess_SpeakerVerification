test_embeds_manifest_json="$main_folder/"
main_folder=""
unver_audio_dir=""
nemo_main=""

mkdir -p $unver_audio_dirtmp
mkdir -p $unver_audio_dirnewfiles
find $unver_audio_dirnewfiles/ -mindepth 1 -maxdepth 1 -name "*.srt" -exec mv {} $unver_audio_dirnewfiles/srt/ \;

# Move old files to tmp to clear the main directory

# Process the files in newfiles and move to downsampled
for file in $unver_audio_dir*.wav; do
	echo "downsampling $file"
  ffmpeg -n -i "$file" -ac 1 -ar 16000 "$unver_audio_dir$(basename "$file")"
done

# Move processed files back to their original location; Correct directory and extension

for file in $unver_audio_dirnewfiles/*.wav; do
	mv "$file" $unver_audio_dir;
done

# Your existing renaming logic for NeMo copy seems correct
q=1
if [[ -e "$main_folder/test_manifest_embeds" ]]; then
    if [[ -e "$main_folder/test_manifest_embeds.old" ]]; then
        while [[ -e "$main_folder/test_manifest_embeds.old$q" ]]; do
            ((q++))
        done
        mv "$main_folder/test_manifest_embeds.old" "$main_folder/test_manifest_embeds.old$q"
    fi
    mv "$main_folder/test_manifest_embeds" "$main_folder/test_manifest_embeds.old"
else
    echo "The original file does not exist."
fi

# JSON entry generation for each audio file
for file in $unver_audio_dir*.wav; do
  if [[ -e "$unver_audio_dir$(basename "$file")" ]]; then
    echo "{\"audio_filepath\": \"$unver_audio_dir$(basename "$file")\", \"duration\": $(ffprobe -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file"), \"label\": \"unlabelled\"}" >> "$test_embeds_manifest_json"
  fi
done

cd /home/mb/gitclones/NeMo
source venv/bin/activate
python3 $nemo_main/examples/speaker_tasks/recognition/extract_speaker_embeddings.py --manifest="$test_embeds_manifest_json"

python3 $nemo_mainmultiprocess_speaker_verificaiton.py
