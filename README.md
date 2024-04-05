doubt anyone will ever read this but I somehow managed to put this together to use NeMo speaker verification to put together datasets for training audio models. The python script expects a few things to work that can be easily changed:

1. text files for each named speaker you're looking to verify. In these text files are full paths to arbitrary number of audio clips containing audio from a known speaker. i've used ten in my experiments which has produced great resutls. The 
2. the location of a results.txt file to to record results. 
3. THe location of the directory containing audio clips from unknown speakers. This is entered into refreshlinepath() which is called frequently to prevent errors cropping up. Since the script uses multiprocessing to go faster, it's important that it always knows what files are still left to be analyzed. 
4. Folders to move matched audio clips to. The script will move every audio clip it verifies to these folders

I don't really understand WHY this is the case but you seem to need to run extract_speaker_embeddings.py from the NeMo toolkit on both the known and unknown speaker audio clips for this script to work correctly. I'm pretty sure the NeMo functions I'm using are getting embeddings dynamically, or at least that's what it looks like the code is doing, but whatever. 

That's about it. This is REALLY good at matching  known speakers and makes assembling huge datasets from things like tv shows a breeze.

In case anyone's interested my pipeline looks like: 

1. Original files (movies, tv shows, whatever)
2. ffmpeg audio extraction
3. whisperx diarization
4. split audio track by srt (I use a modified version of this script: https://github.com/ZovutVanya/split-video-by-srt-python/blob/main/srt-split.py)
5. Use UVR to separate the vocals from everything else (I've been chipping away at automating this - most of the out-of-the-box solutions I've found use models that I haven't found to be super high quality for vocal removal from tv shows, movies, etc.)
6. Downsample audio with 
for file in /.............*.wav; do
	echo "downsampling $file"
  ffmpeg -n -i "$file" -ac 1 -ar 16000 "/................./$(basename "$file")"
7. get embeddings of known / unknown speakers 
unknown:
test_embeds_manifest_json="/home/mb/gitclones/NeMo/copy/test_manifest_embeds.json"
for file in /............*.wav; do
  if [[ -e "............/$(basename "$file")" ]]; then
    echo "{\"audio_filepath\": \"/..........$(basename "$file")\", \"duration\": $(ffprobe -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file"), \"label\": \"infer\"}" >> "$test_embeds_manifest_json"
  fi
done
8. run this script. 

the script moves at a brisk pace and is really accurate. If anyone sees this or is interested I could clean up the bash script I've been using to run through everything except UVR.
