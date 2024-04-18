# whisper-transcribe-example

Get a transcription text file out of an audio or video file.

## Installation

```sh
poetry install
```

## Run

```sh
poetry run trascribe file.m4a file2.m4a
```

## Split files into multiple chunks with ffmpeg

If you need to split files into multiple chunks because it's too long to send to OpenAI, you can use ffmpeg.

e.g.

```sh
ffmpeg -i input_file.m4a -f segment -segment_time 600 -c copy out%03d.m4a
```