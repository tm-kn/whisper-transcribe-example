import argparse
import logging
import json
from collections.abc import Iterable, Sequence
import pathlib
import openai
from typing import TypedDict, TextIO

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class Segment(TypedDict):
    text: str


def transcribe_files(file_paths: Sequence[pathlib.Path], *, language: str) -> Iterable[Segment]:
    for file_path in file_paths:
        yield from transcribe_file(file_path, language=language)


def transcribe_file(file_path: pathlib.Path, *, language: str) -> Iterable[Segment]:
    logger.info("Transcribing file %s", file_path)
    client = openai.OpenAI()
    response = client.audio.transcriptions.with_raw_response.create(
        file=pathlib.Path(file_path),
        model="whisper-1",
        language="en",
        response_format="verbose_json",
        timestamp_granularities=["segment"],
    )
    json_response = json.loads(response.text)
    yield from json_response["segments"]


def write_segments_to_file(*, segments: Iterable[Segment], output_file: TextIO):
    for segment in segments:
        line = "".join([
            str(segment["text"]),
            "\n",
        ])
        output_file.write(line)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Split files with ffmpeg if they are too big
    # ffmpeg -i input_file.m4a -f segment -segment_time 00 -c copy out%03d.m4a
    parser.add_argument('filenames', nargs='+')
    parser.add_argument('--output-file', default="out.txt")
    parser.add_argument('--language', default="en")
    return parser.parse_args()


def main():
    args = get_args()
    filenames = [pathlib.Path(filename) for filename in args.filenames]
    segments = transcribe_files(filenames, language=args.language)
    with open(args.output_file, "w") as output_file:
        write_segments_to_file(segments=segments, output_file=output_file)


if __name__ == "__main__":
    main()