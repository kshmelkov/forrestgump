#!/usr/bin/python

import json
import csv
import pandas as pd
from pysrt import SubRipFile
from pysrt import SubRipItem
from pysrt import SubRipTime
import os

from fg_constants import *

SHIFT = 16  # seconds


# seg_num starting from zero
def get_segment_shift(seg_num):
    return sum(SEGMENTS[:seg_num]) - seg_num*SHIFT


SEGMENT_SHIFT = map(get_segment_shift, range(TOTAL_SEGMENTS))


def parse_json(filename, i):
    subs = []
    with open(filename, 'r') as annotations:
        shift = SEGMENT_SHIFT[i]*1000
        length = SEGMENTS[i]*1000
        data = json.load(annotations)
        for d in data['annotations']:
            begin = int(d['begin'])
            end = int(d['end'])
            s_begin = max(begin-shift, 0)
            s_end = max(end-shift, 0)
            # TODO figure out what to do when the subtitle in on the middle of SEGMENTS
            if s_end > 0 and s_begin < length:
                text = d['parsed']['text']
            # TODO sometimes there is an empty text
                if len(text) > 0:
                    person = d['parsed']['person'] if 'person' in d['parsed'] else 'NARRATOR'
                    subs.append((s_begin, s_end, text, person))
    return pd.DataFrame.from_records(subs, columns=('begin', 'end', 'text', 'person'))


def merge_subtitles(seg_num):
    df = pd.concat((parse_json(BLIND_ANNOTATIONS, seg_num), parse_json(DIALOGS, seg_num)))
    df = df.sort(columns="begin").reset_index(drop=True)
    return df


def convert_time(millis):
    seconds = millis / 1000
    return SubRipTime(hours=seconds/3600, minutes=seconds/60, seconds=seconds % 60, milliseconds=millis % 1000)


def to_srt(df, filename):
    out = SubRipFile(encoding='utf-8')
    for i, r in df.iterrows():
        begin = convert_time(r['begin'])
        end = convert_time(r['end'])
        out.append(SubRipItem(0, begin, end, r['text']))
    out.save(filename)


def convert_all():
    if not os.path.exists(SUBTITLES_DIR):
        os.mkdir(SUBTITLES_DIR)

    for i in range(TOTAL_SEGMENTS):
        df = merge_subtitles(i)
        csv_path = os.path.join(SUBTITLES_DIR, 'fg_ad_seg%i.csv' % i)
        srt_path = os.path.join(SUBTITLES_DIR, 'fg_ad_seg%i.srt' % i)
        df.to_csv(csv_path, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
        to_srt(df, srt_path)


def convert_all_separate(filename, suffix):
    if not os.path.exists(SUBTITLES_DIR):
        os.mkdir(SUBTITLES_DIR)

    for i in range(TOTAL_SEGMENTS):
        df = parse_json(filename, i)
        csv_path = os.path.join(SUBTITLES_DIR, 'fg_ad_seg%i_%s.csv' % (i, suffix))
        srt_path = os.path.join(SUBTITLES_DIR, 'fg_ad_seg%i_%s.srt' % (i, suffix))
        df.to_csv(csv_path, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
        to_srt(df, srt_path)

if __name__ == '__main__':
    # convert_all()
    convert_all_separate(DIALOGS, 'dialogs')
    convert_all_separate(BLIND_ANNOTATIONS, 'annotations')
