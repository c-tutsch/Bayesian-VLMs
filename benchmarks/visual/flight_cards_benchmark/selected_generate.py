#!/usr/bin/env python3
"""Render Bayesian flight conversations as one PNG per question.
Requires: pip install pillow. Supports JSONL and JSON records with messages.
"""
import argparse
import json
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

FLIGHT_RE = re.compile(r'Flight\s+([123])\s*:\s*departure time:\s*([^,\n]+),\s*duration:\s*([^,\n]+),\s*number of stops:\s*(\d+),\s*price:\s*\$([\d,]+)', re.I)
PICK_RE = re.compile(r'The best option is Flight\s+([123])', re.I)
FEEDBACK_RE = re.compile(
    r'Your option Flight ([123]) is (correct|incorrect)'
    r'(?:\. I prefer Flight ([123]))?',
    re.I
)

def font(size, bold=False):
    candidates = (["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if bold else ["/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"])
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()

F_TITLE=font(27,True); F_LABEL=font(13,True); F_VALUE=font(25,True); F_SMALL=font(16); F_BADGE=font(14,True); F_PRICE=font(34,True)


def records(path):
    raw=path.read_text(encoding='utf-8-sig')
    try:
        obj=json.loads(raw)
        if isinstance(obj,list): return obj
        if isinstance(obj,dict) and 'messages' in obj: return [obj]
        raise ValueError('Expected records with messages, not all_options. Use the Bayesian conversation JSONL file.')
    except json.JSONDecodeError:
        return [json.loads(line) for line in raw.splitlines() if line.strip()]



def extract(path):
    samples = []
    known_choices = {}

    for record_no, rec in enumerate(records(path), 1):
        pending = None
        round_no = 0
        record_idx = rec.get('idx', record_no)

        for msg in rec.get('messages', []):
            content = msg.get('content', '')

            if not isinstance(content, str):
                continue

            if msg.get('role') == 'user':

                # Feedback refers to the PREVIOUS round
                feedback = FEEDBACK_RE.search(content)

                if feedback and pending is not None:
                    predicted = feedback.group(1)
                    result = feedback.group(2).lower()
                    preferred = feedback.group(3)

                    if result == 'correct':
                        user_choice = predicted
                    else:
                        user_choice = preferred

                    pending['selected'] = user_choice

                    key = (
                        pending['record_idx'],
                        pending['round'],
                        tuple(map(tuple, pending['flights']))
                    )
                    known_choices[key] = user_choice

                # Extract the NEXT flight question
                matches = FLIGHT_RE.findall(content)

                if (
                    len(matches) == 3
                    and {m[0] for m in matches} == {'1', '2', '3'}
                ):
                    round_no += 1

                    pending = {
                        'record_idx': record_idx,
                        'record_no': record_no,
                        'round': round_no,
                        'flights': matches,
                        'selected': None
                    }

                    samples.append(pending)

    # Recover feedback from longer conversation prefixes
    for sample in samples:
        key = (
            sample['record_idx'],
            sample['round'],
            tuple(map(tuple, sample['flights']))
        )

        if sample['selected'] is None:
            sample['selected'] = known_choices.get(key)

    return samples


def draw_sample(sample, path):
    width,height=1200,470
    img=Image.new('RGB',(width,height),'#f6f8fc')
    d=ImageDraw.Draw(img)
    d.rectangle((0,0,width,9),fill='#345eea')
    d.text((35,34),'FLIGHT CHOICE BENCHMARK',font=F_LABEL,fill='#345eea')
    d.text((35,65),'Choose one of the three flight options',font=F_TITLE,fill='#18243b')
    d.text((930,41),f'RECORD {sample["record_idx"]}  ·  ROUND {sample["round"]}',font=F_LABEL,fill='#66758e')
    for i,(number,departure,duration,stops,price) in enumerate(sorted(sample['flights'],key=lambda x:int(x[0]))):
        x=20+i*395; y=125; w=370; h=322
        selected=(sample['selected']==number)
        border='#13b65b' if selected else '#dfe5f0'
        bg='#f0fff6' if selected else '#ffffff'
        d.rounded_rectangle((x,y,x+w,y+h),radius=19,fill=bg,outline=border,width=3 if selected else 2)
        d.rounded_rectangle((x+20,y+20,x+119,y+52),radius=9,fill='#eef2ff')
        d.text((x+34,y+28),f'FLIGHT {number}',font=F_BADGE,fill='#345eea')
        if selected:
            d.rounded_rectangle((x+219,y+20,x+348,y+52),radius=13,fill='#d7fbe5',outline='#13b65b',width=1)
            d.text((x+233,y+28),'✓ SELECTED',font=F_BADGE,fill='#087c39')
        d.text((x+22,y+81),'DEPARTURE',font=F_LABEL,fill='#66758e')
        d.text((x+22,y+106),departure,font=F_VALUE,fill='#18243b')
        d.text((x+210,y+81),'DURATION',font=F_LABEL,fill='#66758e')
        d.text((x+210,y+112),duration,font=F_SMALL,fill='#18243b')
        d.line((x+22,y+158,x+w-22,y+158),fill='#dfe5f0',width=2)
        d.text((x+22,y+179),'Number of stops',font=F_SMALL,fill='#66758e')
        stoptext='Nonstop' if int(stops)==0 else f'{stops} stop'+('s' if int(stops)!=1 else '')
        d.text((x+260,y+179),stoptext,font=F_SMALL,fill='#18243b')
        d.line((x+22,y+221,x+w-22,y+221),fill='#dfe5f0',width=2)
        d.text((x+22,y+242),'TOTAL PRICE',font=F_LABEL,fill='#66758e')
        d.text((x+22,y+267),'$'+price,font=F_PRICE,fill='#18243b')
        d.text((x+230,y+287),'per passenger',font=F_LABEL,fill='#66758e')
    img.save(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--jsonl',type=Path,default=Path('data/train/bayesian.jsonl'))
    p.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'output/png_selected')    
    p.add_argument('--count',type=int,default=None,help='Optional limit; omit for all rounds')
    args=p.parse_args()
    if not args.jsonl.is_file(): p.error(f'File not found: {args.jsonl}; pass --jsonl PATH_TO_BAYESIAN_FILE')
    args.out.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for sample in extract(args.jsonl):
        if args.count is not None and len(manifest)>=args.count: break
        filename=f'record_{sample["record_no"]:04d}_round_{sample["round"]:02d}.png'
        draw_sample(sample,args.out/filename)
        manifest.append({
            'file': filename,
            'record_idx': sample['record_idx'],
            'round': sample['round'],
            'user_choice': sample['selected']
        })
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Generated {len(manifest)} PNGs in {args.out.resolve()}')
    print(f'{sum(x["user_choice"] is None for x in manifest)} questions had no user feedback and were left unhighlighted.')

if __name__=='__main__': main()
