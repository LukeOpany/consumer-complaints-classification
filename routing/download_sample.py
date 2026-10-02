"""Create a seeded reservoir sample from CFPB's official July 2026 archive."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import random
from urllib.request import urlretrieve
import zipfile

SOURCE='https://files.consumerfinance.gov/f/documents/CCDB_Export_20_July_2026.zip'
PAGE='https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/'

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--size',type=int,default=10000)
    parser.add_argument('--output',default='data/complaints.csv')
    parser.add_argument('--archive',default='data/CCDB_Export_20_July_2026.zip')
    args=parser.parse_args()
    if not 100 <= args.size <= 100000: parser.error('size must be between 100 and 100000')
    archive=Path(args.archive)
    if not archive.exists():
        archive.parent.mkdir(parents=True,exist_ok=True);urlretrieve(SOURCE,archive)
    rng=random.Random(42);sample=[];eligible=0;total=0
    columns=['Product','Consumer complaint narrative','Date received']
    with zipfile.ZipFile(archive) as z:
        names=[n for n in z.namelist() if n.endswith('.csv')]
        if len(names)!=1: raise ValueError('Expected exactly one CSV in the archive')
        reader=csv.DictReader(io.TextIOWrapper(z.open(names[0]),encoding='utf-8-sig'))
        for row in reader:
            total+=1
            if not row['Consumer complaint narrative'].strip() or not row['Product'].strip(): continue
            eligible+=1;item=(eligible,{key:row[key] for key in columns})
            if len(sample)<args.size: sample.append(item)
            else:
                index=rng.randrange(eligible)
                if index<args.size: sample[index]=item
    if not sample: raise RuntimeError('Archive contains no usable narratives')
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=columns);writer.writeheader()
        writer.writerows(row for _,row in sorted(sample))
    metadata=dict(source_url=SOURCE,source_page=PAGE,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                  source_records=total,source_records_with_narratives=eligible,sample_rows=len(sample),sampling='reservoir; Python random.Random(42)',
                  sample_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  limitation='Sample of this archive segment only, not all historical CFPB complaints or current product traffic.')
    output.with_suffix('.source.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Saved {len(sample)} of {eligible} narrative records to {output}')
