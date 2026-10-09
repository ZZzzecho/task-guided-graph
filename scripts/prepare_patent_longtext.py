"""Prepare only the exact graph-training patents from a completed v0.6.2 cache."""
from __future__ import annotations
import argparse, csv, gzip, json
from pathlib import Path
from graph_mvp.patent_longtext import (SNAPSHOT,SOURCES,cache_ids,digest,ensure_archive,
    expand_document,load_sections,request_json,selected_training_rows,write_json)
from graph_mvp.representation_diagnostics import distribution

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',type=Path,required=True)
    p.add_argument('--reference-cache',type=Path,required=True)
    p.add_argument('--raw-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--documents',type=int,default=2048)
    p.add_argument('--candidate-tokens',type=int,default=4096)
    p.add_argument('--chunk-max-tokens',type=int,default=128)
    args = p.parse_args(argv)
    if args.output.exists():
        p.error('Choose a new prepared output directory')
    signature = json.loads((args.reference_cache/'build_signature.json').read_text(encoding='utf-8'))
    if digest(args.train) != signature['train_sha256'] or args.model != signature['qwen_model']:
        p.error('Training input/model differs from reference experiment')
    ids,meta = cache_ids(args.reference_cache,args.documents)
    rows = selected_training_rows(args.train,ids)
    years = {row['patent_id']:int(row['patent_date'][:4]) for row in rows}
    if any(not 2005 <= y <= 2024 for y in years.values()):
        p.error('This workflow requires 2005-2024 granted patents')
    files, paths = [],[]
    for section,(record,template,_,_) in SOURCES.items():
        api = request_json(f'https://zenodo.org/api/records/{record}')
        inventory = {x['key']:x for x in api['files']}
        for year in sorted(set(years.values())):
            name = template.format(year=year)
            if name not in inventory:
                raise ValueError(f'Snapshot missing {name}')
            info = inventory[name]
            print(f'[download] {name}: {int(info["size"])/1024**3:.2f} GiB',flush=True)
            path = ensure_archive(args.raw_dir,record,name,info)
            paths.append((section,year,path))
            files.append({'section':section,'year':year,'file':str(path),
                'source':f'https://zenodo.org/records/{record}/files/{name}?download=1',
                'checksum':info['checksum'],'bytes':info['size']})
    sections = load_sections(paths,years)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model,trust_remote_code=True,local_files_only=True)
    args.output.mkdir(parents=True)
    audits = []
    with gzip.open(args.output/'train.csv.gz','wt',encoding='utf-8',newline='') as stream, \
         gzip.open(args.output/'length_audit.jsonl.gz','wt',encoding='utf-8') as audit_stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        for i,row in enumerate(rows):
            pid = row['patent_id']
            text,audit = expand_document(row['text'],sections[pid],tokenizer,args.candidate_tokens,args.chunk_max_tokens)
            writer.writerow({**row,'text':text})
            audit.update(patent_id=pid,row_index=i,patent_date=row['patent_date'])
            audits.append(audit)
            audit_stream.write(json.dumps(audit,ensure_ascii=False)+'\n')
            if (i+1)%64 == 0 or i+1 == len(rows):
                print(f'[prepare] {i+1}/{len(rows)} documents',flush=True)
    expanded = sum(a['expanded'] for a in audits)
    if expanded == 0:
        raise ValueError('No selected patent gained original text; refusing to repeat the old run')
    report = {'status':'complete','snapshot':SNAPSHOT,'documents':len(rows),'patent_ids':ids,
        'expanded_documents':expanded,'reference_metadata_sha256':digest(args.reference_cache/'metadata.json'),
        'original_train_sha256':digest(args.train),'expanded_train_sha256':digest(args.output/'train.csv.gz'),
        'candidate_token_budget':args.candidate_tokens,'chunk_max_tokens':args.chunk_max_tokens,
        'section_weights':{k:v[3] for k,v in SOURCES.items()},'sources':files,
        'base_tokens':distribution([a['base_tokens'] for a in audits]),
        'candidate_tokens':distribution([a['candidate_tokens'] for a in audits]),
        'section_lengths':{k:{'available_documents':sum(a['sections'][k]['status']=='available' for a in audits),
            'selected_documents':sum(a['sections'][k]['selected_tokens']>0 for a in audits),
            'raw_tokens':distribution([a['sections'][k]['source_tokens'] for a in audits]),
            'selected_tokens':distribution([a['sections'][k]['selected_tokens'] for a in audits])} for k in SOURCES},
        'scope':'Exact prior graph training IDs only; no validation/test documents or retrieval text modified',
        'drawing_description':'not included: matching archived download source not verified',
        'selection_is_not_relevance_label':True}
    write_json(args.output/'preparation_report.json',report)
    print(f'[prepare] expanded {expanded}/{len(rows)}; {args.output}',flush=True)

if __name__ == '__main__':
    main()
