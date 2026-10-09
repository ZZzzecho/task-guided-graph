"""Build full-scale frozen H=W F(c,E), collecting diagnostic-only cosines."""
from __future__ import annotations
import argparse, csv, gzip, json, time
from pathlib import Path
import numpy as np
from graph_mvp.joint_evidence_repr import JointEvidenceMatrixBuilder,format_joint_input
from graph_mvp.patient_repr import PatientMatrixCacheWriter,PatientMatrixDataset,Qwen3EmbeddingEncoder,load_concept_prototypes
from graph_mvp.patent_longtext import write_json,digest
from graph_mvp.representation_diagnostics import distribution

def column_cosine(a,b):
    import torch
    denom = torch.linalg.vector_norm(a,dim=0)*torch.linalg.vector_norm(b,dim=0)
    values = torch.sum(a*b,dim=0)/denom.clamp_min(1e-15)
    values = torch.where(denom>1e-15,values,torch.full_like(values,float('nan')))
    return values.detach().float().cpu().numpy()

def evidence_diversity(traces):
    from collections import Counter
    counts = Counter(x['evidence_sha256'] for x in traces)
    p = len(traces)
    return {'candidate_chunks':traces[0]['candidate_chunks'],'unique_evidence_sets':len(counts),
        'dominant_set_fraction':max(counts.values())/p,
        'same_evidence_pair_fraction':sum(v*(v-1) for v in counts.values())/(p*(p-1)),
        'all_concepts_same_evidence':len(counts)==1,
        'joint_input_tokens_mean':float(np.mean([t['input_tokens'] for t in traces])),
        'joint_input_tokens_max':max(t['input_tokens'] for t in traces),
        'selected_chunks_mean':float(np.mean([len(t['chunks']) for t in traces])),
        'truncated_fraction':float(np.mean([t['truncated'] for t in traces]))}

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('train','prototypes','projection','output','reference-cache'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--documents',type=int,default=2048)
    p.add_argument('--max-length',type=int,default=1024)
    p.add_argument('--chunk-max-tokens',type=int,default=128)
    p.add_argument('--top-k',type=int,default=6)
    p.add_argument('--encode-batch-size',type=int,default=8)
    args = p.parse_args(argv)
    if args.output.exists():
        p.error('New cache output required')
    import torch
    prototypes,meta = load_concept_prototypes(args.prototypes)
    reference = PatientMatrixDataset(args.reference_cache)
    if tuple(meta['concept_ids']) != reference.concept_ids or meta['encoder_id'] != args.model:
        raise ValueError('Concept/prototype/model mismatch')
    encoder = Qwen3EmbeddingEncoder(args.model,dtype='bfloat16',local_files_only=True,max_length=args.max_length)
    with np.load(args.projection,allow_pickle=False) as projection:
        mean,matrix = projection['mean'],projection['matrix']
    builder = JointEvidenceMatrixBuilder(encoder,prototypes,meta['concept_texts'],
        representation_dim=reference.hidden_size,projection_mean=mean,projection_matrix=matrix,
        top_k=args.top_k,chunk_max_tokens=args.chunk_max_tokens,encode_batch_size=args.encode_batch_size)
    old_mean,old_matrix = reference.load_projection()
    if not np.array_equal(old_matrix,builder.projection_matrix.numpy()) or not np.array_equal(old_mean,builder.projection_mean.numpy()):
        raise ValueError('Effective projection differs from prior experiment')
    prototype_full = builder.prototypes.T.float()
    concept_only_full = builder._pooled([format_joint_input(c,'') for c in meta['concept_texts']]).T
    prototype_h = builder.project_full_states(prototype_full[None])[0]
    concept_only_h = builder.project_full_states(concept_only_full[None])[0]
    names = ['native_vs_prototype','native_vs_concept_only','projected_vs_prototype','projected_vs_concept_only']
    values = {k:[] for k in names}
    old_values = {k:[] for k in names[2:]}
    for matrices,_ in reference.iter_shards():
        for h in matrices:
            tensor = torch.as_tensor(np.array(h),dtype=torch.float32,device=prototype_h.device)
            old_values[names[2]].append(column_cosine(tensor,prototype_h))
            old_values[names[3]].append(column_cosine(tensor,concept_only_h.to(tensor)))
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)
    pipeline = builder.projection_metadata()
    prepared_report = json.loads((args.train.parent/'preparation_report.json').read_text(encoding='utf-8'))
    pipeline.update(cache_build_complete=False,evidence_audit='evidence.jsonl',
        candidate_token_budget=prepared_report['candidate_token_budget'],
        source_preparation_report_sha256=digest(args.train.parent/'preparation_report.json'))
    writer = PatientMatrixCacheWriter(args.output,meta['concept_ids'],builder.hidden_size,dtype='float32',
        encoder_id=args.model,temperature=None,encoder_hidden_size=builder.encoder_hidden_size,
        representation_reduction=builder.projection_kind+'_linear_joint',projection_sha256=pipeline['projection_sha256'],
        representation_pipeline=pipeline)
    writer._flush_metadata()
    np.savez_compressed(args.output/'projection.npz',mean=builder.projection_mean.numpy(),matrix=builder.projection_matrix.numpy())
    np.savez_compressed(args.output/'diagnostic_baselines.npz',prototype_h=prototype_h.cpu().numpy(),concept_only_h=concept_only_h.cpu().numpy())
    ids,buffer,diversity,reviews = [],[],[],[]
    began = time.monotonic()
    with gzip.open(args.train,'rt',encoding='utf-8-sig',newline='') as stream, \
         (args.output/'evidence.jsonl').open('w',encoding='utf-8') as evidence:
        rows = list(csv.DictReader(stream))
        expected_ids = json.loads((args.train.parent/'preparation_report.json').read_text(encoding='utf-8'))['patent_ids']
        if len(rows) != args.documents or [x['patent_id'] for x in rows] != expected_ids:
            raise ValueError('Prepared document IDs/count/order changed')
        for i,row in enumerate(rows):
            h,full,traces = builder.encode_document(row['text'])
            for key,a,b in ((names[0],full[0],prototype_full.to(full)),(names[1],full[0],concept_only_full.to(full)),
                            (names[2],h[0],prototype_h.to(h)),(names[3],h[0],concept_only_h.to(h))):
                values[key].append(column_cosine(a,b))
            diversity.append({'row_index':i,'patent_id':row['patent_id'],**evidence_diversity(traces)})
            if i < 8:
                order = sorted(range(len(traces)),key=lambda j:(-traces[j]['max_score'],j))
                review_indices = list(dict.fromkeys(order[:4]+order[-4:]))
                reviews.append({'row_index':i,'patent_id':row['patent_id'],'source_text':row['text'],
                    'selection':'first eight full-run documents; high/low ranking score concepts; no additional inference',
                    'examples':[{'concept_id':meta['concept_ids'][j],**traces[j]} for j in review_indices]})
            # Original spans are relative to this bounded source_text, never to another patent.
            compact = []
            for trace in traces:
                trace = {k:v for k,v in trace.items() if k not in ('input_text','evidence_text')}
                trace['chunks'] = [{k:v for k,v in x.items() if k!='text'} for x in trace['chunks']]
                compact.append(trace)
            evidence.write(json.dumps({'row_index':i,'subject_id':row['patent_id'],'source_text':row['text'],
                'concept_evidence':compact},ensure_ascii=False,allow_nan=False)+'\n')
            buffer.append(h.detach().float().cpu().numpy()[0])
            ids.append(row['patent_id'])
            del full,h,traces
            if len(buffer)==16 or i+1==len(rows):
                writer.write_shard(np.stack(buffer),ids,ids)
                buffer,ids=[],[]
                print(f'[cache] {i+1}/{len(rows)} persisted; elapsed={time.monotonic()-began:.1f}s',flush=True)
    writer.representation_pipeline = {**builder.projection_metadata(),**{k:v for k,v in pipeline.items()
        if k in ('candidate_token_budget','source_preparation_report_sha256','evidence_audit')}}
    writer.representation_pipeline['cache_build_complete']=True
    writer.close()
    comparison = {'documents':len(rows),'concepts':builder.num_concepts,'representation_dim':builder.hidden_size,
        'native_old_unavailable':True,'cosine_is_diagnostic_only':True,
        'prototype_definition':'Original stored frozen Qwen concept prototype; projected baseline uses the identical W',
        'concept_only_definition':'Same Concept/Patent excerpts template with empty evidence; diagnostic only',
        'new':{k:distribution(np.concatenate(v)) for k,v in values.items()},
        'old':{k:distribution(np.concatenate(v)) for k,v in old_values.items()},
        'diversity_summary':{k:distribution([d[k] for d in diversity]) for k in diversity[0] if k not in ('row_index','patent_id')},
        'evidence_diversity_all_documents':diversity,
        'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
        'cuda_devices':{str(index):{'name':torch.cuda.get_device_name(index),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(index),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(index)}
            for index in range(torch.cuda.device_count())} if torch.cuda.is_available() else {},
        'seconds':time.monotonic()-began,'limitations':'Lower prototype cosine or evidence overlap alone does not establish semantic correctness or task benefit.'}
    write_json(args.output/'representation_report.json',comparison)
    write_json(args.output/'evidence_review.json',reviews)
    np.savez_compressed(args.output/'diagnostic_cosines.npz',**{k:np.stack(v) for k,v in values.items()},
        **{'old_'+k:np.stack(v) for k,v in old_values.items()})
    write_json(args.output/'build_integrity.json',{x.name:digest(x) for x in args.output.iterdir()
        if x.is_file() and x.name!='build_integrity.json'})
    print('[cache] complete; no task-model training performed',flush=True)

if __name__ == '__main__':
    main()
