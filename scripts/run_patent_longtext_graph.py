"""Full-scale long-text representation and graph-only experiment entry point."""
from __future__ import annotations
import argparse, json, subprocess, sys, tarfile, time
from pathlib import Path
from graph_mvp import __version__
from graph_mvp.patent_longtext import cache_ids,code_digest,digest,write_json
from graph_mvp.patient_repr import PatientMatrixDataset,load_concept_prototypes

ROOT = Path(__file__).resolve().parents[1]

def commands(args,root=ROOT):
    def absolute(path):
        return path.resolve() if path.is_absolute() else (root/path).resolve()
    out,data,old,raw = map(absolute,(args.output,args.data_dir,args.reference_experiment,args.raw_dir))
    config,projection = map(absolute,(args.config,args.projection_file))
    prefix = [sys.executable,'-u','-m']
    prepare = [*prefix,'scripts.prepare_patent_longtext','--train',str(data/'train.csv.gz'),
        '--reference-cache',str(old/'cache'),'--raw-dir',str(raw),'--output',str(out/'prepared'),
        '--model',args.qwen_model,'--documents',str(args.documents),'--candidate-tokens',str(args.candidate_tokens),
        '--chunk-max-tokens',str(args.chunk_max_tokens)]
    build = [*prefix,'scripts.build_patent_longtext_matrices','--train',str(out/'prepared/train.csv.gz'),
        '--prototypes',str(data/'concept_prototypes.npz'),'--projection',str(projection),'--output',str(out/'cache'),
        '--reference-cache',str(old/'cache'),'--model',args.qwen_model,'--documents',str(args.documents),
        '--max-length',str(args.max_length),'--chunk-max-tokens',str(args.chunk_max_tokens),
        '--top-k',str(args.top_k),'--encode-batch-size',str(args.encode_batch_size)]
    graph = [*prefix,'scripts.fit_patent_longtext_graph','--cache',str(out/'cache'),
        '--reference-experiment',str(old),'--config',str(config),'--output',str(out/'graph'),
        '--initial-lambda','0.8']
    return out,data,old,config,projection,[('prepare',prepare),('build',build),('graph',graph)]

def pack_reports(out):
    """Reports/small diagnostics only, excluding model weights and H/evidence cache."""
    files = [out/'experiment_manifest.json',out/'report.json',out/'prepare.log',out/'build.log',out/'graph.log']
    files += [out/'prepared'/name for name in ('preparation_report.json','length_audit.jsonl.gz')]
    files += [out/'cache'/name for name in ('metadata.json','build_signature.json','build_integrity.json',
        'representation_report.json','diagnostic_cosines.npz','diagnostic_baselines.npz','evidence_review.json')]
    files += list((out/'graph').glob('*.json'))+list((out/'graph').glob('*.npz'))
    bundle = out.parent/(out.name+'_reports.tar.gz')
    with tarfile.open(bundle,'w:gz') as archive:
        for path in files:
            if path.is_file():
                archive.add(path,arcname=str(path.relative_to(out)))
    return bundle

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('outputs/patent_longtext_graph_n2048_b4096_k6_v063'))
    p.add_argument('--data-dir',type=Path,default=Path('data/patents_h04l'))
    p.add_argument('--reference-experiment',type=Path,default=Path('outputs/joint_evidence_full_n2048_k6_v062'))
    p.add_argument('--raw-dir',type=Path,default=Path('/laijizheng/datasets/patentsview_longtext_20241231'))
    p.add_argument('--config',type=Path,default=Path('configs/patient_h04l_main_v1.json'))
    p.add_argument('--projection-file',type=Path,default=Path('data/patents_h04l/cache_main_4096_r64_pca/projection.npz'))
    p.add_argument('--qwen-model',default='/laijizheng/models/Qwen3-Embedding-0.6B')
    p.add_argument('--documents',type=int,default=2048)
    p.add_argument('--candidate-tokens',type=int,default=4096)
    p.add_argument('--max-length',type=int,default=1024)
    p.add_argument('--chunk-max-tokens',type=int,default=128)
    p.add_argument('--top-k',type=int,default=6)
    p.add_argument('--encode-batch-size',type=int,default=8)
    p.add_argument('--plan-only',action='store_true')
    args = p.parse_args(argv)
    if args.documents<2 or args.encode_batch_size<1 or args.top_k<1 or not 8<=args.chunk_max_tokens<=args.max_length<=args.candidate_tokens:
        p.error('Invalid document/batch/token settings')
    out,data,old,config,projection,stages = commands(args,ROOT)
    if out.exists():
        p.error(f'Output already exists: {out}; use a new --output')
    source = json.loads((old/'experiment_manifest.json').read_text(encoding='utf-8'))
    if source['status'] not in ('complete','training_complete_controls_failed'):
        p.error('Reference experiment is not complete')
    inputs = {'train':data/'train.csv.gz','prototypes':data/'concept_prototypes.npz',
        'config':config,'projection':projection}
    for key,path in inputs.items():
        if digest(path) != source['input_hashes'][key]:
            p.error(f'Reference input mismatch: {key}')
    ids,meta = cache_ids(old/'cache',args.documents)
    reference = PatientMatrixDataset(old/'cache')
    _,proto_meta = load_concept_prototypes(inputs['prototypes'])
    if (meta['encoder_id'] != args.qwen_model or proto_meta['encoder_id'] != args.qwen_model
        or tuple(proto_meta['concept_ids']) != reference.concept_ids):
        p.error('Reference/model/prototype axis mismatch')
    integrity = json.loads((old/'cache/build_integrity.json').read_text(encoding='utf-8'))
    for path in ['evidence.jsonl','projection.npz']+[s['matrix'] for s in meta['shards']]:
        if digest(old/'cache'/path) != integrity[path]:
            p.error(f'Reference cache changed: {path}')
    out.mkdir(parents=True)
    try:
        revision = subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=False)
        git_sha = revision.stdout.strip() if revision.returncode==0 else None
    except OSError:
        git_sha = None
    code_paths = ['graph_mvp/patent_longtext.py','graph_mvp/joint_evidence_repr.py',
        'graph_mvp/patient_repr.py','graph_mvp/estimators.py','graph_mvp/weighted_glasso.py',
        'scripts/prepare_patent_longtext.py','scripts/build_patent_longtext_matrices.py',
        'scripts/fit_patent_longtext_graph.py','scripts/run_patent_longtext_graph.py']
    manifest = {'status':'plan_only' if args.plan_only else 'running','graph_mvp_version':__version__,
        'extension_version':'0.6.3-longtext-graph-only','git_sha':git_sha,
        'source_files_sha256':{path:code_digest(ROOT/path) for path in code_paths},
        'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'reference_experiment':str(old),'reference_manifest_sha256':digest(old/'experiment_manifest.json'),
        'input_hashes':{k:digest(v) for k,v in inputs.items()},'sample_ids':ids,
        'commands':dict(stages),'stages':{},'task_training_performed':False,'planned_grpo_updates':0,
        'planned_joint_pairs':args.documents*reference.num_concepts,
        'formula':'H_d[:,c] = W @ F(concept_c, selected_original_evidence_dc)',
        'changes':'Original same-patent long text and bounded section-balanced candidate pool only'}
    write_json(out/'experiment_manifest.json',manifest)
    if args.plan_only:
        print(json.dumps(manifest,indent=2,ensure_ascii=False))
        return
    try:
        for name,command in stages:
            began = time.monotonic()
            print(f'[{name}] starting; log: {out/(name+".log")}',flush=True)
            try:
                with (out/(name+'.log')).open('w',encoding='utf-8') as log:
                    subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                manifest['stages'][name]={'status':'complete','seconds':time.monotonic()-began}
                if name=='build':
                    if len(PatientMatrixDataset(out/'cache')) != args.documents:
                        raise ValueError('New complete cache count mismatch')
                    write_json(out/'cache/build_signature.json',{
                        **manifest['input_hashes'],'expanded_train_sha256':digest(out/'prepared/train.csv.gz'),
                        'candidate_token_budget':args.candidate_tokens,'top_k':args.top_k,
                        'max_length':args.max_length,'chunk_max_tokens':args.chunk_max_tokens,
                        'qwen_model':args.qwen_model,'reference_cache_metadata_sha256':digest(old/'cache/metadata.json')})
                    write_json(out/'cache/build_integrity.json',{x.name:digest(x) for x in (out/'cache').iterdir()
                        if x.is_file() and x.name!='build_integrity.json'})
            except Exception as exc:
                manifest['stages'][name]={'status':'failed','seconds':time.monotonic()-began,'error':str(exc)}
                raise
            finally:
                write_json(out/'experiment_manifest.json',manifest)
            print(f'[{name}] complete',flush=True)
        prep = json.loads((out/'prepared/preparation_report.json').read_text(encoding='utf-8'))
        rep = json.loads((out/'cache/representation_report.json').read_text(encoding='utf-8'))
        graphs = json.loads((out/'graph/summary.json').read_text(encoding='utf-8'))
        write_json(out/'report.json',{'status':'complete','preparation':prep,'representation':rep,'graph':graphs,
            'task_training_performed':False,'grpo_updates':0,
            'interpretation':'Evidence diversity, prototype sensitivity and covariance/graph structure are diagnostics; task benefit remains untested.'})
        manifest['status']='complete'
    except Exception as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(out/'experiment_manifest.json',manifest)
        bundle = pack_reports(out)
        print(f'[reports] {bundle}',flush=True)

if __name__ == '__main__':
    main()
