"""Fit initial/fixed-lambda MNGM graphs; no task model, LoRA, or GRPO loop."""
from __future__ import annotations
import argparse, json, subprocess, time
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
from graph_mvp.config import Config
from graph_mvp.estimators import MNGMEstimator,PATIENT_MATRIX_MODE
from graph_mvp.patient_repr import PatientMatrixDataset
from graph_mvp.patent_longtext import cache_ids,code_digest,digest,write_json
from graph_mvp.representation_diagnostics import spectrum,correlation,distribution
from graph_mvp.weighted_glasso import penalty_matrix
from scripts.build_patent_longtext_matrices import evidence_diversity
from scripts.run_patient_mngm import partial_correlation

def covariance_report(s):
    return {'concept_covariance':spectrum(s),
        'off_diagonal_correlation':distribution(correlation(s)[np.triu_indices(len(s),1)])}

def graph_difference(new,old):
    if new.shape != old.shape:
        raise ValueError('Graph axis size differs')
    upper = np.triu_indices(len(new),1)
    a,b = np.abs(new[upper])>1e-10,np.abs(old[upper])>1e-10
    union = np.count_nonzero(a|b)
    return {'new_edges':int(a.sum()),'old_edges':int(b.sum()),'added':int(np.count_nonzero(a&~b)),
        'removed':int(np.count_nonzero(b&~a)),
        'edge_jaccard':float(np.count_nonzero(a&b)/union) if union else 1.,
        'theta_difference_frobenius':float(np.linalg.norm(new-old,'fro')),
        'note':'Same lambda/IDs/projection; descriptive graph change, not resampling stability or task improvement.'}

def fit_graph(args,config):
    ds = PatientMatrixDataset(args.cache)
    reference = PatientMatrixDataset(args.reference_experiment/'cache')
    if ds.concept_ids != reference.concept_ids or ds.hidden_size != reference.hidden_size or len(ds)!=len(reference):
        raise ValueError('New/reference cache dimensions or concept axes differ')
    old_diversity = []
    with (args.reference_experiment/'cache/evidence.jsonl').open(encoding='utf-8') as stream:
        for i,line in enumerate(stream):
            row = json.loads(line)
            if row['row_index'] != i or len(row['concept_evidence']) != ds.num_concepts:
                raise ValueError('Old evidence axis/order invalid')
            old_diversity.append({'row_index':i,'patent_id':str(row['subject_id']),**evidence_diversity(row['concept_evidence'])})
    new_rep = json.loads((args.cache/'representation_report.json').read_text(encoding='utf-8'))
    if [d['patent_id'] for d in old_diversity] != [d['patent_id'] for d in new_rep['evidence_diversity_all_documents']]:
        raise ValueError('Document IDs/order differ between old/new evidence')
    write_json(args.output/'evidence_comparison.json',{
        'scope':'All exact paired training documents and concepts',
        'old':{k:distribution([d[k] for d in old_diversity]) for k in old_diversity[0] if k not in ('row_index','patent_id')},
        'new':new_rep['diversity_summary'],
        'paired_documents':[{'patent_id':a['patent_id'],
            **{k+'_change':b[k]-a[k] for k in ('candidate_chunks','unique_evidence_sets','same_evidence_pair_fraction')}}
            for a,b in zip(old_diversity,new_rep['evidence_diversity_all_documents'])]})
    began = time.monotonic()
    print(f'[graph] materializing {len(ds)} x {ds.hidden_size} x {ds.num_concepts}',flush=True)
    x = ds.materialize(dtype=np.float32)
    estimator = MNGMEstimator(x,PATIENT_MATRIX_MODE,config.mngm,config.solver)
    del x
    penalty = penalty_matrix(ds.num_concepts,args.initial_lambda)
    def progress(info):
        with (args.output/'progress.jsonl').open('a',encoding='utf-8') as stream:
            # Representation CD has no primal residual; JSON stores missing values as null.
            row = {k:None if isinstance(v,float) and not np.isfinite(v) else v for k,v in info.items()}
            stream.write(json.dumps({**row,'seconds':time.monotonic()-began},allow_nan=False)+'\n')
        if info['stage']=='mngm_outer':
            print(f"[MNGM] outer {info['iteration']}/{info['max_iter']}",flush=True)
        else:
            print(f"[{info['stage']}] {info['iteration']}/{info['max_iter']} primal={info['primal']:.3e} dual={info['dual']:.3e}",flush=True)
    initial = estimator.solve_initial(penalty,progress_callback=progress)
    write_json(args.output/'initial_summary.json',initial.info())
    if not initial.converged:
        raise RuntimeError(f'Initial weighted Glasso failed: {initial.message}')
    initial_cov = covariance_report(initial.S)
    write_json(args.output/'initial_covariance_diagnostics.json',initial_cov)
    def save_graph(name,result):
        np.savez_compressed(args.output/name,concept_ids=np.asarray(ds.concept_ids),
            Lambda=penalty,Theta=result.Theta,S=result.S,
            B=result.auxiliary['representation_precision'],partial_corr=partial_correlation(result.Theta))
    save_graph('initial_graph.npz',initial)
    with np.load(args.reference_experiment/'training/initial_graph.npz',allow_pickle=False) as graph:
        if tuple(graph['concept_ids'].astype(str)) != ds.concept_ids:
            raise ValueError('Reference graph concept IDs differ')
        if not np.allclose(graph['Lambda'],penalty,rtol=0,atol=1e-12):
            raise ValueError('Reference graph lambda differs')
        initial_diff = graph_difference(initial.Theta,graph['Theta'])
    old_cov = json.loads((args.reference_experiment/'training/mngm/initial_covariance_diagnostics.json').read_text(encoding='utf-8'))
    summary = {'status':'initial_complete','documents':len(ds),'representation_dim':ds.hidden_size,
        'concepts':ds.num_concepts,'lambda':args.initial_lambda,'initial':initial.info(),
        'effective_solver':asdict(config.solver),'mngm_config':asdict(config.mngm),
        'initial_graph_vs_old_initial_graph':initial_diff,'initial_covariance_new':initial_cov,
        'initial_covariance_old':old_cov,'task_training_performed':False,'grpo_updates':0,
        'scope':'Initial B=I graph plus one fixed-lambda alternating MNGM fit; no task reward or evaluation'}
    write_json(args.output/'summary.json',summary)
    print('[graph] initial graph complete; fitting fixed-lambda MNGM',flush=True)
    fitted = estimator.solve(penalty,initial_theta=initial.Theta,
        initial_representation_precision=initial.auxiliary['representation_precision'],progress_callback=progress)
    write_json(args.output/'fitted_summary.json',fitted.info())
    summary.update(status='complete' if fitted.converged else 'fixed_lambda_fit_failed',fitted=fitted.info(),seconds=time.monotonic()-began)
    if fitted.converged:
        save_graph('fitted_graph.npz',fitted)
        write_json(args.output/'fitted_covariance_diagnostics.json',covariance_report(fitted.S))
    write_json(args.output/'summary.json',summary)
    if not fitted.converged:
        raise RuntimeError(f'Fixed-lambda MNGM failed: {fitted.message}; initial graph retained')
    print('[graph] complete; second-stage training was not started',flush=True)

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('cache','reference-experiment','config','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--initial-lambda',type=float,default=.8)
    p.add_argument('--solver-max-iter',type=int,help='Explicitly increase the inner solver budget; keep all original tolerances')
    args = p.parse_args(argv)
    if args.output.exists():
        p.error('Choose a new graph output')
    config = Config.load(args.config)
    original_solver = asdict(config.solver)
    reference_manifest = json.loads((args.reference_experiment/'experiment_manifest.json').read_text(encoding='utf-8'))
    if digest(args.config) != reference_manifest['input_hashes']['config']:
        raise ValueError('MNGM configuration changed from reference')
    if args.solver_max_iter is not None:
        if args.solver_max_iter < config.solver.max_iter:
            p.error('--solver-max-iter must be at least the original solver budget')
        config = replace(config,solver=replace(config.solver,max_iter=args.solver_max_iter))
    new_ids,_ = cache_ids(args.cache)
    old_ids,_ = cache_ids(args.reference_experiment/'cache')
    if new_ids != old_ids:
        raise ValueError('New/reference cache document IDs/order differ')
    for cache in (args.cache,args.reference_experiment/'cache'):
        integrity = json.loads((cache/'build_integrity.json').read_text(encoding='utf-8'))
        for name,expected in integrity.items():
            if digest(cache/name) != expected:
                raise ValueError(f'Cache checksum mismatch: {cache/name}')
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True,check=False)
    manifest = {'status':'running','git_sha':revision.stdout.strip() if revision.returncode==0 else None,
        'cache':str(args.cache.resolve()),'reference_experiment':str(args.reference_experiment.resolve()),
        'config_sha256':digest(args.config),'cache_integrity_sha256':digest(args.cache/'build_integrity.json'),
        'original_solver':original_solver,'effective_solver':asdict(config.solver),
        'mngm_config':asdict(config.mngm),'initial_lambda':args.initial_lambda,'documents':len(new_ids),
        'source_sha256':{name:code_digest(root/name) for name in
            ('scripts/fit_patent_longtext_graph.py','graph_mvp/estimators.py','graph_mvp/weighted_glasso.py')},
        'task_training_performed':False,'grpo_updates':0,'gpu_encoding_performed':False}
    args.output.mkdir(parents=True)
    write_json(args.output/'run_manifest.json',manifest)
    began = time.monotonic()
    try:
        fit_graph(args,config)
        manifest['status'] = 'complete'
    except Exception as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        manifest['seconds'] = time.monotonic()-began
        write_json(args.output/'run_manifest.json',manifest)

if __name__ == '__main__':
    main()
