"""Replay only initial B=I Glasso from a verified H cache, observing loss and KKT."""
from __future__ import annotations
import argparse, dataclasses, json, subprocess, time
from pathlib import Path
import numpy as np
from graph_mvp.config import Config
from graph_mvp.estimators import MNGMEstimator, PATIENT_MATRIX_MODE
from graph_mvp.patient_repr import PatientMatrixDataset
from graph_mvp.patent_longtext import cache_ids, code_digest, digest, write_json
from graph_mvp.weighted_glasso import penalty_matrix


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'reference-experiment', 'config', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--max-iter', type=int, default=6000)
    parser.add_argument('--diagnostic-every', type=int, default=250)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Choose a new diagnostic output directory')
    config = Config.load(args.config)
    old = json.loads((args.reference_experiment/'experiment_manifest.json').read_text())
    if digest(args.config) != old['input_hashes']['config']:
        parser.error('Reference configuration changed')
    if args.max_iter < config.solver.max_iter or args.diagnostic_every < 1:
        parser.error('Do not reduce the original iteration budget; use positive diagnostic interval')
    ids, _ = cache_ids(args.cache)
    ds = PatientMatrixDataset(args.cache)
    reference_ids, _ = cache_ids(args.reference_experiment/'cache', len(ids))
    reference = PatientMatrixDataset(args.reference_experiment/'cache')
    if ids != reference_ids or ds.concept_ids != reference.concept_ids or ds.hidden_size != reference.hidden_size:
        parser.error('Cache sample/concept/projection dimensions differ from reference')
    integrity = json.loads((args.cache/'build_integrity.json').read_text())
    for name in ['evidence.jsonl', 'projection.npz']+[s['matrix'] for s in ds.metadata['shards']]:
        if digest(args.cache/name) != integrity[name]:
            parser.error(f'Cache checksum changed: {name}')
    solver = dataclasses.replace(config.solver, max_iter=args.max_iter)
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    git = subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True,check=False)
    report = {'status':'running','scope':'Initial B=I diagnostic only; no GPU encoding, alternating B fit or task training',
        'git_sha':git.stdout.strip() if git.returncode==0 else None,
        'source_sha256':{p:code_digest(root/p) for p in ('graph_mvp/weighted_glasso.py','graph_mvp/estimators.py',
            'scripts/diagnose_patent_glasso_convergence.py')},
        'cache':str(args.cache.resolve()),'cache_integrity_sha256':digest(args.cache/'build_integrity.json'),
        'config_sha256':digest(args.config),'original_solver':dataclasses.asdict(config.solver),
        'effective_solver':dataclasses.asdict(solver),'concept_lambda':.8,'fixed_B':'identity',
        'documents':len(ids),'concepts':ds.num_concepts,'representation_dim':ds.hidden_size,
        'diagnostic_every':args.diagnostic_every}
    write_json(args.output/'report.json',report)
    began = time.monotonic()
    print('[diagnostic] verified H cache; preparing identical B=I covariance',flush=True)
    estimator = MNGMEstimator(ds.materialize(dtype=np.float32),PATIENT_MATRIX_MODE,config.mngm,solver)
    s = estimator._concept_covariance(np.eye(ds.hidden_size))
    penalty = penalty_matrix(ds.num_concepts,.8)
    np.savez_compressed(args.output/'covariance.npz',S=s,Lambda=penalty)
    snapshots = []
    with (args.output/'loss_trace.jsonl').open('w',encoding='utf-8') as stream:
        def observe(row):
            row = {**row,'seconds':time.monotonic()-began}
            snapshots.append(row)
            stream.write(json.dumps(row,allow_nan=False)+'\n'); stream.flush()
            print('[loss] '+json.dumps(row,allow_nan=False),flush=True)
        result = estimator.concept_solver.solve(s,penalty,diagnostic_callback=observe,
            diagnostic_every=args.diagnostic_every,label='initial_concept_glasso')
    finite = [r for r in snapshots if r['objective_z'] is not None]
    report.update(status='converged' if result.converged else 'not_converged',result=result.info(),
        seconds=time.monotonic()-began,sampled_spd_rows=len(finite),
        sampled_objective_z_increases=sum(b['objective_z']>a['objective_z']+1e-9 for a,b in zip(finite,finite[1:])),
        sampled_initial_objective=finite[0]['objective_z'] if finite else None,
        sampled_final_objective=finite[-1]['objective_z'] if finite else None,
        scope_note='Sampled loss need not be monotonic in ADMM; convergence requires residual/SPD/KKT checks')
    if result.converged:
        np.savez_compressed(args.output/'initial_graph_diagnostic.npz',Theta=result.Theta,S=s,Lambda=penalty,
            B=np.eye(ds.hidden_size),concept_ids=np.asarray(ds.concept_ids))
    write_json(args.output/'report.json',report)
    print('[diagnostic] '+report['status']+'; '+str(args.output),flush=True)


if __name__ == '__main__':
    main()
