"""Compare exact full-population float64 MNGM contractions on CPU and CUDA."""
import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from graph_mvp.config import Config
from graph_mvp.patient_repr import PatientMatrixDataset
from graph_mvp.estimators import MNGMEstimator,PATIENT_MATRIX_MODE


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.source/'run_manifest.json').read_text())
    cfg=Config.load(manifest['args']['config'])
    print(json.dumps({'stage':'materialize_and_rank_transform'}),flush=True)
    ds=PatientMatrixDataset(args.source/'cache')
    estimator=MNGMEstimator(ds.materialize(dtype=np.float32),PATIENT_MATRIX_MODE,cfg.mngm,covariance_device='cuda:0')
    tensor=estimator._covariance_tensor
    graph=np.load(args.source/'initial_graph.npz',allow_pickle=False)
    values={};rows=[]
    for mode in ['numpy','cuda_float64']:
        estimator._covariance_tensor=None if mode=='numpy' else tensor
        for name,method,arguments in [
            ('concept_covariance',estimator._concept_covariance,[graph['representation_precision']]),
            ('representation_covariance',estimator._representation_covariance,[graph['Theta']]),
            ('statistical_loss',estimator._statistical_loss,[graph['Theta'],graph['representation_precision']])]:
            torch.cuda.synchronize();start=time.monotonic();value=method(*arguments);torch.cuda.synchronize()
            row={'mode':mode,'operation':name,'elapsed_seconds':time.monotonic()-start}
            if mode=='numpy':values[name]=value
            else:
                reference=np.asarray(values[name]);actual=np.asarray(value)
                row.update(max_absolute_difference=float(np.max(np.abs(reference-actual))),
                    relative_norm_difference=float(np.linalg.norm(reference-actual)/max(np.linalg.norm(reference),1e-12)))
                np.testing.assert_allclose(actual,reference,rtol=1e-10,atol=1e-9)
            rows.append(row);print(json.dumps({'stage':'operation_complete',**row}),flush=True)
            (args.output/'progress.json').write_text(json.dumps(rows,indent=2))
    (args.output/'summary.json').write_text(json.dumps({'rows':rows,'shape':list(estimator.transformed.shape),
        'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'test_split_touched':False,'dtype':'float64','warning':'contraction benchmark, not full MNGM timing'},indent=2))


if __name__=='__main__':main()
