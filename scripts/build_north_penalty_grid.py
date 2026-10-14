"""Fixed-B conditional MNGM penalty sweep; exploratory graphs, not full refits."""
import argparse,json,time,subprocess
from pathlib import Path
import numpy as np
from graph_mvp.weighted_glasso import WeightedGraphicalLasso
from graph_mvp.config import SolverConfig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--penalties',type=float,nargs='+',default=[.8,1.2,.4,.2])
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    data=np.load(args.source/'initial_graph.npz',allow_pickle=False)
    rows=[];start=time.monotonic()
    for penalty in args.penalties:
        began=time.monotonic()
        def progress(x):
            if x.get('iteration',0)%250==0 or x.get('iteration')==1:
                print(json.dumps({**x,'penalty':penalty,'seconds':time.monotonic()-start}),flush=True)
        result=WeightedGraphicalLasso(SolverConfig(max_iter=10000)).solve(
            data['S'],penalty,initial_theta=data['Theta'],progress_callback=progress)
        if not result.converged:raise RuntimeError(f'penalty {penalty} failed: {result.message}')
        theta=result.Theta;scale=np.sqrt(np.diag(theta));rho=-theta/np.outer(scale,scale);np.fill_diagonal(rho,0.)
        name=f'lambda_{penalty:g}.npz'
        np.savez_compressed(args.output/name,concept_ids=data['concept_ids'],S=data['S'],Theta=theta,
            partial_corr=rho,representation_precision=data['representation_precision'],
            Lambda=np.full_like(theta,penalty)-np.eye(len(theta))*penalty)
        edges=int((np.abs(rho[np.triu_indices(len(theta),1)])>1e-5).sum())
        row={'penalty':penalty,'file':name,'num_edges':edges,'density':edges/(len(theta)*(len(theta)-1)/2),
             'elapsed_seconds':time.monotonic()-began,**result.info()}
        rows.append(row);(args.output/'progress.json').write_text(json.dumps(rows,indent=2))
        print(json.dumps({'stage':'grid_graph_complete',**row}),flush=True)
    (args.output/'summary.json').write_text(json.dumps({'graphs':rows,'mode':'fixed_B_conditional_glasso',
        'test_split_touched':False,'source':str(args.source),'runtime_seconds':time.monotonic()-start,
        'git_sha':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip(),
        'warning':'conditional search graphs; full MNGM refinement required before claiming the original full alternating method'},indent=2))


if __name__=='__main__':main()
