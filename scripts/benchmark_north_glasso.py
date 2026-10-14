"""Compare warm starts on the original full-size stored problem."""
import argparse,json,time,subprocess
from pathlib import Path
import numpy as np
from graph_mvp.weighted_glasso import WeightedGraphicalLasso,objective,penalty_matrix
from graph_mvp.config import SolverConfig


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    arrays=np.load(args.source/'initial_graph.npz',allow_pickle=False)
    s,lam,theta=arrays['S'],arrays['Lambda'],arrays['Theta']
    rows=[];matrices={}
    for case,factor in [('unchanged',1.),('penalty_plus_5pct',1.05)]:
        for mode in ['reconstructed','legacy']:
            started=time.monotonic()
            def progress(x):
                if x.get('iteration',0)%250==0 or x.get('iteration')==1:
                    print(json.dumps({**x,'case':case,'mode':mode,'seconds':time.monotonic()-started}),flush=True)
            result=WeightedGraphicalLasso(SolverConfig(max_iter=10000)).solve(
                s,lam*factor,initial_theta=theta,reconstruct_dual=mode=='reconstructed',progress_callback=progress)
            row={'case':case,'mode':mode,'elapsed_seconds':time.monotonic()-started,**result.info()}
            rows.append(row)
            if result.Theta is not None:matrices[(case,mode)]=result.Theta
            (args.output/'progress.json').write_text(json.dumps(rows,indent=2))
            print(json.dumps({'stage':'solve_complete',**row}),flush=True)
            if not result.converged:raise RuntimeError('benchmark solve failed')
        left,right=matrices[(case,'reconstructed')],matrices[(case,'legacy')]
        comparison={'case':case,'theta_relative_error':float(np.linalg.norm(left-right)/np.linalg.norm(right)),
                    'objective_difference':objective(s,lam*factor,left)-objective(s,lam*factor,right)}
        print(json.dumps({'stage':'comparison',**comparison}),flush=True)
        rows.append(comparison)
    (args.output/'summary.json').write_text(json.dumps({'rows':rows,'dimension':len(theta),
        'git_sha':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()},indent=2))


if __name__=='__main__':main()
