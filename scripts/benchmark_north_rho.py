"""Strict fixed/adaptive-rho solver comparisons on the same private covariance."""
import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
from graph_mvp.weighted_glasso import WeightedGraphicalLasso
from graph_mvp.config import SolverConfig


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    data=np.load(args.source/'initial_graph.npz',allow_pickle=False);rows=[]
    # First try a finite adaptive prefix, then two recorded fixed alternatives.
    for name,rho,adaptive in [('adaptive',1.,True),('rho10',10.,False),('rho100',100.,False)]:
        start=time.monotonic()
        def progress(row):
            if row.get('iteration',0)%250==0 or row.get('iteration')==1:
                print(json.dumps({**row,'mode':name,'seconds':time.monotonic()-start}),flush=True)
        result=WeightedGraphicalLasso(SolverConfig(max_iter=10000,rho=rho)).solve(
            data['S'],data['Lambda']*1.05,initial_theta=data['Theta'],adaptive_rho=adaptive,progress_callback=progress)
        row={'mode':name,'rho_initial':rho,'adaptive_rho':adaptive,'elapsed_seconds':time.monotonic()-start,**result.info()}
        rows.append(row)
        if result.Theta is not None:np.savez_compressed(args.output/f'{name}.npz',Theta=result.Theta)
        (args.output/'progress.json').write_text(json.dumps(rows,indent=2))
        print(json.dumps({'stage':'solve_complete',**row}),flush=True)
    (args.output/'summary.json').write_text(json.dumps({'rows':rows,'test_split_touched':False,
        'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'reference':'Boyd et al. 2011 sec. 3.4.1; adaptation stops after iteration 1000'},indent=2))


if __name__=='__main__':main()
