import csv, gzip, io, json, zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from graph_mvp.patent_longtext import (allocate_budget,cache_ids,code_digest,digest,expand_document,
    load_sections,plain_text,section_excerpts,spread_indices,write_json)
from graph_mvp.diagnostic_sampling import SyntheticJointEncoder
from graph_mvp.joint_evidence_repr import JointEvidenceMatrixBuilder,token_count
from graph_mvp.patient_repr import (ConceptVocabulary,PatientMatrixCacheWriter,PatientMatrixDataset,save_concept_prototypes)
from scripts.diagnose_joint_controls import recover_evidence

def test_budget_source_spans_cover_multiple_sections_and_late_text():
    tokenizer = SyntheticJointEncoder.Tokenizer()
    sections = {k:'\n'.join(f'{k} original sentence {i:03d} with network information.' for i in range(150))
        for k in ('summary','claims','description')}
    base = 'Title: Network\n\nAbstract: A routing mechanism.'
    expanded,audit = expand_document(base,sections,tokenizer,budget=4096)
    assert expanded.startswith(base) and token_count(tokenizer,expanded,special=False)<=4096
    assert audit['expanded'] and audit['selection_is_not_relevance_label']
    assert set(audit['sections'])==set(sections)
    for section,report in audit['sections'].items():
        assert report['selected_tokens']>0 and report['source_shortened']
        source = plain_text(sections[section])
        fragments = [source[x['start']:x['end']] for x in report['selected_source_spans']]
        assert all(fragment in expanded for fragment in fragments)
        assert max(x['start'] for x in report['selected_source_spans'])>len(source)*.9
    assert '149' in expanded

def test_missing_sections_preserve_base_and_redistribute_budget():
    tokenizer = SyntheticJointEncoder.Tokenizer()
    base = 'Title: Routing. Abstract: Packets.'
    text,audit = expand_document(base,{},tokenizer)
    assert text==base and not audit['expanded']
    assert all(v['status']=='missing' for v in audit['sections'].values())
    allocation = allocate_budget({'summary':10,'claims':0,'description':10000},1000)
    assert allocation=={'summary':10,'claims':0,'description':990}
    with pytest.raises(ValueError,match='exhaust'):
        expand_document('x'*4096,{},tokenizer)
    with pytest.raises(ValueError,match='>=128'):
        expand_document(base,{},tokenizer,budget=64)

def test_spread_indices_are_unique_complete_and_source_order_reconstructed():
    for n in range(60):
        indices = list(spread_indices(n))
        assert sorted(indices)==list(range(n))
    text = 'First original sentence. Second original sentence. Last original sentence.'
    kept,spans = section_excerpts(text,SyntheticJointEncoder.Tokenizer(),55,32)
    assert kept=='\n\n'.join(text[x['start']:x['end']] for x in spans)
    assert len(kept)<=55

def test_zip_join_uses_grant_id_and_year_and_preserves_claim_order(tmp_path):
    files=[]
    for section,field,rows in (
        ('summary','summary_text',[['100','<p>Network &amp; packets.</p>'],['200','Unrelated.']]),
        ('description','description_text',[['100','Description.'],['300','Wrong year.']]),
        ('claims','claim_text',[['100','2','Dependent claim.'],['100','0','Independent claim.']])):
        buffer=io.StringIO()
        header=['patent_id','claim_sequence',field] if section=='claims' else ['patent_id',field]
        writer=csv.writer(buffer,delimiter='\t'); writer.writerow(header);writer.writerows(rows)
        path=tmp_path/(section+'.zip')
        with zipfile.ZipFile(path,'w') as archive:
            archive.writestr(section+'.tsv',buffer.getvalue())
        files.append((section,2005,path))
    result=load_sections(files,{'100':2005,'300':2006})
    assert result['100']['summary']=='Network & packets.'
    assert result['100']['claims']=='Independent claim.\n\nDependent claim.'
    assert result['300']['description']=='' and '200' not in result

def make_reference(tmp_path,n=6):
    encoder=SyntheticJointEncoder(1024)
    vocab=ConceptVocabulary(tuple(f'c{i}' for i in range(5)),tuple(f'concept {i}' for i in range(5)))
    proto=encoder.encode_pooled(vocab.concept_texts).numpy()
    prototypes=tmp_path/'prototypes.npz'
    save_concept_prototypes(prototypes,proto,vocab,encoder_id='test-model')
    projection=tmp_path/'projection.npz'
    np.savez(projection,mean=np.zeros(32,dtype=np.float32),matrix=np.eye(32,dtype=np.float32)[:4])
    builder=JointEvidenceMatrixBuilder(encoder,proto,vocab.concept_texts,top_k=6,representation_dim=4,
        projection_mean=np.zeros(32),projection_matrix=np.eye(32)[:4],encode_batch_size=2)
    cache=tmp_path/'old/cache'
    pipeline=builder.projection_metadata();pipeline['cache_build_complete']=True
    writer=PatientMatrixCacheWriter(cache,vocab.concept_ids,4,dtype='float32',encoder_id='test-model',
        encoder_hidden_size=32,projection_sha256=pipeline['projection_sha256'],representation_pipeline=pipeline)
    np.savez(cache/'projection.npz',mean=np.zeros(32,dtype=np.float32),matrix=np.eye(32,dtype=np.float32)[:4])
    source=[f'Original routing case {i}. Security question {i%3}. Queues and channels {i}.' for i in range(n)]
    encoded=[builder.encode_document(text) for text in source]
    matrices=[x[0].numpy()[0] for x in encoded]
    with (cache/'evidence.jsonl').open('w',encoding='utf-8') as stream:
        for i,(text,(_,_,traces)) in enumerate(zip(source,encoded)):
            stream.write(json.dumps({'row_index':i,'subject_id':str(i),'source_text':text,'concept_evidence':traces})+'\n')
    writer.write_shard(np.stack(matrices),[str(i) for i in range(n)],[str(i) for i in range(n)])
    writer.close()
    write_json(cache/'build_integrity.json',{p.name:digest(p) for p in cache.iterdir() if p.is_file()})
    return cache,source,prototypes,projection,builder

def test_reference_ids_are_verified_and_corruption_fails(tmp_path):
    cache,*_=make_reference(tmp_path)
    ids,_=cache_ids(cache,6)
    assert ids==[str(i) for i in range(6)]
    with pytest.raises(ValueError,match='Requested'):
        cache_ids(cache,7)
    (cache/'shard_00000.records.json').write_text('[]')
    with pytest.raises(ValueError,match='checksum'):
        cache_ids(cache)

@pytest.mark.parametrize('solver_budget',[None,1200])
def test_full_cache_build_diagnostics_and_compact_evidence_match_direct_F(tmp_path,monkeypatch,solver_budget):
    import scripts.build_patent_longtext_matrices as build
    cache,source,prototypes,projection,builder=make_reference(tmp_path)
    monkeypatch.setattr(build,'Qwen3EmbeddingEncoder',lambda *a,**kw:SyntheticJointEncoder(kw['max_length']))
    prepared=tmp_path/'prepared';prepared.mkdir()
    train=prepared/'train.csv.gz'
    texts=[t+' '+'. '.join(f'Expanded case {i} detail {j}' for j in range(20))+'.' for i,t in enumerate(source)]
    with gzip.open(train,'wt',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['patent_id','text']);writer.writeheader()
        writer.writerows({'patent_id':str(i),'text':t} for i,t in enumerate(texts))
    write_json(prepared/'preparation_report.json',{'candidate_token_budget':4096,'patent_ids':[str(i) for i in range(6)]})
    output=tmp_path/'new-cache'
    build.main(['--train',str(train),'--prototypes',str(prototypes),'--projection',str(projection),
        '--output',str(output),'--reference-cache',str(cache),'--model','test-model','--documents','6'])
    ds=PatientMatrixDataset(output)
    x=ds.materialize()
    assert x.shape==(6,4,5)
    for i,text in enumerate(texts):
        np.testing.assert_allclose(x[i],builder.encode_document(text)[0].numpy()[0],rtol=1e-6,atol=1e-6)
    records=[json.loads(line) for line in (output/'evidence.jsonl').read_text().splitlines()]
    for record in records:
        for trace in record['concept_evidence']:
            assert recover_evidence(record,trace)
            assert trace['input_tokens']<=1024
    report=json.loads((output/'representation_report.json').read_text())
    assert report['new']['native_vs_prototype']['count']==30
    assert report['old']['projected_vs_prototype']['count']==30
    assert report['new']['projected_vs_concept_only']['count']==30
    assert ds.metadata['representation_pipeline']['residual_subtraction'] is False
    assert ds.metadata['representation_pipeline']['prototype_addition'] is False
    assert ds.metadata['representation_pipeline']['per_concept_l2_after_projection'] is False
    integrity=json.loads((output/'build_integrity.json').read_text())
    assert all(digest(output/name)==expected for name,expected in integrity.items())
    # Exercise real weighted Glasso and alternating MNGM on the synthetic cache.
    from graph_mvp.config import Config
    from graph_mvp.estimators import MNGMEstimator,PATIENT_MATRIX_MODE
    from graph_mvp.weighted_glasso import penalty_matrix
    from scripts.fit_patent_longtext_graph import main as fit,covariance_report
    config=tmp_path/'config.json'
    write_json(config,{'solver':{'max_iter':1000,'kkt_tol':1e-4},
        'mngm':{'max_iter':12,'tol':1e-3,'representation_penalty':.4,'transform':'rank_gaussian'}})
    cfg=Config.load(config)
    old_ds=PatientMatrixDataset(cache)
    old_estimator=MNGMEstimator(old_ds.materialize(),PATIENT_MATRIX_MODE,cfg.mngm,cfg.solver)
    penalty=penalty_matrix(5,.8)
    initial=old_estimator.solve_initial(penalty)
    assert initial.converged
    training=cache.parent/'training';(training/'mngm').mkdir(parents=True)
    np.savez(training/'initial_graph.npz',concept_ids=np.array(old_ds.concept_ids),Lambda=penalty,Theta=initial.Theta)
    write_json(training/'mngm/initial_covariance_diagnostics.json',covariance_report(initial.S))
    write_json(cache.parent/'experiment_manifest.json',{'input_hashes':{'config':digest(config)}})
    graph_out=tmp_path/'new-graph'
    fit_args=['--cache',str(output),'--reference-experiment',str(cache.parent),'--config',str(config),'--output',str(graph_out)]
    if solver_budget is not None:
        fit_args+=['--solver-max-iter',str(solver_budget)]
    fit(fit_args)
    result=json.loads((graph_out/'summary.json').read_text())
    assert result['status']=='complete' and result['grpo_updates']==0
    assert result['task_training_performed'] is False
    assert (graph_out/'initial_graph.npz').exists() and (graph_out/'fitted_graph.npz').exists()
    paired=json.loads((graph_out/'evidence_comparison.json').read_text())
    assert len(paired['paired_documents'])==6
    manifest=json.loads((graph_out/'run_manifest.json').read_text())
    assert manifest['status']=='complete' and manifest['documents']==6
    assert manifest['original_solver']['max_iter']==1000
    assert manifest['effective_solver']['max_iter']==(solver_budget or 1000)
    assert all(manifest['original_solver'][k]==manifest['effective_solver'][k]
        for k in manifest['original_solver'] if k!='max_iter')
    assert digest(config)==manifest['config_sha256']
    assert manifest['gpu_encoding_performed'] is False
    progress=[json.loads(line) for line in (graph_out/'progress.jsonl').read_text().splitlines()]
    assert any(row['stage']=='mngm_outer' for row in progress)
    # Budget overrides cannot bypass original-config or cached-input checks.
    blocked=tmp_path/'blocked-graph'
    fit_args[fit_args.index('--output')+1]=str(blocked)
    with pytest.raises(SystemExit):
        fit(fit_args+['--solver-max-iter','999'])
    assert not blocked.exists()
    (output/'shard_00000.npy').write_bytes(b'corrupt cached matrix')
    with pytest.raises(ValueError,match='checksum mismatch'):
        fit(fit_args)
    assert not blocked.exists()

def test_graph_only_command_plan_has_no_second_stage_or_pilot(tmp_path):
    from scripts.run_patent_longtext_graph import commands
    args=SimpleNamespace(output=Path('outputs/new'),data_dir=Path('data'),reference_experiment=Path('old'),
        raw_dir=Path('raw'),config=Path('cfg'),projection_file=Path('proj'),qwen_model='qwen',documents=2048,
        candidate_tokens=4096,chunk_max_tokens=128,max_length=1024,top_k=6,encode_batch_size=8)
    stages=commands(args,tmp_path)[-1]
    assert [name for name,_ in stages]==['prepare','build','graph']
    assert stages[1][1][stages[1][1].index('--documents')+1]=='2048'
    text=' '.join(' '.join(command) for _,command in stages)
    assert 'GLM' not in text and 'run_patent_graph_rl' not in text
    assert 'warmup' not in text and 'adapt' not in text and 'GRPO' not in text

def test_full_entry_validates_reference_and_only_plans_three_stages(tmp_path,monkeypatch):
    import scripts.run_patent_longtext_graph as runner
    cache,source,prototypes,projection,_=make_reference(tmp_path)
    train=tmp_path/'train.csv.gz'
    with gzip.open(train,'wt',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['patent_id','patent_date','text','split']);writer.writeheader()
        writer.writerows({'patent_id':str(i),'patent_date':'2005-01-01','text':t,'split':'train'} for i,t in enumerate(source))
    # Copy inputs to the entry point's expected data directory.
    (tmp_path/'concept_prototypes.npz').write_bytes(prototypes.read_bytes())
    config=tmp_path/'config.json';config.write_text('{}')
    manifest={'status':'complete','input_hashes':{'train':digest(train),'prototypes':digest(prototypes),
        'config':digest(config),'projection':digest(projection)}}
    write_json(cache.parent/'experiment_manifest.json',manifest)
    for name in ('graph_mvp/patent_longtext.py','graph_mvp/joint_evidence_repr.py',
        'graph_mvp/patient_repr.py','graph_mvp/estimators.py','graph_mvp/weighted_glasso.py',
        'scripts/prepare_patent_longtext.py','scripts/build_patent_longtext_matrices.py',
        'scripts/fit_patent_longtext_graph.py','scripts/run_patent_longtext_graph.py'):
        target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((runner.ROOT/name).read_bytes())
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    args=['--data-dir',str(tmp_path),'--reference-experiment',str(cache.parent),'--projection-file',str(projection),
        '--config',str(config),'--qwen-model','test-model','--documents','6','--output',str(tmp_path/'plan'),'--plan-only']
    runner.main(args)
    report=json.loads((tmp_path/'plan/experiment_manifest.json').read_text())
    assert report['status']=='plan_only' and list(report['commands'])==['prepare','build','graph']
    assert report['planned_grpo_updates']==0 and not (tmp_path/'plan/cache').exists()
    assert report['graph_mvp_version']=='0.6.3' and report['git_sha'] is None
    assert report['source_files_sha256']['graph_mvp/patent_longtext.py']==code_digest(tmp_path/'graph_mvp/patent_longtext.py')
    # Corrupt reference input before any expensive download or model inference.
    config.write_text('{"solver":{}}')
    args[args.index('--output')+1]=str(tmp_path/'blocked-plan')
    with pytest.raises(SystemExit):
        runner.main(args)
    assert not (tmp_path/'blocked-plan').exists()

def test_source_fingerprint_ignores_only_cross_platform_newlines(tmp_path):
    path=tmp_path/'source.py'
    path.write_bytes(b'x = 1\r\n')
    checksum=code_digest(path)
    path.write_bytes(b'x = 1\n')
    assert code_digest(path)==checksum
    path.write_bytes(b'x = 2\n')
    assert code_digest(path)!=checksum
