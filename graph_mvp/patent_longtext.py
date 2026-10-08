"""Same-patent, section-balanced original excerpts with an exact token budget."""
from __future__ import annotations
import csv, gzip, hashlib, heapq, html, json, re, time, urllib.request, zipfile
from pathlib import Path
from .joint_evidence_repr import _fitting_prefix, split_document, token_count

SNAPSHOT = '2024-12-31'
SOURCES = {
    'summary': ('15062198', 'g_brf_sum_text_{year}.tsv.zip', 'summary_text', .20),
    'claims': ('15062183', 'g_claims_{year}.tsv.zip', 'claim_text', .25),
    'description': ('15062212', 'g_detail_desc_text_{year}.tsv.zip', 'description_text', .55),
}
HEADINGS = {'summary':'Invention summary', 'claims':'Claims', 'description':'Detailed description'}

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def code_digest(path):
    """Ignore only Windows vs Linux newline conversion for source provenance."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()

def cache_ids(cache, limit=None):
    cache = Path(cache)
    meta = json.loads((cache/'metadata.json').read_text(encoding='utf-8'))
    if meta['representation_pipeline'].get('cache_build_complete') is not True:
        raise ValueError('Reference cache must be complete')
    integrity = json.loads((cache/'build_integrity.json').read_text(encoding='utf-8'))
    if digest(cache/'metadata.json') != integrity['metadata.json']:
        raise ValueError('Reference metadata checksum mismatch')
    ids = []
    for shard in meta['shards']:
        path = cache/shard['records']
        if digest(path) != integrity[path.name]:
            raise ValueError(f'Reference IDs checksum mismatch: {path.name}')
        records = json.loads(path.read_text(encoding='utf-8'))
        if len(records) != shard['n_samples']:
            raise ValueError('Reference shard record count mismatch')
        ids += [str(row['subject_id']) for row in records]
    if len(ids) != meta['n_samples'] or len(set(ids)) != len(ids):
        raise ValueError('Reference cache IDs duplicated or count incorrect')
    if limit is not None and len(ids) != limit:
        raise ValueError(f'Requested {limit} documents but reference cache has {len(ids)}')
    return ids, meta

def selected_training_rows(train, ids):
    wanted, found = set(ids), {}
    with gzip.open(train,'rt',encoding='utf-8-sig',newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'patent_id','patent_date','text'}.issubset(reader.fieldnames or []):
            raise ValueError('Training CSV missing patent ID/date/text')
        for row in reader:
            pid = row['patent_id'].strip()
            if pid in wanted:
                if pid in found:
                    raise ValueError(f'Duplicate selected patent: {pid}')
                if not row['text'].strip() or row.get('split','train') != 'train':
                    raise ValueError('Only nonempty training documents can enter graph fit')
                found[pid] = row
    if set(found) != wanted:
        raise ValueError(f'Selected cache IDs absent from train: {sorted(wanted-set(found))[:10]}')
    return [found[pid] for pid in ids]

def plain_text(text):
    text = re.sub(r'<(?:br\s*/?|/?p|/?div)\b[^>]*>', '\n', str(text), flags=re.I)
    text = html.unescape(re.sub(r'<[^>]+>', ' ', text))
    return re.sub(r'[ \t\r\f\v]+',' ',text).strip()

def spread_indices(count):
    """Deterministic broad coverage including the beginning and end, no labels."""
    if count < 1:
        return
    yield 0
    if count == 1:
        return
    yield count-1
    heap = [(-(count-2), 1, count-2)] if count > 2 else []
    while heap:
        _, start, end = heapq.heappop(heap)
        mid = (start+end)//2
        yield mid
        for lo, hi in ((start,mid-1),(mid+1,end)):
            if lo <= hi:
                heapq.heappush(heap, (-(hi-lo+1),lo,hi))

def allocate_budget(capacities, available):
    """Redistribute unused shares of short/missing sections to other sections."""
    allocation = {k:0 for k in SOURCES}
    left = max(0,int(available))
    while left:
        active = [k for k in SOURCES if allocation[k] < capacities.get(k,0)]
        if not active:
            break
        weight = sum(SOURCES[k][3] for k in active)
        original = left
        for k in active:
            share = max(1,int(original*SOURCES[k][3]/weight))
            take = min(left,share,capacities[k]-allocation[k])
            allocation[k] += take
            left -= take
    return allocation

def section_excerpts(text, tokenizer, budget, chunk_max_tokens=128):
    if not text or budget < 8:
        return '', []
    if token_count(tokenizer,text,special=False) <= budget:
        return text, [{'start':0,'end':len(text),'source_chunk_index':None}]
    chunks = split_document(text,tokenizer,chunk_max_tokens)
    selected = []
    for index in spread_indices(len(chunks)):
        unit = chunks[index]
        def fits(prefix):
            parts = sorted([*selected,{**unit,'text':prefix}],key=lambda x:x['index'])
            return token_count(tokenizer,'\n\n'.join(x['text'] for x in parts),special=False) <= budget
        prefix = unit['text'] if fits(unit['text']) else _fitting_prefix(unit['text'],fits)
        if prefix and token_count(tokenizer,prefix,special=False) >= 8:
            selected.append({**unit,'text':prefix,'end':unit['start']+len(prefix)})
        if selected and budget-token_count(tokenizer,'\n\n'.join(x['text'] for x in sorted(selected,key=lambda x:x['index'])),special=False) < 8:
            break
    selected.sort(key=lambda x:x['index'])
    return '\n\n'.join(x['text'] for x in selected), [
        {'start':x['start'],'end':x['end'],'source_chunk_index':x['index']} for x in selected]

def expand_document(base_text, sections, tokenizer, budget=4096, chunk_max_tokens=128):
    if budget < 128:
        raise ValueError('Candidate token budget must be >=128')
    base_tokens = token_count(tokenizer,base_text,special=False)
    if base_tokens >= budget:
        raise ValueError('Original title/abstract exhaust candidate budget; do not silently truncate them')
    sections = {k:plain_text(sections.get(k,'')) for k in SOURCES}
    raw_lengths = {k:token_count(tokenizer,t,special=False) if t else 0 for k,t in sections.items()}
    overhead = sum(token_count(tokenizer,'\n\n'+HEADINGS[k]+':\n',special=False)+4 for k,t in sections.items() if t)
    allocation = allocate_budget(raw_lengths,budget-base_tokens-overhead)
    text, audit = base_text, {}
    for key, source in sections.items():
        chosen, spans = section_excerpts(source,tokenizer,allocation[key],chunk_max_tokens)
        prefix = '\n\n'+HEADINGS[key]+':\n'
        # Verify the whole concatenation because BPE counts need not be additive.
        if chosen and token_count(tokenizer,text+prefix+chosen,special=False) > budget:
            chosen = _fitting_prefix(chosen,lambda x:token_count(tokenizer,text+prefix+x,special=False)<=budget)
            # Clip original spans in parallel with the selected source-order text.
            clipped, remaining = [],len(chosen)
            for span in spans:
                take = min(remaining,span['end']-span['start'])
                if take > 0:
                    clipped.append({**span,'end':span['start']+take})
                remaining -= take+2
                if remaining <= 0:
                    break
            spans = clipped
        if chosen:
            text += prefix+chosen
        audit[key] = {'status':'available' if source else 'missing',
            'source_chars':len(source),'source_tokens':raw_lengths[key],
            'source_sha256':hashlib.sha256(source.encode()).hexdigest() if source else None,
            'allocated_tokens':allocation[key],'selected_tokens':token_count(tokenizer,chosen,special=False) if chosen else 0,
            'selected_source_spans':spans,'source_shortened':bool(source and chosen != source)}
    total = token_count(tokenizer,text,special=False)
    if total > budget:
        raise ValueError('Candidate document exceeds exact token budget')
    return text, {'base_tokens':base_tokens,'candidate_tokens':total,'budget':budget,
        'expanded':text!=base_text,'selection':'section_balanced_spread_original_excerpts',
        'selection_is_not_relevance_label':True,'sections':audit}

def request_json(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'PatentsViewGraph/0.6.3'}),timeout=60) as response:
                return json.load(response)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2*(attempt+1))

def ensure_archive(directory, record, name, file_info):
    """Resume incomplete transfers; quarantine corrupt partials, never publish them."""
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    target = directory/name
    algorithm, expected = file_info['checksum'].split(':',1)
    size = int(file_info['size'])
    if target.exists():
        if target.stat().st_size != size or digest(target,algorithm) != expected:
            raise ValueError(f'Existing archive checksum mismatch: {target}')
        return target
    part = directory/(name+'.part')
    url = f'https://zenodo.org/records/{record}/files/{name}?download=1'
    def quarantine(reason):
        saved = part.with_name(part.name+f'.invalid.{time.time_ns()}')
        part.rename(saved)
        print(f'[download] {reason}; preserved as {saved}; restarting from byte 0',flush=True)

    last_error = None
    for attempt in range(3):
        try:
            offset = part.stat().st_size if part.exists() else 0
            if offset > size:
                quarantine(f'Partial size {offset} exceeds expected {size}')
                offset = 0
            while offset < size:
                headers = {'User-Agent':'PatentsViewGraph/0.6.3',
                    'Accept-Encoding':'identity','Cache-Control':'no-cache'}
                if offset:
                    headers['Range'] = f'bytes={offset}-'
                print(f'[download] {name}: attempt {attempt+1}/3, bytes {offset}/{size}',flush=True)
                with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=120) as response:
                    if response.status == 206:
                        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                        if not match:
                            raise ValueError('Invalid or missing Content-Range')
                        start,end,total = map(int,match.groups())
                        if start != offset or total != size or not start <= end < total:
                            raise ValueError(f'Unexpected Content-Range: {response.headers.get("Content-Range")}; '
                                f'expected offset {offset}, total {size}')
                        mode, response_end = 'ab', end+1
                    elif response.status == 200:
                        # A server may ignore Range; restart rather than append a full file.
                        mode, offset, response_end = 'wb', 0, size
                    else:
                        raise ValueError(f'Unexpected download status {response.status}')
                    length = response.headers.get('Content-Length')
                    if length is not None and int(length) != response_end-offset:
                        raise ValueError(f'Unexpected Content-Length {length}; expected {response_end-offset}')
                    with part.open(mode) as stream:
                        while block := response.read(1024*1024):
                            if offset+len(block) > response_end:
                                raise ValueError('Response body exceeds declared archive/range size')
                            stream.write(block)
                            offset += len(block)
                    if offset != response_end:
                        raise ValueError(f'Incomplete response: received through byte {offset}, expected {response_end}')
                # A valid 206 may cover only part of the requested suffix. Keep resuming
                # successful ranges without spending the transport failure retry budget.
            actual = digest(part,algorithm)
            if actual != expected:
                quarantine(f'{algorithm} mismatch: expected {expected}, received {actual} ({size} bytes)')
                raise ValueError(f'Archive {algorithm} mismatch: expected {expected}, received {actual}')
            part.rename(target)
            print(f'[download] verified {name}: {size} bytes, {algorithm}:{expected}',flush=True)
            return target
        except Exception as exc:
            last_error = exc
            received = part.stat().st_size if part.exists() else 0
            print(f'[download] {name}: attempt {attempt+1}/3 failed, partial bytes {received}/{size}: {exc}',flush=True)
            if attempt < 2:
                time.sleep(2*(attempt+1))
    raise ValueError(f'Download failed after 3 attempts: {part}; '
        f'expected {size} bytes, {algorithm}:{expected}; last error: {last_error}') from last_error

def load_sections(paths, patent_years):
    """Join granted text only, by exact grant ID and grant year, preserving order."""
    csv.field_size_limit(128*1024*1024)
    result = {pid:{k:[] for k in SOURCES} for pid in patent_years}
    for section, year, archive_path in paths:
        field = SOURCES[section][2]
        matched = 0
        with zipfile.ZipFile(archive_path) as archive:
            members = [x for x in archive.namelist() if x.endswith('.tsv') and not Path(x).name.startswith('._')]
            if len(members) != 1:
                raise ValueError(f'Expected one TSV in {archive_path}')
            import io
            with archive.open(members[0]) as raw, io.TextIOWrapper(raw,encoding='utf-8-sig',newline='') as stream:
                reader = csv.DictReader(stream,delimiter='\t')
                if not {'patent_id',field}.issubset(reader.fieldnames or []):
                    raise ValueError(f'Unexpected text-table columns in {archive_path}: {reader.fieldnames}')
                for number,row in enumerate(reader):
                    pid = row['patent_id'].strip()
                    if pid in result and patent_years[pid] == year:
                        value = plain_text(row.get(field,''))
                        if value:
                            order = int(row['claim_sequence']) if section == 'claims' else number
                            result[pid][section].append((order,value))
                            matched += 1
        print(f'[join] {section} {year}: matched {matched} source rows',flush=True)
    return {pid:{k:'\n\n'.join(v for _,v in sorted(items)) for k,items in fields.items()} for pid,fields in result.items()}
