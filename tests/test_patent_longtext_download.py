import hashlib
import io

import pytest

from graph_mvp import patent_longtext as downloads


PAYLOAD = b'Original patent archive bytes, verified by source metadata.'
NAME = 'g_detail_desc_text_2005.tsv.zip'
INFO = {'size': len(PAYLOAD), 'checksum': 'md5:'+hashlib.md5(PAYLOAD).hexdigest()}


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}


def install_transport(monkeypatch, handler):
    calls = []

    def open_url(request, timeout):
        assert timeout == 120
        calls.append(dict(request.header_items()))
        return handler(calls[-1])

    monkeypatch.setattr(downloads.urllib.request, 'urlopen', open_url)
    monkeypatch.setattr(downloads.time, 'sleep', lambda seconds: None)
    return calls


def suffix_response(offset):
    return Response(PAYLOAD[offset:], 206, {
        'Content-Range': f'bytes {offset}-{len(PAYLOAD)-1}/{len(PAYLOAD)}',
        'Content-Length': str(len(PAYLOAD)-offset)})


def test_existing_short_partial_is_resumed_and_verified(tmp_path, monkeypatch):
    (tmp_path/(NAME+'.part')).write_bytes(PAYLOAD[:13])
    calls = install_transport(monkeypatch, lambda headers: suffix_response(13))
    target = downloads.ensure_archive(tmp_path, '15062212', NAME, INFO)
    assert calls[0]['Range'] == 'bytes=13-'
    assert calls[0]['Accept-encoding'] == 'identity'
    assert target.read_bytes() == PAYLOAD
    assert not (tmp_path/(NAME+'.part')).exists()
    assert not list(tmp_path.glob('*.invalid.*'))


def test_short_eof_keeps_received_bytes_and_retries_suffix(tmp_path, monkeypatch):
    def handler(headers):
        if 'Range' not in headers:
            return Response(PAYLOAD[:11])  # No exception at EOF, but incomplete.
        assert headers['Range'] == 'bytes=11-'
        return suffix_response(11)

    calls = install_transport(monkeypatch, handler)
    target = downloads.ensure_archive(tmp_path, '15062212', NAME, INFO)
    assert len(calls) == 2 and target.read_bytes() == PAYLOAD


def test_interrupted_read_retains_written_prefix(tmp_path, monkeypatch):
    class InterruptedResponse(Response):
        def read(self, size):
            if self.tell():
                raise TimeoutError('connection interrupted')
            return super().read(9)

    def handler(headers):
        if 'Range' not in headers:
            return InterruptedResponse(PAYLOAD)
        assert headers['Range'] == 'bytes=9-'
        return suffix_response(9)

    calls = install_transport(monkeypatch, handler)
    assert downloads.ensure_archive(tmp_path, '15062212', NAME, INFO).read_bytes() == PAYLOAD
    assert len(calls) == 2


def test_more_than_three_successful_partial_ranges_complete(tmp_path, monkeypatch):
    def handler(headers):
        start = int(headers.get('Range', 'bytes=0-')[6:-1])
        end = min(start+6, len(PAYLOAD))-1
        return Response(PAYLOAD[start:end+1], 206, {
            'Content-Range': f'bytes {start}-{end}/{len(PAYLOAD)}',
            'Content-Length': str(end-start+1)})

    calls = install_transport(monkeypatch, handler)
    assert downloads.ensure_archive(tmp_path, '15062212', NAME, INFO).read_bytes() == PAYLOAD
    assert len(calls) > 3


def test_ignored_range_restarts_full_body_without_appending(tmp_path, monkeypatch):
    (tmp_path/(NAME+'.part')).write_bytes(PAYLOAD[:7])
    calls = install_transport(monkeypatch, lambda headers: Response(PAYLOAD, headers={'Content-Length':str(len(PAYLOAD))}))
    assert downloads.ensure_archive(tmp_path, '15062212', NAME, INFO).read_bytes() == PAYLOAD
    assert len(calls) == 1 and calls[0]['Range'] == 'bytes=7-'


@pytest.mark.parametrize('corrupt', [b'x'*len(PAYLOAD), b'x'*(len(PAYLOAD)+2)])
def test_corrupt_complete_or_oversize_partial_is_preserved_and_replaced(tmp_path, monkeypatch, corrupt):
    (tmp_path/(NAME+'.part')).write_bytes(corrupt)
    calls = install_transport(monkeypatch, lambda headers: Response(PAYLOAD))
    assert downloads.ensure_archive(tmp_path, '15062212', NAME, INFO).read_bytes() == PAYLOAD
    saved = list(tmp_path.glob(NAME+'.part.invalid.*'))
    assert len(saved) == 1 and saved[0].read_bytes() == corrupt
    assert len(calls) == 1 and 'Range' not in calls[0]


@pytest.mark.parametrize('headers', [
    {'Content-Range':f'bytes 0-{len(PAYLOAD)-1}/{len(PAYLOAD)}'},
    {'Content-Range':f'bytes 7-{len(PAYLOAD)-1}/{len(PAYLOAD)+1}'},
    {'Content-Range':'bytes 7-6/58'},
    {},
    {'Content-Range':f'bytes 7-{len(PAYLOAD)-1}/{len(PAYLOAD)}','Content-Length':'1'},
])
def test_invalid_range_headers_do_not_change_existing_prefix(tmp_path, monkeypatch, headers):
    part = tmp_path/(NAME+'.part')
    part.write_bytes(PAYLOAD[:7])
    calls = install_transport(monkeypatch, lambda request: Response(PAYLOAD[7:], 206, headers))
    with pytest.raises(ValueError, match='failed after 3 attempts'):
        downloads.ensure_archive(tmp_path, '15062212', NAME, INFO)
    assert len(calls) == 3 and part.read_bytes() == PAYLOAD[:7]
    assert not (tmp_path/NAME).exists()


def test_repeated_bad_checksums_are_never_published(tmp_path, monkeypatch):
    calls = install_transport(monkeypatch, lambda headers: Response(b'x'*len(PAYLOAD)))
    with pytest.raises(ValueError, match='md5 mismatch'):
        downloads.ensure_archive(tmp_path, '15062212', NAME, INFO)
    assert len(calls) == 3 and not (tmp_path/NAME).exists()
    assert len(list(tmp_path.glob(NAME+'.part.invalid.*'))) == 3


def test_stalled_transfer_is_bounded_and_reports_actual_size(tmp_path, monkeypatch):
    calls = install_transport(monkeypatch, lambda headers: Response(b''))
    with pytest.raises(ValueError, match=f'expected {len(PAYLOAD)} bytes'):
        downloads.ensure_archive(tmp_path, '15062212', NAME, INFO)
    assert len(calls) == 3 and not (tmp_path/NAME).exists()


def test_verified_archive_is_reused_and_bad_existing_target_is_preserved(tmp_path, monkeypatch):
    target = tmp_path/NAME
    target.write_bytes(PAYLOAD)
    calls = install_transport(monkeypatch, lambda headers: pytest.fail('Must not download'))
    assert downloads.ensure_archive(tmp_path, '15062212', NAME, INFO) == target
    target.write_bytes(b'bad complete target')
    with pytest.raises(ValueError, match='Existing archive checksum mismatch'):
        downloads.ensure_archive(tmp_path, '15062212', NAME, INFO)
    assert target.read_bytes() == b'bad complete target' and not calls
