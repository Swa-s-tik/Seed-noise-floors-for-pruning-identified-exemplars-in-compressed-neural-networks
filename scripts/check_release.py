"""Release check: exit 1 if any tracked file is larger than 90 MB or contains a local path, an e-mail address, a
credential-shaped string, an internal label or a term from the release deny-list.

The deny-list is stored as truncated SHA-256 digests of lower-cased words and word pairs, so this file does not
spell the terms it searches for. Exceptions:
  - the hashed pre-registration files and the frozen checkpoint inventory must stay byte-identical, so they are checked
    for e-mail addresses and credentials only;
  - verbatim third-party files (licence texts, source excerpts, the airbench94 script) keep their authors' wording,
    including contact addresses, so they are checked for local paths and credentials only;
  - the PyTorch identifiers num_workers and get_worker_info are not reported.
Binary files are checked for absolute paths, lower-case e-mail addresses, credentials and deny-listed words of five or
more letters (looser patterns match random bytes in compressed data).

Usage: python3 scripts/check_release.py
"""
import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = 'scripts/check_release.py'
MAX_BYTES = 90 * 2 ** 20
FROZEN = {'INVENTORY.json', 'preregistration/PREREG.md', *(f'preregistration/PREREG-ADDENDUM-{i}.md' for i in range(1, 5))}
VENDORED = ('provenance/licences/', 'provenance/tf_source/', 'prior-gate/g13/code/', 'cifar/airbench94.py',
            'cifar/LICENSE.airbench')
SKIP_DIRS = {'.git', '.venv', 'data', '__pycache__'}  # only used outside a git checkout

EMAIL = re.compile(rb'[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}')
BINEMAIL = re.compile(rb'(?<![A-Za-z0-9._%+-])[a-z0-9._%+-]{2,}@[a-z0-9-]{2,}(?:\.[a-z0-9-]{2,})*\.[a-z]{2,6}(?![A-Za-z0-9])')  # binary files
NOREPLY = b'@users.noreply.github.com'
SECRET = re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}'
                    rb'|xox[abprs]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----'
                    rb'|(?i:(?:api[_-]?key|secret|passw(?:or)?d|token)\s*[:=]\s*[\'"][^\'"\s]{8,})')
PATH = re.compile(rb'/home/|/Users/|(?<![\w.])~/\w')  # absolute and home-relative paths
BINPATH = re.compile(rb'/home/|/Users/')  # binary files: absolute paths only (short patterns match random bytes)
LABEL = re.compile(rb'\b[PG][1-5]\b|\.gpu\.(?:p3|lock)\b')
WORDS = {  # lower-cased words and word pairs
    'c857d09db23e6822', 'c70eca6b0f88f44d', '57de4cf40144bdf7', '5d72436256ada538', 'd4f0bc5a29de06b5',
    '8c70b25cbbbe8e79', '87eba76e7f316453', 'ed574aa71eb87d6c', '868727054248980e', '11a5c7af5e18a14b',
    'ba6301dad0900f40', 'c4d58602369ee394', '0e417ba3e146104c', 'e12ce8285efc67c6',
    'c9ad8f2cc1294afa', '134e361432daa42f', 'cf7883bcd2ce7ddb', '57671672b1c9254d', 'f4d00bc46e4f40e8',
    '220eac825ce71ff0',
}
API = re.compile(rb'\b(?:num_workers|get_worker_info)\b')  # PyTorch identifiers
CASED = {'a628d0e473bfb2d5', 'b1762f76a9b4b006', '182e499256bea391', 'a9e0ba49b003e1cc'}  # exact-case words


def digest(b):
    return hashlib.sha256(b).hexdigest()[:16]


def tracked():
    try:
        out = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT, capture_output=True, check=True).stdout
        return sorted(p.decode() for p in out.split(b'\0') if p)
    except (OSError, subprocess.CalledProcessError):  # not a git checkout (e.g. an archive download)
        return sorted(str(p.relative_to(ROOT)) for p in ROOT.rglob('*')
                      if p.is_file() and not set(p.relative_to(ROOT).parts) & SKIP_DIRS)


def word_hits(data, binary):
    toks = re.findall(rb'[A-Za-z0-9]+', API.sub(b' ', data))
    hits = {t.decode() for t in toks if (not binary or len(t) >= 5) and
            (digest(t.lower()) in WORDS or digest(t) in CASED)}
    if not binary:
        low = [t.lower() for t in toks]
        hits |= {f'{a.decode()} {b.decode()}' for a, b in zip(low, low[1:]) if digest(a + b' ' + b) in WORDS}
    return hits


def check(rel):
    p = ROOT / rel
    if not p.is_file():  # e.g. a tracked file deleted in the working tree
        return []
    if p.stat().st_size > MAX_BYTES:
        return [f'{rel}: larger than 90 MB']
    data = p.read_bytes()
    try:
        data.decode('utf-8')
        binary = b'\0' in data
    except UnicodeDecodeError:
        binary = True
    frozen, hits = rel in FROZEN, []
    lines = [data] if binary else data.splitlines()
    for i, line in enumerate(lines, 1):
        where = rel if binary else f'{rel}:{i}'
        vendored = rel.startswith(VENDORED)
        if not vendored:
            hits += [f'{where}: e-mail address' for m in (BINEMAIL if binary else EMAIL).finditer(line)
                     if not m.group().endswith(NOREPLY)]
        if SECRET.search(line):
            hits.append(f'{where}: credential-shaped string')
        if frozen:
            continue
        if vendored:
            hits += [f'{where}: local path'] if BINPATH.search(line) else []
            continue
        if (BINPATH if binary else PATH).search(line):
            hits.append(f'{where}: local path')
        if not binary and LABEL.search(line):
            hits.append(f'{where}: internal label')
        hits += [f'{where}: deny-listed term {w!r}' for w in sorted(word_hits(line, binary))]
    return hits


def main():
    files = [f for f in tracked() if f != SELF]
    hits = [h for f in files for h in check(f)]
    print(f'check_release: {len(files)} files scanned, {len(hits)} hits')
    print('\n'.join(hits))
    sys.exit(1 if hits else 0)


if __name__ == '__main__':
    main()
