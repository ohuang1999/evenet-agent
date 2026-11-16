#!/usr/bin/env python3
"""Key=value comparison of MG cards: a reference run vs a new run (or process dir) Cards.
Reference implementation from the H5+ -> WZ background extension (charged_higgs/bkg_WZ, Nov 2026); adapt as needed.

run_card: compared against the <MGRunCard> block of the reference production banner (the card the run
actually used; Cards/run_card.dat is rewritten by MG after the run, iseed reset to 0) AND against the
reference Cards/run_card.dat. pythia8_card / madspin_card / param_card / proc_card: key=value (or line) sets.
Usage: cards_semantic_diff.py <REF_RUN_DIR> <NEW_CARDS_DIR | NEW_RUN_DIR>
Prints only differing keys; 'IDENTICAL' when none.
"""
import glob, os, re, sys

ref, new = sys.argv[1], sys.argv[2]
if os.path.isdir(f'{new}/Cards'):
    new = f'{new}/Cards'


def kv_run(text):
    d = {}
    for line in text.splitlines():
        line = line.split('!')[0].split('#')[0]
        m = re.match(r'\s*(.*?)\s*=\s*(\S+)\s*$', line)
        if m and m.group(2):
            d[m.group(2).lower()] = m.group(1).strip()
    return d


def kv_py8(text):
    d = {}
    for line in text.splitlines():
        line = line.split('!')[0].strip()
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            d[k.strip().lower()] = v.strip()
    return d


def lines(text):
    return {l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith('#')}


def cmp(name, a, b):
    if isinstance(a, dict):
        diff = [(k, a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
        print(f'[{name}]', 'IDENTICAL' if not diff else '')
        for k, x, y in diff:
            print(f'    {k}: {x}  ->  {y}')
    else:
        print(f'[{name}]', 'IDENTICAL' if a == b else '')
        for l in sorted(a - b):
            print(f'    - {l}')
        for l in sorted(b - a):
            print(f'    + {l}')


rd = lambda p: open(p, errors='replace').read()
banner = [b for b in sorted(glob.glob(f'{ref}/*banner.txt')) if 'decayed' not in b][0]
m = re.search(r'<MGRunCard>(.*?)</MGRunCard>', rd(banner), re.S)
newrun = kv_run(rd(f'{new}/run_card.dat'))
cmp('run_card vs ref banner (card used)', kv_run(m.group(1)), newrun)
cmp('run_card vs ref Cards/run_card.dat', kv_run(rd(f'{ref}/Cards/run_card.dat')), newrun)
cmp('pythia8_card', kv_py8(rd(f'{ref}/Cards/pythia8_card.dat')), kv_py8(rd(f'{new}/pythia8_card.dat')))
for c in ('param_card.dat', 'proc_card_mg5.dat', 'madspin_card.dat', 'delphes_card.dat'):
    if os.path.exists(f'{ref}/Cards/{c}') or os.path.exists(f'{new}/{c}'):
        cmp(c, lines(rd(f'{ref}/Cards/{c}')), lines(rd(f'{new}/{c}')))
