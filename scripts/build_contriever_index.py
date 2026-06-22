"""Encode all 8.8M MS MARCO passages with Contriever and store as a FlexIndex.

The corpus GRAPH is NOT rebuilt — we symlink the existing TCT-ColBERT-v2 k16
graph so the GNN uses the same graph topology with different node features.

Usage:
  python scripts/build_contriever_index.py
  python scripts/build_contriever_index.py --model facebook/contriever-msmarco
  python scripts/build_contriever_index.py --batch_size 256   # if you have VRAM

Runtime: ~2-3 hours on RTX 3090 Ti at batch_size=128.
Output:  data/msmarco-index_contriever/vecs.f4  (27 GB)
         data/msmarco-index_contriever/corpusgraph_k16  -> symlink to tctcolbert2 graph
"""
import os, sys, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import pyterrier as pt
if not pt.started():
    pt.init()

from pyterrier_dr import FlexIndex
from more_itertools import chunked
from transformers import AutoTokenizer, AutoModel
from npids import Lookup


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='facebook/contriever',
                    help='HuggingFace model id (facebook/contriever or facebook/contriever-msmarco)')
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--max_length', type=int, default=512)
    ap.add_argument('--out_dir', default='data/msmarco-index_contriever')
    ap.add_argument('--tct_index', default='data/msmarco-index_tctcolbert2',
                    help='Existing TCT index to symlink corpus graph from')
    args = ap.parse_args()

    out = Path(args.out_dir)
    tct = Path(args.tct_index)
    vecs_path = out / 'vecs.f4'
    docnos_txt = out / 'docnos.txt'   # plain-text accumulator for resume support
    vec_size = 768

    out.mkdir(parents=True, exist_ok=True)

    # Determine resume point from already-written vectors.
    n_done = 0
    if vecs_path.exists():
        n_done = vecs_path.stat().st_size // (vec_size * 4)
        print(f'Resuming from passage {n_done:,} (found {vecs_path.stat().st_size / 1e9:.2f} GB)')
        # If docnos.txt is missing but npids exists (from a previous run that used the
        # old code path), reconstruct docnos.txt from the corpus to stay consistent.
        if not docnos_txt.exists():
            print('Rebuilding docnos.txt from corpus (one-time migration)...')
            corpus_ds0 = pt.get_dataset('irds:msmarco-passage')
            with open(docnos_txt, 'w') as f:
                for i, rec in enumerate(corpus_ds0.get_corpus_iter(verbose=True)):
                    if i >= n_done:
                        break
                    f.write(rec['docno'] + '\n')
            print(f'  docnos.txt written with {n_done:,} entries')
    else:
        print(f'Starting fresh encoding → {vecs_path}')

    print(f'Loading model: {args.model}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    print('Loading MS MARCO corpus iterator...')
    corpus_ds = pt.get_dataset('irds:msmarco-passage')
    corpus_iter = corpus_ds.get_corpus_iter(verbose=True)

    count = n_done
    skipped = 0

    # Append to vecs.f4; append docnos to plain-text file.
    fvecs_mode = 'ab' if n_done > 0 else 'wb'
    fdocnos_mode = 'a' if n_done > 0 else 'w'

    with open(vecs_path, fvecs_mode) as fvecs, \
         open(docnos_txt, fdocnos_mode) as fdocnos:

        buf_texts, buf_docnos = [], []
        for record in corpus_iter:
            # Skip passages already encoded in a previous run.
            if skipped < n_done:
                skipped += 1
                continue

            buf_texts.append(record['text'])
            buf_docnos.append(record['docno'])
            if len(buf_texts) >= args.batch_size:
                _encode_and_write_plain(model, tokenizer, buf_texts, buf_docnos,
                                        device, args.max_length, fvecs, fdocnos)
                count += len(buf_texts)
                buf_texts, buf_docnos = [], []
                if count % 100_000 == 0:
                    print(f'  {count:,} passages encoded', flush=True)

        if buf_texts:
            _encode_and_write_plain(model, tokenizer, buf_texts, buf_docnos,
                                    device, args.max_length, fvecs, fdocnos)
            count += len(buf_texts)

    print(f'All {count:,} passages encoded. Building npids lookup...')
    # Convert plain-text docnos → npids binary lookup (fast, in-memory).
    with open(docnos_txt) as f:
        docno_list = [line.rstrip('\n') for line in f if line.strip()]
    with Lookup.builder(out / 'docnos.npids') as lkp:
        for dn in docno_list:
            lkp.add(dn)

    import json
    (out / 'pt_meta.json').write_text(json.dumps({
        'type': 'dense_index', 'format': 'flex',
        'vec_size': vec_size, 'doc_count': count,
    }))
    print(f'Done. {count:,} passages → {vecs_path}')
    _symlink_graph(tct, out)


def _encode_and_write_plain(model, tokenizer, texts, docnos, device, max_length, fvecs, fdocnos):
    with torch.no_grad():
        inps = tokenizer(texts, padding=True, truncation=True,
                         max_length=max_length, return_tensors='pt')
        inps = {k: v.to(device) for k, v in inps.items()}
        out = model(**inps).last_hidden_state
        mask = inps['attention_mask']
        emb = (out * mask.unsqueeze(-1).float()).sum(1) / mask.float().sum(1, keepdim=True).clamp(min=1e-9)
        emb_np = emb.cpu().float().numpy()
    for vec, dn in zip(emb_np, docnos):
        fvecs.write(vec.astype(np.float32).tobytes())
        fdocnos.write(dn + '\n')


def _symlink_graph(tct: Path, out: Path):
    """Symlink the TCT corpus graph into the Contriever index directory."""
    graph_src = tct / 'corpusgraph_k16'
    graph_dst = out / 'corpusgraph_k16'
    if not graph_src.exists():
        print(f'WARNING: TCT corpus graph not found at {graph_src} — skipping symlink.')
        return
    if graph_dst.exists() or graph_dst.is_symlink():
        print(f'Corpus graph already linked at {graph_dst}')
        return
    graph_dst.symlink_to(graph_src.resolve())
    print(f'Symlinked corpus graph: {graph_dst} -> {graph_src.resolve()}')


if __name__ == '__main__':
    main()
