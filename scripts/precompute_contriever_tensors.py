"""Precompute fast train/val tensors with Contriever features.

Reuses the existing TCT fast tensor directory structure:
  - adjacency_matrix.pt  → copied as-is (graph stays the same)
  - qrels_tensor_new.pt  → copied as-is
  - doc_feat_tensor_new.pt  → re-encoded with Contriever
  - query_tensor_new.pt     → re-encoded with Contriever

Output directories:
  data/msmarco_data/train_data_fast_contriever/
  data/msmarco_data/val_data_fast_contriever/

Usage:
  python scripts/precompute_contriever_tensors.py
  python scripts/precompute_contriever_tensors.py --split val
  python scripts/precompute_contriever_tensors.py --model facebook/contriever-msmarco
"""
import os, sys, json, shutil, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
import pandas as pd
import pyterrier as pt
if not pt.started():
    pt.init()

from src.contriever_encoder import ContrieverEncoder


MAX_DOCS = 1000


def encode_query_and_docs(encoder, qid, query_text, bm25_json_path, add_text):
    """Return (z_q [1,768], z_d [N,768], docnos [N]) for one query."""
    json_file = bm25_json_path / f'{qid}.json'
    if not json_file.exists():
        return None, None, None

    with open(json_file) as f:
        candidates = json.load(f)
    cand_df = pd.DataFrame(candidates)
    cand_df = add_text(cand_df).drop_duplicates(subset='docno')
    docs = list(cand_df['text'])
    docnos = list(cand_df['docno'])
    if len(docs) == 0:
        return None, None, None

    z_q = encoder.encode_queries([query_text])             # [1, 768]
    z_d = encoder.encode_docs(docs)                        # [N, 768]
    return (torch.tensor(z_q, dtype=torch.float32),
            torch.tensor(z_d, dtype=torch.float32),
            docnos)


def process_split(split, encoder, bm25_base, src_base, dst_base, topics_df, add_text, skip_existing=True):
    bm25_path = Path(bm25_base)
    src_path = Path(src_base)
    dst_path = Path(dst_base)

    src_tensor_dir = src_path / 'tensors'
    dst_tensor_dir = dst_path / 'tensors'
    dst_tensor_dir.mkdir(parents=True, exist_ok=True)

    # Process each qid that has a source fast-tensor directory
    qid_dirs = sorted(src_tensor_dir.glob('qid_*_tensors'))
    topics_map = dict(zip(topics_df['qid'].astype(str), topics_df['query']))

    # count already-done up front so the user can see resume state
    n_done = sum(
        1 for d in qid_dirs if (dst_tensor_dir / d.name / 'doc_feat_tensor_new.pt').exists()
    ) if skip_existing else 0
    n_todo = len(qid_dirs) - n_done
    print(f'[{split}] {len(qid_dirs)} total queries — '
          f'{n_done} already done, {n_todo} to encode', flush=True)

    encoded = 0
    for src_qdir in qid_dirs:
        qid = src_qdir.name.replace('qid_', '').replace('_tensors', '')
        dst_qdir = dst_tensor_dir / src_qdir.name

        if skip_existing and (dst_qdir / 'doc_feat_tensor_new.pt').exists():
            continue

        query_text = topics_map.get(qid)
        if query_text is None:
            continue

        z_q, z_d, docnos = encode_query_and_docs(
            encoder, qid, query_text, bm25_path, add_text)
        if z_q is None:
            continue

        dst_qdir.mkdir(exist_ok=True)

        # Save new Contriever features
        torch.save(z_d, dst_qdir / 'doc_feat_tensor_new.pt')      # [N, 768]
        torch.save(z_q, dst_qdir / 'query_tensor_new.pt')          # [1, 768]

        # Copy graph structure and labels unchanged
        for fname in ('adjacency_matrix.pt', 'qrels_tensor_new.pt'):
            src_f = src_qdir / fname
            if src_f.exists():
                shutil.copy2(src_f, dst_qdir / fname)

        encoded += 1
        if encoded % 500 == 0:
            print(f'  [{split}] encoded {encoded}/{n_todo}', flush=True)

    print(f'[{split}] Done — encoded {encoded} new queries → {dst_tensor_dir}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='facebook/contriever')
    ap.add_argument('--split', default='both', choices=['train', 'val', 'both'])
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--no_skip', action='store_true', help='Re-encode even if output exists')
    args = ap.parse_args()

    emb_tag = 'contriever' if args.model == 'facebook/contriever' else 'contriever-msmarco'
    print(f'Encoder: {args.model}  →  tag: {emb_tag}')

    encoder = ContrieverEncoder(args.model, batch_size=args.batch_size)
    corpus_ds = pt.get_dataset('irds:msmarco-passage')
    add_text = pt.text.get_text(corpus_ds, 'text')

    if args.split in ('train', 'both'):
        train_ds = pt.get_dataset('irds:msmarco-passage/train/split200-train')
        process_split(
            split='train',
            encoder=encoder,
            bm25_base='data/msmarco_data/msmarco_pre-computed_bm25/train',
            src_base='data/msmarco_data/train_data_fast',
            dst_base=f'data/msmarco_data/train_data_fast_{emb_tag}',
            topics_df=train_ds.get_topics(),
            add_text=add_text,
            skip_existing=not args.no_skip,
        )

    if args.split in ('val', 'both'):
        val_ds = pt.get_dataset('irds:msmarco-passage/train/split200-valid')
        process_split(
            split='val',
            encoder=encoder,
            bm25_base='data/msmarco_data/msmarco_pre-computed_bm25/val',
            src_base='data/msmarco_data/val_data_fast',
            dst_base=f'data/msmarco_data/val_data_fast_{emb_tag}',
            topics_df=val_ds.get_topics(),
            add_text=add_text,
            skip_existing=not args.no_skip,
        )


if __name__ == '__main__':
    main()
