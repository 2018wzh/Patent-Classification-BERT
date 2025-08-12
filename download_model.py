import os
import argparse
from typing import List
from huggingface_hub import list_repo_files, hf_hub_download

NEEDED_FILES_PRIORITY = [
    'config.json',
    'model.safetensors',  # 权重
    'vocab.txt',          # WordPiece 词表
    'tokenizer.json',
    'tokenizer_config.json',
    'special_tokens_map.json'
]


def filter_existing(repo_files: List[str]):
    existing = {}
    s = set(repo_files)
    for name in NEEDED_FILES_PRIORITY:
        if name in s:
            existing[name] = name
    return existing


def main():
    parser = argparse.ArgumentParser(description='Download minimal subset of a HF model (safetensors + vocab)')
    parser.add_argument('--model', default='anferico/bert-for-patents', help='模型仓库名称')
    parser.add_argument('--out', default='model', help='输出目录')
    parser.add_argument('--revision', default=None, help='可选指定 revision/commit/tag')
    parser.add_argument('--force', action='store_true', help='已存在仍重新下载覆盖')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f'[INFO] 列出远端文件: {args.model} ...')
    repo_files = list_repo_files(args.model, revision=args.revision)  # 需要 huggingface_hub >=0.17
    selected = filter_existing(repo_files)
    if 'model.safetensors' not in selected:
        raise SystemExit('远端仓库未找到 model.safetensors，请确认该模型是否提供 safetensors 文件。')
    if 'vocab.txt' not in selected:
        print('[WARN] 未找到 vocab.txt，可能该模型使用不同 tokenizer。')

    print('[INFO] 将下载以下文件:')
    for f in selected:
        print('  -', f)

    for fname in selected:
        target_path = os.path.join(args.out, fname)
        if os.path.exists(target_path) and not args.force:
            print(f'[SKIP] {fname} 已存在 (使用 --force 覆盖)')
            continue
        print(f'[DL] 下载 {fname} ...')
        local_file = hf_hub_download(
            repo_id=args.model,
            filename=fname,
            revision=args.revision,
            local_dir=args.out,
            local_dir_use_symlinks=False,
            force_download=args.force
        )
        if local_file != target_path:
            # hf_hub_download 可能带子目录；确保移动或复制 (通常已在 out 内)
            pass
    print('[DONE] 下载完成，文件保存在:', os.path.abspath(args.out))
    print('可在 train.py 中使用 --model', args.out)


if __name__ == '__main__':
    main()
