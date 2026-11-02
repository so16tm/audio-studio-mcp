"""ACE / MOSS の検証済み重みを全量取得する（トークン入力なし）。"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", help="Hugging Face キャッシュの保存先（省略時は標準設定）")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    from audio_studio_mcp.backends.ace_moss import (
        ACE_MODEL_ID,
        ACE_MODEL_REVISION,
        MOSS_MODEL_ID,
        MOSS_MODEL_REVISION,
    )

    # ACE は LM 推論を無効にしてもバンドル全体の存在を要求する。
    for repo_id, revision in (
        (ACE_MODEL_ID, ACE_MODEL_REVISION),
        (MOSS_MODEL_ID, MOSS_MODEL_REVISION),
    ):
        print(f"Downloading {repo_id} @ {revision}", flush=True)
        path = snapshot_download(repo_id=repo_id, revision=revision, cache_dir=args.cache_dir)
        print(f"Complete: {path}", flush=True)


if __name__ == "__main__":
    main()
