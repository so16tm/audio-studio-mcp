# ACE-Step／MOSSへのMCP移行

目的: 既存のClaude向けMCPの4ツールを維持し、BGM・楽器・rawをACE-Step 1.5、効果音をMOSS-SoundEffect v2で生成する。CodexとClaude Codeから同じサーバーを利用できるようにする。

方式: 前段の実機検証で動作したPython 3.12／PyTorch 2.9 CUDA 12.8を専用環境へ導入する。モデルは単一ワーカーで切り替え、前モデルを解放してから次を読む。元のStable Audio環境は明示指定の旧方式として扱う。前段の検証記録は `outputs/model-replacement-validation/REPORT.ja.md`。

## 契約

- `generate_audio`、`edit_audio`、`process_audio`、`unload_models`を維持する。
- 用途別の不変なprofileでモデル名・サンプルレート・既定値・制限・revisionを定め、生成後のメタデータに別モデルの情報が混ざらないようにする。
- ACE: 48kHzステレオ、10〜600秒、8 steps、cfg=1、インストのみ。カバー変換・部分再生成・延長を実装する。turboのCFG変更・negative promptは受け付けない。
- MOSS: 48kHzモノラル、0.5〜30秒、既定100 steps／cfg=4。音源のAI編集は明確に拒否し、仕上げは利用できる。
- 未指定のsteps/cfgを用途別に解決する。既存のsecondsエイリアス、候補seed、sidecar JSON、ループ・音量調整・書き出しを維持する。
- 編集の用途を省略した場合は元ファイルのsidecar/sourceから推定し、SFXが誤ってACEへ流れないようにする。
- 新規の本番環境は `.venv-hybrid`。固定ソースを `vendor/` へ導入し、全モデル重みを先に取得する。既存の他MCP設定は保持する。

## 作業と検証

- [x] バックエンド: profile、用途別ルーティング、固定revision、モデル切替・解放。担当範囲のテスト23件成功。
- [x] MCP境界: プロンプト・既定値・制約・元音声の用途推定・metadataをテスト先行で実装。明示的な用途指定でもMOSS音源のAI編集は拒否する。
- [x] セットアップと説明: 専用インストーラー、重み取得、Claude/Codex登録手順。Windows PowerShell 5.1で実行し、Python 3.12.10／PyTorch 2.9.0+cu128と両モデルを確認。依存関係検査も成功。
- [x] Web UI: 用途別制約、モデル表示、SFX編集不可、レシピ復元。担当範囲48件成功、ブラウザ試験は環境依存で4件skip。
- [x] 統合: 本番の専用環境で142件成功、4件skip（Playwright未導入のブラウザ試験）。Web APIとJavaScriptの操作ロジックは検証済み。独立レビューで見つかったSFX編集制限の回避とACE編集の尺指定を修正し、回帰試験を実施。GitHub Actionsは未使用。
- [x] 実機: 登録した実行ファイルへMCP接続し、ACE→MOSS→ACEの生成・編集・書き出し・解放が成功。複数候補のseed、48kHz／チャンネル数、非無音・有限値、metadata、元ファイル不変、20秒から10秒への変換も確認。生成物16ファイルとログは `outputs/mcp-hybrid-smoke/`、`outputs/mcp-hybrid-smoke.log`。終了後のGPU使用量は0 MiB。
- [x] 登録: Codexのaudio-studio項目を `.venv-hybrid` へ更新し、有効状態を確認。Claude Codeはユーザースコープで登録し `Connected` を確認。他MCP設定は保持。Claude Desktopの既存設定ファイルは存在しなかったため、READMEに設定例を掲載。既存セッションは再起動で新しい登録を読み込む。

完了条件: 上記の機能が実モデルで動き、クライアントの登録先と依存環境・READMEが一致していること。音質の一般的な優劣や長時間の負荷試験は、この移行の完了条件には含めない。
