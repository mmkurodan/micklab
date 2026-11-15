#!/usr/bin/env bash
# サイト上のコードジェネレータ（codegen）とローカルファイルに差がないかを検証する。
# 差があれば警告を表示して終了コード 1、検証できなければ 2、一致すれば 0。
#   使い方: verify-codegen.sh [-v]     （-v: 一致した場合も結果を表示）
#   環境変数: CODEGEN_SITE（既定 https://micklab.web.app）
set -u

SITE="${CODEGEN_SITE:-https://micklab.web.app}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PUBLIC="$REPO/public"
VERBOSE=0
[ "${1:-}" = "-v" ] && VERBOSE=1

# 検証対象: ページ本体・サービスワーカー・PWA 用ファイル一式
FILES=(codegen.html codegen-sw.js)
for f in "$PUBLIC"/codegen/*; do
  [ -f "$f" ] && FILES+=("codegen/$(basename "$f")")
done

if [ -t 2 ]; then RED=$'\033[1;31m'; YEL=$'\033[1;33m'; GRN=$'\033[32m'; RST=$'\033[0m'; else RED=; YEL=; GRN=; RST=; fi

TMP="$(mktemp -d)" || exit 2
trap 'rm -rf "$TMP"' EXIT

diffs=(); errors=()

# 全ファイルを 1 回の curl（接続を再利用）で取得する。
# CDN や中間キャッシュの古い内容を掴まないよう no-cache を付ける。
args=()
for i in "${!FILES[@]}"; do
  [ "$i" -gt 0 ] && args+=(--next)
  args+=(-sS --connect-timeout 4 --max-time 12 -H 'Cache-Control: no-cache'
         -o "$TMP/$i" -w '%{http_code}\n' "$SITE/${FILES[$i]}")
done
mapfile -t codes < <(curl "${args[@]}" 2>"$TMP/err")

for i in "${!FILES[@]}"; do
  f="${FILES[$i]}"; code="${codes[$i]:-000}"
  if [ "$code" = "000" ]; then
    errors+=("$(head -n 1 "$TMP/err")")
    break   # 通信できないなら残りも同じなので打ち切る
  elif [ "$code" != "200" ]; then
    diffs+=("$f（サイトが HTTP $code を返しました）")
  elif ! cmp -s "$TMP/$i" "$PUBLIC/$f"; then
    l="$(sha256sum <"$PUBLIC/$f" | cut -c1-12)"; r="$(sha256sum <"$TMP/$i" | cut -c1-12)"
    diffs+=("$f（ローカル sha256:$l… / サイト sha256:$r…）")
  fi
done

if [ ${#diffs[@]} -gt 0 ]; then
  {
    echo "${RED}⚠ 警告: サイト上の codegen とローカルファイルに差があります${RST}"
    echo "  サイト  : $SITE"
    echo "  ローカル: $PUBLIC"
    for d in "${diffs[@]}"; do echo "   - $d"; done
    # 原因の切り分けの手がかり（未コミット / 未 push なら自分の作業中の変更である可能性が高い）
    if git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
      if [ -n "$(git -C "$REPO" status --porcelain -- public/codegen.html public/codegen-sw.js public/codegen 2>/dev/null)" ]; then
        echo "  ※ ローカルに未コミットの変更があります。"
      elif [ -n "$(git -C "$REPO" log --oneline '@{u}..HEAD' -- public/codegen.html public/codegen-sw.js public/codegen 2>/dev/null)" ]; then
        echo "  ※ ローカルに未 push のコミットがあります。"
      else
        echo "  ※ ローカルは push 済みの内容と同じです。デプロイ未完了、またはサイト側が意図せず変わっている可能性があります。"
      fi
    fi
    echo "  確認: diff <(curl -s $SITE/codegen.html) $PUBLIC/codegen.html"
  } >&2
  exit 1
fi

if [ ${#errors[@]} -gt 0 ]; then
  {
    echo "${YEL}codegen 検証: サイトに接続できず、検証できませんでした${RST}"
    for e in "${errors[@]}"; do echo "   - $e"; done
  } >&2
  exit 2
fi

[ "$VERBOSE" = 1 ] && echo "${GRN}codegen 検証: OK（サイトとローカルは一致 / ${#FILES[@]} ファイル）${RST}" >&2
exit 0
