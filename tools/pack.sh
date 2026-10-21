#!/bin/bash
# Đóng gói repo AI_park để gửi đi (gửi Claude đầu phiên...): MỘT lệnh chuẩn, thay cho nén tay thư mục.
#
#   tools/pack.sh            -> ../AI_park_<ngày>_<giờ>.zip (cạnh thư mục repo, không nằm trong repo) + mã md5
#
# Lấy file: git ls-files -co --exclude-standard = file đã commit + file mới chưa commit, BỎ mọi thứ .gitignore chặn
# (.env, log, video, model, out/...). Nén tay 2026-10-21 đã lọt 2 file .env (link camera có mật khẩu), log, video văn phòng.
# Chặn bí mật: đọc GIÁ TRỊ thật trong mọi file .env của repo, có trong file nào của gói thì dừng, không tạo zip.
# Không chặn theo mẫu "rtsp://user:pass@": test và .env.example cố ý chứa link giả, sẽ báo nhầm.
# Chỉ đọc repo, chỉ ghi 1 file zip. Không in giá trị bí mật ra màn hình.
set -u
export LC_ALL=C
ROOT=$(cd "$(dirname "$0")/.." && pwd)   # AI_park
die() { echo "LỖI: $*" >&2; exit 1; }
cd "$ROOT" || die "không vào được $ROOT"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "$ROOT không phải repo git"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
git ls-files -co --exclude-standard | while read -r f; do [ -f "$f" ] && echo "$f"; done > "$TMP/files"
[ -s "$TMP/files" ] || die "không có file nào để gói"

# 1. tên file: .env thật không bao giờ được vào gói (chỉ .env.example), kể cả khi lỡ git add -f
bad=$(grep -E '(^|/)\.env(\.[^/]*)?$' "$TMP/files" | grep -vE '\.env\.example$')
[ -z "$bad" ] || die "gói có file .env: $bad"

# 2. nội dung: giá trị thật của mọi .env trên máy (bỏ giá trị < 8 ký tự: quá ngắn, dễ trùng chữ thường)
find . -name '.env' -not -path './.git/*' -exec grep -hvE '^[[:space:]]*(#|$)' {} + 2>/dev/null \
    | cut -d= -f2- | sed -E "s/^[[:space:]]*['\"]?//; s/['\"]?[[:space:]]*$//" | awk 'length >= 8' | sort -u > "$TMP/secrets"
if [ -s "$TMP/secrets" ]; then
    leak=$(tr '\n' '\0' < "$TMP/files" | xargs -0 grep -lF -f "$TMP/secrets" 2>/dev/null)
    [ -z "$leak" ] || die "các file sau chứa giá trị lấy từ .env (link / mật khẩu camera), sửa rồi gói lại: $(echo $leak)"
fi

OUT=$(dirname "$ROOT")/AI_park_$(date +%Y%m%d_%H%M).zip
rm -f "$OUT"
tr '\n' '\0' < "$TMP/files" | xargs -0 zip -q "$OUT" || die "zip lỗi"
echo "đã gói $(wc -l < "$TMP/files") file, $(du -h "$OUT" | cut -f1) -> $OUT"
echo "đã kiểm: không có file .env, không có giá trị nào từ $(wc -l < "$TMP/secrets") dòng .env"
md5sum "$OUT"