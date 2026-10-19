"""Che user:pass trong URL trước khi in log. Dùng chung cho mọi bài toán đọc camera."""
import re

# Che tới dấu @ CUỐI CÙNG trước dấu / đầu tiên: mật khẩu dán nguyên văn có thể chứa @ (vd p@ss123).
# Bản cũ dừng ở @ đầu tiên -> lộ phần sau (phát hiện 2026-10-15, khoá bằng test T11b của camera_tamper).
_CRED = re.compile(r"//[^/\s]*@")


def mask(s):
    """Che user:pass trong URL trước khi in log."""
    return _CRED.sub("//***@", s)