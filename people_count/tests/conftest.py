"""Cấu hình chung cho test people_count.

Test chỉ chạm phần logic (vùng, đếm, config, event): không cần GPU, DeepStream hay video.
Phần dính DeepStream (pipeline.py) KHÔNG được import ở đây (KIEN_TRUC mục 3.4), có test riêng khoá điều đó.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # thư mục people_count
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(ROOT))  # thư mục AI_park: để import common/