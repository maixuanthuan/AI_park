"""Test demo: chỉ phần không cần model (đọc đối số, đổi kết quả model, module băng rôn). Không cần GPU, ultralytics."""
import os
import sys

DEMO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(DEMO)
sys.path[:0] = [DEMO, os.path.join(ROOT, "people_count"), ROOT]