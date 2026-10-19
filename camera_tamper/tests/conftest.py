"""Cấu hình chung cho test camera_tamper.

Test dùng đúng phần `detector` của config.yaml (bản đang giao), chỉ ghi đè mục nào test cần.
Vì vậy test kiểm hành vi của cấu hình thật, và mọi mốc thời gian trong test đọc từ cấu hình,
không chép cứng con số: chỉnh ngưỡng hợp lý thì test vẫn xanh.
"""
import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # thư mục camera_tamper
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(ROOT))  # thư mục AI_park: để import common/


@pytest.fixture
def cfg():
    with open(os.path.join(ROOT, "config.yaml")) as f:
        base = yaml.safe_load(f)["detector"]

    def make(**override):
        return {**base, **override}

    return make
