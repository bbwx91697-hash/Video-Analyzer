# 动态抽帧核心规则的单元测试。
import sys
from pathlib import Path
import unittest
import numpy as np
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dynamic_frames as d

class SelectionTests(unittest.TestCase):
    # 输入测试实例 self；断言缩放不超过包围盒且保持比例，无返回值。
    def test_analysis_size_preserves_aspect(self):
        w, h = d.analysis_size(1954, 1080, 480, 216)
        self.assertLessEqual(w, 480)
        self.assertLessEqual(h, 216)
        self.assertAlmostEqual(w / h, 1954 / 1080, delta=.01)

    # 输入测试实例 self；用合成帧验证密集抽帧相对上次选中帧去重，无返回值。
    def test_dense_motion_requires_change_from_selected_frame(self):
        pts = [0, .1, .2, .3, .4]
        imgs = {i: np.full((16,16), i, dtype=np.uint8) for i in range(5)}
        metrics = {i: (1, 20, .6, 8, 0) for i in range(5)}
        a = SimpleNamespace(dense_fps=5, normal_fps=2.5, static_interval=2.5,
                            cumulative_threshold=12, min_interval=.1, selected_diff_threshold=3, motion_max_interval=.6, cumulative_interval=.2)
        result = d.select(pts, imgs, list(range(5)), metrics, {0:['first_frame']}, {}, a)
        self.assertEqual([x['frame_index'] for x in result], [0, 3])

if __name__ == '__main__': unittest.main()
