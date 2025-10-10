import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import os
from collections import defaultdict
import math
'''
模型5跟踪重捕模型具体算法
'''

class FeatureDatabase:
    """特征数据库类，管理正负样本特征"""

    def __init__(self, max_features=500):
        self.positive_features = []  # 正样本特征（目标区域）
        self.negative_features = []  # 负样本特征（背景区域）
        self.positive_descriptors = []
        self.negative_descriptors = []
        self.max_features = max_features

    def add_positive_features(self, keypoints, descriptors):
        """添加正样本特征"""
        self.positive_features.extend(keypoints)
        if descriptors is not None:
            self.positive_descriptors.extend(descriptors)

        # 限制特征数量
        if len(self.positive_features) > self.max_features:
            self.positive_features = self.positive_features[-self.max_features:]
            if self.positive_descriptors:
                self.positive_descriptors = self.positive_descriptors[-self.max_features:]

    def add_negative_features(self, keypoints, descriptors):
        """添加负样本特征"""
        self.negative_features.extend(keypoints)
        if descriptors is not None:
            self.negative_descriptors.extend(descriptors)

        # 限制特征数量
        if len(self.negative_features) > self.max_features:
            self.negative_features = self.negative_features[-self.max_features:]
            if self.negative_descriptors:
                self.negative_descriptors = self.negative_descriptors[-self.max_features:]

    def update_features(self, new_positive_kp, new_positive_desc,
                        new_negative_kp=None, new_negative_desc=None):
        """更新特征数据库"""
        # 更新正样本
        self.positive_features = new_positive_kp
        self.positive_descriptors = new_positive_desc if new_positive_desc is not None else []

        # 更新负样本（如果提供）
        if new_negative_kp is not None:
            self.negative_features = new_negative_kp
            self.negative_descriptors = new_negative_desc if new_negative_desc is not None else []


class DroneTracker:
    """无人机跟踪器类"""

    def __init__(self, track_id, initial_bbox, initial_frame):
        self.track_id = track_id
        self.bbox = initial_bbox  # (x1, y1, x2, y2)
        self.center = self._get_center(initial_bbox)
        self.size = self._get_size(initial_bbox)

        # 跟踪状态
        self.state = "TRACKING"  # TRACKING, LOST, WAITING
        self.lost_frames = 0
        self.max_lost_frames = 30
        self.min_hits = 1
        self.hits = 0
        self.time_since_update = 0

        # —— 尺度/尺寸保护阈值（放这里，避免找不到属性） ——
        self.scale_step_clip = (0.88, 1.12)  # 单帧尺度允许 [−12%, +12%]
        self.max_box_area_ratio = 0.12  # 框最大面积占画面比例
        self.min_box_side = 8  # 最小边（像素）

        # 仅用于“显示”的平滑/放大框（不影响真实跟踪）
        self.display_bbox = None
        self.disp_inflate = 1.22   # 显示时放大比例
        self.disp_pad = 4           # 显示时额外像素边距
        self.disp_alpha = 0.5       # 显示框EMA平滑系数
        self.hold_frames = 2        # 可视化保留帧，避免短暂丢帧闪烁
        self.scale_ema = 1.0        # 可选：尺度EMA，缓解“呼吸感”

        # 特征数据库
        self.feature_db = FeatureDatabase()

        # 光流跟踪
        self.prev_gray = None
        self.prev_features = None

        # 匹配阈值
        self.global_match_threshold = 0.25  # 原0.3
        self.local_match_threshold = 0.3    # 原0.4
        self.combined_match_threshold = 0.35

        # 变换参数
        self.scale_factor = 1.0
        self.rotation_angle = 0.0

        # 初始化特征提取器
        self.sift = cv2.SIFT_create(nfeatures=200)
        self.bf_matcher = cv2.BFMatcher()

        # 初始化跟踪器
        self._initialize_tracker(initial_frame)

    def _get_center(self, bbox):
        """获取边界框中心点"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _get_size(self, bbox):
        """获取边界框尺寸"""
        x1, y1, x2, y2 = bbox
        return (x2 - x1, y2 - y1)

    def _initialize_tracker(self, frame):
        """初始化跟踪器参数"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_gray = gray.copy()

        # 提取目标区域和背景区域的特征
        self._extract_initial_features(gray)

        print(f"跟踪器 {self.track_id} 初始化完成")

    def _extract_initial_features(self, gray_frame):
        """提取初始特征并建立正负样本数据库"""
        x1, y1, x2, y2 = self.bbox

        # 目标区域
        target_roi = gray_frame[y1:y2, x1:x2]

        # 背景区域（目标尺寸的2倍范围）
        width, height = x2 - x1, y2 - y1
        bg_x1 = max(0, x1 - width)
        bg_y1 = max(0, y1 - height)
        bg_x2 = min(gray_frame.shape[1], x2 + width)
        bg_y2 = min(gray_frame.shape[0], y2 + height)

        # 提取目标特征（正样本）
        target_kp, target_desc = self.sift.detectAndCompute(target_roi, None)
        if target_kp:
            # 调整关键点坐标到全图坐标系
            adjusted_kp = []
            for kp in target_kp:
                new_kp = cv2.KeyPoint(kp.pt[0] + x1, kp.pt[1] + y1, kp.size,
                                      kp.angle, kp.response, kp.octave, kp.class_id)
                adjusted_kp.append(new_kp)

            self.feature_db.add_positive_features(adjusted_kp, target_desc)

            # 存储用于光流跟踪的特征点
            self.prev_features = np.array([[kp.pt[0], kp.pt[1]] for kp in adjusted_kp],
                                          dtype=np.float32).reshape(-1, 1, 2)
        else:
            # 兜底：若SIFT未取到点，用KLT角点作为首帧光流种子
            corners = cv2.goodFeaturesToTrack(target_roi, maxCorners=80, qualityLevel=0.01,
                                              minDistance=5, blockSize=3)
            if corners is not None:
                adjusted = corners.reshape(-1, 1, 2) + np.array([x1, y1])
                self.prev_features = adjusted.astype(np.float32)

        # 提取背景特征（负样本）
        background_roi = gray_frame[bg_y1:bg_y2, bg_x1:bg_x2]

        # 创建掩码，排除目标区域
        mask = np.ones(background_roi.shape, dtype=np.uint8) * 255
        target_in_bg_x1 = x1 - bg_x1
        target_in_bg_y1 = y1 - bg_y1
        target_in_bg_x2 = x2 - bg_x1
        target_in_bg_y2 = y2 - bg_y1
        mask[target_in_bg_y1:target_in_bg_y2, target_in_bg_x1:target_in_bg_x2] = 0

        bg_kp, bg_desc = self.sift.detectAndCompute(background_roi, mask)
        if bg_kp:
            # 调整背景关键点坐标
            adjusted_bg_kp = []
            for kp in bg_kp:
                new_kp = cv2.KeyPoint(kp.pt[0] + bg_x1, kp.pt[1] + bg_y1, kp.size,
                                      kp.angle, kp.response, kp.octave, kp.class_id)
                adjusted_bg_kp.append(new_kp)

            self.feature_db.add_negative_features(adjusted_bg_kp, bg_desc)

    def update(self, frame, detection_bbox=None):
        """更新跟踪器"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.state == "TRACKING":
            success = self._track_features(gray, detection_bbox)
            if success:
                self.hits += 1
                self.time_since_update = 0
            else:
                self.state = "LOST"
                self.lost_frames = 1
                self.time_since_update += 1
                print(f"跟踪器 {self.track_id} 跟踪丢失")

        elif self.state == "LOST":
            self.lost_frames += 1
            self.time_since_update += 1
            if self.lost_frames > self.max_lost_frames:
                self.state = "DEAD"
                return False

            # 无检测框时的盲重捕
            success = self._try_reacquire(gray)
            if success:
                self.state = "TRACKING"
                self.lost_frames = 0
                self.hits = 1  # 重获后从1开始累计
                self.time_since_update = 0
                print(f"跟踪器 {self.track_id} 重新捕获目标")

            # 有检测框则尝试基于检测的重捕
            if detection_bbox is not None:
                success = self._attempt_reacquisition(gray, detection_bbox)
                if success:
                    self.state = "TRACKING"
                    self.lost_frames = 0
                    self.hits = 1
                    self.time_since_update = 0
                    print(f"跟踪器 {self.track_id} 重新捕获目标")

        self.prev_gray = gray.copy()
        return self.state != "DEAD"

    def _track_features(self, current_gray, detection_bbox=None):
        """特征跟踪主函数"""
        if self.prev_features is None or len(self.prev_features) == 0:
            return False

        # 1. 计算光流
        optical_flow_points = self._calculate_optical_flow(current_gray)

        # 2. 检测当前帧特征
        current_features = self._detect_current_features(current_gray)

        # 3. 全局匹配
        global_matches = self._global_feature_matching(current_features)

        # 4. 局部匹配（光流）
        local_matches = self._local_feature_matching(optical_flow_points, current_features)

        # 5. 融合匹配点
        valid_matches = self._fuse_matches(global_matches, local_matches)

        # 6. 计算匹配率
        global_match_rate = len(global_matches) / max(len(self.feature_db.positive_features), 1)
        local_match_rate = len(local_matches) / max(len(optical_flow_points), 1)

        # 7. 判断跟踪状态
        if (global_match_rate < self.global_match_threshold and
                local_match_rate < self.local_match_threshold):
            return False

        MIN_INLIERS = 12  # 可按目标大小在10~15范围微调
        if len(valid_matches) < MIN_INLIERS:
            return False

        # 8. 更新目标位置
        if len(valid_matches) > 4:
            self._update_target_position(valid_matches)

            # 9. 更新特征数据库
            self._update_feature_database(current_gray, current_features)

            # 10. 更新光流特征点
            self._update_optical_flow_features(current_gray)

        return True

    def _calculate_optical_flow(self, current_gray):
        """计算光流数据"""
        if self.prev_features is None:
            return []

        # Lucas-Kanade光流跟踪
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        next_pts, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, current_gray, self.prev_features, None, **lk_params)

        # 筛选有效的光流点
        valid_flow_points = []
        if next_pts is not None:
            for i, (st, pt) in enumerate(zip(status, next_pts)):
                if st == 1:  # 光流跟踪成功
                    valid_flow_points.append(pt[0])

        return valid_flow_points

    def _detect_current_features(self, current_gray):
        """检测当前帧附近区域的特征"""
        # 扩展搜索区域
        x1, y1, x2, y2 = self.bbox
        width, height = x2 - x1, y2 - y1

        search_x1 = max(0, x1 - width // 2)
        search_y1 = max(0, y1 - height // 2)
        search_x2 = min(current_gray.shape[1], x2 + width // 2)
        search_y2 = min(current_gray.shape[0], y2 + height // 2)

        # 在搜索区域内检测特征
        search_roi = current_gray[search_y1:search_y2, search_x1:search_x2]
        kp, desc = self.sift.detectAndCompute(search_roi, None)

        # 调整坐标到全图坐标系
        adjusted_kp = []
        if kp:
            for keypoint in kp:
                new_kp = cv2.KeyPoint(keypoint.pt[0] + search_x1, keypoint.pt[1] + search_y1,
                                      keypoint.size, keypoint.angle, keypoint.response,
                                      keypoint.octave, keypoint.class_id)
                adjusted_kp.append(new_kp)

        return adjusted_kp, desc

    def _global_feature_matching(self, current_features):
        """全局特征匹配"""
        current_kp, current_desc = current_features

        if (not current_kp or current_desc is None or
                not self.feature_db.positive_descriptors):
            return []

        # 与正样本数据库匹配
        matches = []
        try:
            positive_desc_array = np.array(self.feature_db.positive_descriptors)
            raw_matches = self.bf_matcher.knnMatch(current_desc, positive_desc_array, k=2)

            # Lowe's ratio test
            for match_pair in raw_matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        matches.append((current_kp[m.queryIdx],
                                        self.feature_db.positive_features[m.trainIdx]))
        except:
            pass

        return matches

    def _local_feature_matching(self, optical_flow_points, current_features):
        """局部特征匹配（基于光流）"""
        current_kp, _ = current_features

        if not optical_flow_points or not current_kp:
            return []

        matches = []
        # 将光流点与检测到的特征点进行距离匹配
        for flow_pt in optical_flow_points:
            min_dist = float('inf')
            best_match = None

            for kp in current_kp:
                dist = np.linalg.norm(np.array(flow_pt) - np.array(kp.pt))
                if dist < min_dist and dist < 30:  # 放宽到30像素，适配小目标抖动
                    min_dist = dist
                    best_match = kp

            if best_match is not None:
                matches.append((best_match, flow_pt))

        return matches

    def _fuse_matches(self, global_matches, local_matches):
        """融合全局和局部匹配点"""
        all_matches = []

        # 添加全局匹配点
        for match in global_matches:
            all_matches.append({
                'point': match[0].pt,
                'type': 'global',
                'confidence': 1.0
            })

        # 添加局部匹配点
        for match in local_matches:
            all_matches.append({
                'point': match[0].pt,
                'type': 'local',
                'confidence': 0.8
            })

        # 去除重复点
        unique_matches = []
        for match in all_matches:
            is_duplicate = False
            for existing in unique_matches:
                if np.linalg.norm(np.array(match['point']) - np.array(existing['point'])) < 10:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_matches.append(match)

        return unique_matches

    def _calculate_transformation(self, valid_matches):
        """计算尺度和旋转变换因子"""
        if len(valid_matches) < 4:
            return 1.0, 0.0

        # 提取匹配点坐标
        current_points = np.array([match['point'] for match in valid_matches])

        # 计算质心
        centroid = np.mean(current_points, axis=0)

        # 计算相对于前一帧的变换
        prev_centroid = np.array(self.center)

        # 尺度变换（基于点云的平均距离变化）
        if len(current_points) > 1:
            current_distances = [np.linalg.norm(pt - centroid) for pt in current_points]
            avg_current_dist = np.mean(current_distances)

            # 估算前一帧的平均距离
            prev_size = np.mean(self.size)
            scale_factor = avg_current_dist / (prev_size / 4) if prev_size > 0 else 1.0
            # 尺度做EMA缓冲并限制范围，减少“呼吸感”
            self.scale_ema = 0.8 * self.scale_ema + 0.2 * scale_factor
            scale_factor = float(np.clip(self.scale_ema, *self.scale_step_clip))
        else:
            scale_factor = 1.0

        # 旋转变换（简化处理）
        rotation_angle = 0.0  # 可以根据需要实现更复杂的旋转计算

        return scale_factor, rotation_angle

    def _inflate_bbox(self, bbox, scale, pad, frame_shape):
        """仅用于显示：放大并裁剪到画面内"""
        H, W = frame_shape[:2]
        x1, y1, x2, y2 = map(float, bbox)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = (x2 - x1) * scale, (y2 - y1) * scale
        nx1, ny1 = int(round(cx - w / 2 - pad)), int(round(cy - h / 2 - pad))
        nx2, ny2 = int(round(cx + w / 2 + pad)), int(round(cy + h / 2 + pad))
        nx1 = max(0, min(nx1, W - 1)); ny1 = max(0, min(ny1, H - 1))
        nx2 = max(0, min(nx2, W - 1)); ny2 = max(0, min(ny2, H - 1))
        return (nx1, ny1, nx2, ny2)

    def _smooth_bbox(self, prev_bbox, cur_bbox, alpha):
        """仅用于显示：对框做EMA平滑"""
        return tuple(int(round((1 - alpha) * p + alpha * c)) for p, c in zip(prev_bbox, cur_bbox))

    def _update_target_position(self, valid_matches):
        """根据有效匹配点更新目标位置"""
        if not valid_matches:
            return

        # 计算新的中心位置
        points = np.array([match['point'] for match in valid_matches])
        new_center = np.mean(points, axis=0)

        # 计算变换因子
        scale_factor, rotation_angle = self._calculate_transformation(valid_matches)

        # 更新尺寸（先按尺度放大/缩小，给个最小边兜底）
        new_width = int(max(self.min_box_side, self.size[0] * scale_factor))
        new_height = int(max(self.min_box_side, self.size[1] * scale_factor))

        # —— 防爆炸：限制候选新框的绝对尺寸（面积/边长上限） ——
        H, W = self.prev_gray.shape

        # 1) 面积上限：不超过画面一定比例（max_box_area_ratio）
        max_area = self.max_box_area_ratio * H * W
        new_area = new_width * new_height
        if new_area > max_area:
            scale = (max_area / (new_area + 1e-6)) ** 0.5  # 按面积等比缩回
            new_width = max(self.min_box_side, int(new_width * scale))
            new_height = max(self.min_box_side, int(new_height * scale))

        # 2) 单边上限：任何一边不超过画面 1/3（可按需 0.30~0.40 调）
        new_width = int(np.clip(new_width, self.min_box_side, int(W * 0.33)))
        new_height = int(np.clip(new_height, self.min_box_side, int(H * 0.33)))

        # 更新中心与尺寸
        self.center = (int(new_center[0]), int(new_center[1]))
        self.size = (new_width, new_height)

        # 更新bbox
        self.bbox = (
            self.center[0] - new_width // 2,
            self.center[1] - new_height // 2,
            self.center[0] + new_width // 2,
            self.center[1] + new_height // 2
        )

        # 裁剪到画面内
        x1 = max(0, min(self.bbox[0], W - 1))
        y1 = max(0, min(self.bbox[1], H - 1))
        x2 = max(0, min(self.bbox[2], W - 1))
        y2 = max(0, min(self.bbox[3], H - 1))
        self.bbox = (x1, y1, x2, y2)

        # 存储变换参数
        self.scale_factor = scale_factor
        self.rotation_angle = rotation_angle

        # —— 仅用于显示：放大+平滑显示框（不影响真实bbox） ——
        inflated = self._inflate_bbox(self.bbox, self.disp_inflate, self.disp_pad, self.prev_gray.shape)
        if self.display_bbox is None:
            self.display_bbox = inflated
        else:
            self.display_bbox = self._smooth_bbox(self.display_bbox, inflated, self.disp_alpha)

    def _update_feature_database(self, current_gray, current_features):
        """更新特征数据库"""
        current_kp, current_desc = current_features

        if not current_kp:
            return

        # 筛选目标区域内的特征作为新的正样本
        x1, y1, x2, y2 = self.bbox
        new_positive_kp = []
        new_positive_desc = []

        for i, kp in enumerate(current_kp):
            if x1 <= kp.pt[0] <= x2 and y1 <= kp.pt[1] <= y2:
                new_positive_kp.append(kp)
                if current_desc is not None:
                    new_positive_desc.append(current_desc[i])

        # 更新数据库
        if new_positive_kp:
            self.feature_db.update_features(new_positive_kp, new_positive_desc)

    def _update_optical_flow_features(self, current_gray):
        """更新光流特征点"""
        # 在当前目标区域提取新的特征点用于下一帧光流跟踪
        x1, y1, x2, y2 = self.bbox
        roi = current_gray[y1:y2, x1:x2]

        corners = cv2.goodFeaturesToTrack(roi, maxCorners=100, qualityLevel=0.01,
                                          minDistance=10, blockSize=3)

        if corners is not None:
            # 调整坐标并保存
            adjusted_corners = corners + np.array([x1, y1])
            self.prev_features = adjusted_corners.reshape(-1, 1, 2).astype(np.float32)
        else:
            self.prev_features = None

    def _try_reacquire(self, current_gray):
        """
        在没有 detection_bbox 的情况下做一次“盲重捕”：
        直接复用光流+特征匹配的跟踪逻辑尝试找回目标。
        成功返回 True，失败返回 False。
        """
        return self._track_features(current_gray, detection_bbox=None)

    def _attempt_reacquisition(self, current_gray, detection_bbox):
        """尝试重新捕获目标"""
        # 计算检测框与上次位置的距离
        det_center = self._get_center(detection_bbox)
        distance = np.linalg.norm(np.array(det_center) - np.array(self.center))

        # 如果距离合理，尝试重新初始化
        max_distance = max(self.size) * 2
        if distance < max_distance:
            self.bbox = detection_bbox
            self.center = det_center
            self.size = self._get_size(detection_bbox)
            self._extract_initial_features(current_gray)
            # 同步显示框（避免重捕后一两帧框跳变）
            H, W = current_gray.shape[:2]
            inflated = self._inflate_bbox(self.bbox, self.disp_inflate, self.disp_pad, (H, W))
            self.display_bbox = inflated
            return True

        return False

    def get_state_color(self):
        """获取跟踪状态对应的颜色（保留，但绘制时统一用绿）"""
        if self.state == "TRACKING":
            return (0, 255, 0)  # 绿色
        elif self.state == "LOST":
            return (0, 165, 255)  # 橙色
        else:
            return (0, 0, 255)  # 红色


class DroneTrackingSystem:
    """无人机跟踪系统主类"""

    def __init__(self, model_path):
        """最后三条约束适当更改"""
        self.model = YOLO(model_path)
        self.trackers = {}
        self.next_track_id = 1
        self.max_track_distance = 100
        self.uav_class_ids = {0}  # 无人机类别ID；不是0就改
        self.min_area = 10 * 10   # 太小的框直接丢
        self.max_area_ratio = 0.25  # 超过画面25%的巨框丢弃
        self.ar_range = (0.4, 2.5)  # 长宽比限制

        self.scale_step_clip = (0.88, 1.12)  # 单帧尺度允许 [−12%, +12%]
        self.max_box_area_ratio = 0.12  # 显示框最多占画面 12% 面积
        self.min_box_side = 8  # 最小边，避免被夹到 0

        # —— 可视化控制 ——
        self.show_dets = False          # 不显示检测框（避免黄↔绿频闪）
        self.show_features = False
        self.track_color = (0, 255, 0)  # 统一绿色展示确认后的轨迹

        # —— 统计信息显示控制 ——
        self.show_stats = True  # 想干净就设 False
        self.stats_period_sec = 0.25  # 每0.25s刷新一次信息
        self.stats_every_n = 15  # 每 15 帧更新一次（30fps ≈ 0.5s 刷新）
        self._cached_stats = ["", "", ""]
        self._last_stats_update = -1
        self._frame_index = 0

        # —— ID 复用池（短期记忆） ——
        self.recent_dead = []  # 列表元素: dict(id, bbox, center, frame_idx)
        self.reuse_ttl = 90  # 允许在最近 90 帧内复活同一ID（按需要调）

    def _remember_dead_tracker(self, tracker):
        self.recent_dead.append({
            "id": tracker.track_id,
            "bbox": tracker.bbox,
            "center": tracker.center,
            "frame_idx": self._frame_index
        })
        # 清理过期
        self.recent_dead = [d for d in self.recent_dead if self._frame_index - d["frame_idx"] <= self.reuse_ttl]

    def _reuse_dead_id_if_possible(self, bbox, frame):
        """
        若有近邻的 DEAD 轨迹，则复用其 ID；返回 True 表示已用旧号复活。
        复用条件：IoU >= 0.15  或  中心距 < 0.06 * 图像对角线
        """
        if not self.recent_dead:
            return False

        H, W = frame.shape[:2]
        diag = float(np.hypot(W, H))  # 图像对角线长度
        det_center = self._get_bbox_center(bbox)

        best, best_score = None, -1.0
        for d in self.recent_dead:
            iou = self._iou(d["bbox"], bbox)
            dist = np.linalg.norm(np.array(d["center"]) - np.array(det_center))
            score = iou - 0.001 * dist  # 先看 IoU，再轻微惩罚距离
            # 放宽复用门限：IoU 或 距离满足其一即可
            if score > best_score and (iou >= 0.15 or dist <= 0.06 * diag):
                best, best_score = d, score

        if best is None:
            return False

        reused_id = best["id"]
        tracker = DroneTracker(reused_id, bbox, frame)
        self.trackers[reused_id] = tracker
        # 从复用池移除该ID
        self.recent_dead = [d for d in self.recent_dead if d["id"] != reused_id]
        print(f"复用旧ID创建跟踪器: ID {reused_id}")
        return True

    def process_video(self, input_path, output_path, conf_threshold=0.5):
        """处理视频文件"""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {input_path}")

        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 自适应统计刷新步长
        self.stats_every_n = max(1, int(fps * self.stats_period_sec))
        # 自适应服用时间
        self.reuse_ttl = max(60, int(fps * 4)) # 约3秒内优先复用旧ID

        diag = (width ** 2 + height ** 2) ** 0.5
        self.max_track_distance = max(self.max_track_distance, int(0.10 * diag))  # 允许更大位移


        print(f"视频信息: {width}x{height}, FPS: {fps}, 总帧数: {total_frames}")

        # 设置输出视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                self._frame_index += 1
                if not ret:
                    break

                frame_count += 1
                print(f"处理帧: {frame_count}/{total_frames}", end='\r')

                # YOLO检测（加上类与更紧的iou）
                detections = self.model(frame, conf=conf_threshold, iou=0.45,
                                        classes=list(self.uav_class_ids), verbose=False)[0]

                detection_boxes = []
                if detections.boxes is not None:
                    boxes = detections.boxes.xyxy.cpu().numpy().astype(int)
                    confidences = detections.boxes.conf.cpu().numpy()
                    classes = detections.boxes.cls.cpu().numpy().astype(int)

                    H, W = frame.shape[:2]
                    max_area = self.max_area_ratio * W * H
                    min_area_px = max(self.min_area, int(0.00006 * W * H))

                    for box, conf, cls in zip(boxes, confidences, classes):
                        if cls not in self.uav_class_ids:
                            continue
                        x1, y1, x2, y2 = box
                        w, h = x2 - x1, y2 - y1
                        area = w * h
                        if area < min_area_px or area > max_area:
                            continue
                        ar = w / max(h, 1)
                        if not (self.ar_range[0] <= ar <= self.ar_range[1]):
                            continue

                        detection_boxes.append({'bbox': box, 'confidence': float(conf), 'cls': int(cls)})

                # 更新跟踪器
                self._update_trackers(frame, detection_boxes)

                # 绘制结果
                annotated_frame = self._draw_tracking_results(frame, detection_boxes)

                out.write(annotated_frame)

        except KeyboardInterrupt:
            print("\n处理过程被用户中断")

        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()

        print(f"\n处理完成! 输出视频已保存至: {output_path}")

    def _update_trackers(self, frame, detections):
        active_trackers = {}
        used_det_idx = set()  # 记录已被某轨迹占用的检测索引

        # 先更新现有轨迹
        for track_id, tracker in self.trackers.items():
            det_idx, det_bbox = self._assign_detection_to_tracker(tracker, detections, used_det_idx)
            if tracker.update(frame, det_bbox):
                active_trackers[track_id] = tracker
                if det_idx is not None:
                    used_det_idx.add(det_idx)
            else:
                # 如果 DEAD，记录到“可复用ID池”
                if tracker.state == "DEAD":
                    self._remember_dead_tracker(tracker)

        self.trackers = active_trackers

        # 2) 对“未被任何轨迹使用”的检测，优先复用旧ID，否则新建
        unassigned = self._get_unassigned_detections(detections, used_det_idx)
        for idx, det in unassigned:
            if self._reuse_dead_id_if_possible(det['bbox'], frame):
                used_det_idx.add(idx)  # ← 复用后也标记为已使用
            else:
                self._create_new_tracker(det['bbox'], frame)
                used_det_idx.add(idx)  # ← 新建后标记为已使用

    def _iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        ua = (ax2 - ax1) * (ay2 - ay1)
        ub = (bx2 - bx1) * (by2 - by1)
        return inter / (ua + ub - inter + 1e-6)

    def _assign_detection_to_tracker(self, tracker, detections, used_det_idx):
        if not detections:
            return None, None

        best_idx, best_bbox, best_score = None, None, -1.0

        for idx, det in enumerate(detections):
            if idx in used_det_idx:
                continue
            det_bbox = det['bbox']
            iou = self._iou(tracker.bbox, det_bbox)
            if iou >= 0.1:
                score = 1.0 + iou
            else:
                det_center = self._get_bbox_center(det_bbox)
                dist = np.linalg.norm(np.array(tracker.center) - np.array(det_center))
                if dist > self.max_track_distance:
                    continue
                score = 1.0 / (1.0 + dist)

            if score > best_score:
                best_score = score
                best_idx = idx
                best_bbox = det_bbox

        return best_idx, best_bbox

    def _get_unassigned_detections(self, detections, used_det_idx):
        return [(idx, det) for idx, det in enumerate(detections) if idx not in used_det_idx]

    # def _get_unassigned_detections(self, detections):
    #     """获取未分配的检测框"""
    #     unassigned = []
    #     for det in detections:
    #         assigned = False
    #         for trk in self.trackers.values():
    #             if trk.state in ("TRACKING","LOST"):
    #                 if self._iou(trk.bbox, det['bbox']) > 0.1:
    #                     assigned = True
    #                     break
    #                 cdist = np.linalg.norm(np.array(trk.center) -
    #                                        np.array(self._get_bbox_center(det['bbox'])))
    #                 if cdist < self.max_track_distance:
    #                     assigned = True
    #                     break
    #         if not assigned:
    #             unassigned.append(det)
    #     return unassigned

    def _create_new_tracker(self, bbox, frame):
        """创建新的跟踪器"""
        tracker = DroneTracker(self.next_track_id, bbox, frame)
        self.trackers[self.next_track_id] = tracker
        print(f"创建新跟踪器: ID {self.next_track_id}")
        self.next_track_id += 1

    def _get_bbox_center(self, bbox):
        """获取边界框中心点"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _draw_tracking_results(self, frame, detections):
        """绘制跟踪结果"""
        # （可选）绘制检测框：默认关闭，避免黄↔绿频闪
        if self.show_dets:
            for detection in detections:
                bbox = detection['bbox']
                conf = detection['confidence']
                x1, y1, x2, y2 = bbox

                # 绘制检测框（虚线）
                self._draw_dashed_rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

                # 检测标签
                det_label = f"Det: {conf:.2f}"
                cv2.putText(frame, det_label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 绘制跟踪框
        for tracker in self.trackers.values():
            self._draw_tracker(frame, tracker)

        # 绘制统计信息
        self._draw_statistics(frame)

        return frame

    def _draw_tracker(self, frame, tracker):
        """绘制单个跟踪器：仅显示稳定的绿色框"""
        # 只显示“已确认”的轨迹，并容忍短暂未更新，避免闪烁
        if tracker.hits < tracker.min_hits:
            return
        if tracker.time_since_update > tracker.hold_frames:
            return

        # 优先使用显示框（放大+平滑），没有则回退真实bbox
        box = tracker.display_bbox if tracker.display_bbox is not None else tracker.bbox
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0)  # 统一绿色

        H, W = frame.shape[:2]
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return
        # 任何一边超过画面 1/3，判为异常，不画（也不会闪大框）
        if w > 0.33 * W or h > 0.33 * H:
            return

        # 绘制跟踪框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # 绘制中心点（用显示框中心）
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        cv2.circle(frame, (cx, cy), 5, color, -1)

        # 绘制跟踪ID（统一用同一颜色）
        label = f"ID:{tracker.track_id}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - text_height - 10),
                      (x1 + text_width, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # （可选）显示少量特征点（系统级开关）
        if self.show_features and tracker.state == "TRACKING" and tracker.feature_db.positive_features:
            for kp in tracker.feature_db.positive_features[-20:]:
                pt = (int(kp.pt[0]), int(kp.pt[1]))
                cv2.circle(frame, pt, 2, self.track_color, -1)

    def _draw_dashed_rectangle(self, frame, pt1, pt2, color, thickness):
        """绘制虚线矩形"""
        x1, y1 = pt1
        x2, y2 = pt2

        dash_length = 10

        # 顶边
        for x in range(x1, x2, dash_length * 2):
            cv2.line(frame, (x, y1), (min(x + dash_length, x2), y1), color, thickness)

        # 底边
        for x in range(x1, x2, dash_length * 2):
            cv2.line(frame, (x, y2), (min(x + dash_length, x2), y2), color, thickness)

        # 左边
        for y in range(y1, y2, dash_length * 2):
            cv2.line(frame, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)

        # 右边
        for y in range(y1, y2, dash_length * 2):
            cv2.line(frame, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)

    def _draw_statistics(self, frame):
        if not self.show_stats:
            return

        # 仅每 N 帧更新一次数据
        if (self._frame_index - self._last_stats_update) >= self.stats_every_n:
            # “可见”轨迹：与 _draw_tracker 的判定一致
            visible_count = sum(
                1 for t in self.trackers.values()
                if t.hits >= t.min_hits and t.time_since_update <= t.hold_frames
            )
            tracking_count = sum(1 for t in self.trackers.values() if t.state == "TRACKING")
            lost_count = sum(1 for t in self.trackers.values() if t.state == "LOST")
            total_count = len(self.trackers)

            # 与视觉一致的三行
            self._cached_stats = [
                f"Total Drones: {total_count}",
                f"Visible: {visible_count}",  # 新增，可见数量 = 屏上应该看到的框
                f"Lost: {lost_count}"
            ]
            self._last_stats_update = self._frame_index

        stats_text = self._cached_stats

        # 以下保留你原来的绘制框背景的逻辑
        max_width = 0
        total_height = 0
        for text in stats_text:
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            max_width = max(max_width, w)
            total_height += h + 5

        cv2.rectangle(frame, (10, 10), (max_width + 20, total_height + 20), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (max_width + 20, total_height + 20), (255, 255, 255), 2)

        y_offset = 35
        for text in stats_text:
            cv2.putText(frame, text, (15, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 25


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOv8n无人机检测与高级特征跟踪系统')
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='YOLOv8n模型文件路径 (best.pt)')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='输入视频文件路径 (.mp4)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='输出视频文件路径 (.mp4)')
    parser.add_argument('--conf', '-c', type=float, default=0.8,
                        help='置信度阈值 (默认: 0.8)')

    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"模型文件不存在: {args.model}")

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入视频文件不存在: {args.input}")

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tracking_system = DroneTrackingSystem(args.model)
    tracking_system.process_video(args.input, args.output, args.conf)


def example_usage():
    """使用示例"""
    model_path = "best.pt"
    input_video = "input_video.mp4"
    output_video = "output_result_tracking.mp4"

    if not os.path.exists(model_path):
        print(f"请确保模型文件存在: {model_path}")
        return

    if not os.path.exists(input_video):
        print(f"请确保输入视频文件存在: {input_video}")
        return

    tracking_system = DroneTrackingSystem(model_path)
    tracking_system.process_video(input_video, output_video, conf_threshold=0.5)


if __name__ == "__main__":
    # example_usage()
    main()

"""
高级无人机跟踪系统说明:

主要功能:
1. 基于YOLOv8n的无人机检测
2. 多目标特征跟踪
3. 正负样本特征数据库管理
4. 光流与全局特征融合跟踪
5. 自动跟踪丢失检测与重捕获

核心算法实现:
- SIFT特征提取和描述子计算
- Lucas-Kanade光流跟踪
- 特征匹配与融合
- 尺度和旋转变换估计
- 跟踪状态自动管理

跟踪状态:
- TRACKING: 正常跟踪状态
- LOST: 跟踪丢失，正在尝试重捕获
- DEAD: 跟踪器已失效

可视化策略:
- 只显示“已确认”的绿色跟踪框，检测框默认不显示
- 显示框做放大+EMA平滑+1–2帧保留，避免频闪，视觉更稳

依赖:
pip install ultralytics opencv-python numpy
"""
