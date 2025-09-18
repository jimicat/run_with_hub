import cv2
import csv
from datetime import datetime, timezone, timedelta
import numpy as np
import os # 导入 os 库用于检查图标文件

class GPXVideoOverlay:
    """
    A class for overlaying GPX data onto a video.
    """

    def __init__(self, csv_path="aligned.csv"):
        self.gpx_data = self._load_aligned_csv(csv_path)
        self.icons = self._preload_icons()
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2):
        """
        Calculate the distance (in meters) between two points on the earth
        (specified in decimal degrees) using the Haversine formula.
        """
        R = 6371e3  # earth radius in meters
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)

        a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    def _load_aligned_csv(self, csv_path):
        times, lats, lons, eles, hrs, cads, date_r = [], [], [], [], [], [], []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                times.append(float(row["video_time_sec"]))
                lats.append(float(row["lat"]))
                lons.append(float(row["lon"]))
                eles.append(float(row["ele"]))
                hrs.append(float(row["hr"]))
                cads.append(float(row["cad"]))
                # parse ISO string, ensure it's timezone-aware, then convert to UTC+8
                dt = datetime.fromisoformat(row["gpx_time_iso"])
                if dt.tzinfo is None:
                    # assume UTC if no tzinfo present
                    dt = dt.replace(tzinfo=timezone.utc)
                # convert to desired timezone (UTC+8)
                dt = dt.astimezone(timezone(timedelta(hours=8)))
                date_r.append(dt)

        # ----------------------------------------------------
        # 新增：计算累计距离 (Cumulative Distance)
        # ----------------------------------------------------
        distances = [0.0]
        for i in range(1, len(lats)):
            dist = self._haversine(lats[i-1], lons[i-1], lats[i], lons[i])
            distances.append(distances[-1] + dist)

        return {
            "times": np.array(times, dtype=np.float32),
            "lats": np.array(lats, dtype=np.float32),
            "lons": np.array(lons, dtype=np.float32),
            "eles": np.array(eles, dtype=np.float32),
            "hrs": np.array(hrs, dtype=np.float32),
            "cads": np.array(cads, dtype=np.float32),
            "date_r": date_r,
            # 将累计距离加入数据字典
            "distances": np.array(distances, dtype=np.float32),
        }

    def _preload_icons(self):
        icon_paths = {
            "time": "icons/time.png",
            "pace": "icons/pace.png", # 新增: 配速图标
            "alt": "icons/alt.png",
            "hr": "icons/hr.png",
            "cad": "icons/cad.png",
        }
        icons = {}
        for name, path in icon_paths.items():
            # 检查文件是否存在，防止 cv2.imread 失败
            if not os.path.exists(path):
                 print(f"Warning: Icon file not found at {path}. Display for {name} will be skipped.")
                 continue

            icon = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if icon is not None:
                icons[name] = icon
            else:
                print(f"Warning: Could not load icon {path}")
        return icons

    def _overlay_icon(self, frame, icon_name, pos, size):
        icon = self.icons.get(icon_name)
        if icon is None:
            return frame
        
        resized_icon = cv2.resize(icon, size, interpolation=cv2.INTER_AREA)

        h, w = resized_icon.shape[:2]
        x, y = pos
        y_end, x_end = y + h, x + w
        y_end = min(y_end, frame.shape[0])
        x_end = min(x_end, frame.shape[1])
        
        roi = frame[y:y_end, x:x_end]
        
        if roi.shape[0] != h or roi.shape[1] != w:
            resized_icon = resized_icon[:roi.shape[0], :roi.shape[1]]
            
        b, g, r, a = cv2.split(resized_icon)
        mask = a / 255.0
        
        for c in range(3):
            roi[:, :, c] = roi[:, :, c] * (1 - mask) + resized_icon[:, :, c] * mask

        frame[y:y_end, x:x_end] = roi
        return frame
    
    def _draw_path(self, frame, t_idx):
        """
        在视频画面上绘制跑步路径小地图，并标记当前位置
        """
        h_frame, w_frame = frame.shape[:2]
        data = self.gpx_data
        lats = data["lats"]
        lons = data["lons"]

        # ==== 1. 归一化坐标到小地图区域 ====
        min_lat, max_lat = np.min(lats), np.max(lats)
        min_lon, max_lon = np.min(lons), np.max(lons)

        # 小地图大小 (200x200 像素，右上角显示)
        map_size = 200
        margin = 20
        map_x1, map_y1 = w_frame - map_size - margin, margin
        map_x2, map_y2 = w_frame - margin, margin + map_size

        # 避免除零
        lat_range = max(max_lat - min_lat, 1e-6)
        lon_range = max(max_lon - min_lon, 1e-6)

        # 将经纬度映射到小地图区域
        norm_x = (lons - min_lon) / lon_range
        norm_y = (lats - min_lat) / lat_range
        # 注意 Y 轴翻转（图像坐标是下大上小）
        norm_y = 1.0 - norm_y  

        pts = np.stack([
            map_x1 + norm_x * map_size,
            map_y1 + norm_y * map_size
        ], axis=-1).astype(int)

        # ==== 2. 绘制轨迹 ====
        for i in range(1, len(pts)):
            cv2.line(frame, tuple(pts[i-1]), tuple(pts[i]), (0, 255, 255), 2)

        # ==== 3. 绘制当前位置 ====
        cur_pt = tuple(pts[t_idx])
        cv2.circle(frame, cur_pt, 6, (0, 0, 255), -1)  # 红色圆点

        # ==== 4. 绘制小地图边框 ====
        cv2.rectangle(frame, (map_x1, map_y1), (map_x2, map_y2), (255, 255, 255), 2)

        return frame


    def _draw_hud(self, frame, t_idx):
        """
        Using a flexible layout to distribute icons and text evenly.
        """
        h_frame, w_frame = frame.shape[:2]

        base_width = 1280.0
        scale = w_frame / base_width
        
        icon_size = (int(40 * scale), int(40 * scale))
        font_scale = 1.0 * scale
        thickness = int(2 * scale) if int(2 * scale) > 0 else 1
        
        data = self.gpx_data
        
        # 提取当前帧数据
        ele = data["eles"][t_idx]
        hr = data["hrs"][t_idx]
        cad = data["cads"][t_idx]
        gpx_time_iso = data["date_r"][t_idx]
        
        current_time_sec = data["times"][t_idx]
        current_dist_m = data["distances"][t_idx]

        # ----------------------------------------------------
        # 新增：计算配速 (Pace)
        # ----------------------------------------------------
        pace_str = "--:--"
        if current_dist_m > 0 and current_time_sec > 0:
            # 距离 (km) / 时间 (h) = 速度 (km/h)
            # 1 / 速度 (km/h) = 时间 (h) / 距离 (km) = 配速 (h/km)
            # 配速 (sec/km) = 3600 / 速度 (km/h)
            
            # 使用总距离/总时间计算平均配速
            speed_kmh = (current_dist_m / 1000) / (current_time_sec / 3600)
            
            if speed_kmh > 0.05: # 避免除以接近0的速度导致配速无限大
                pace_sec_per_km = 3600 / speed_kmh
                pace_min = int(pace_sec_per_km // 60)
                pace_sec = int(pace_sec_per_km % 60)
                pace_str = f"{pace_min:02d}'{pace_sec:02d}\"" # 格式化为 分'秒"
            else:
                pace_str = "--'--"


        bg_height = int(80 * scale)
        overlay = np.zeros_like(frame, dtype=np.uint8)
        cv2.rectangle(overlay, (0, h_frame - bg_height), (w_frame, h_frame), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
        
        y_center_px = h_frame - bg_height // 2
        
        if hr < 120:
            hr_color = (0, 255, 0)
        elif hr < 160:
            hr_color = (0, 255, 255)
        else:
            hr_color = (0, 0, 255)

        # ----------------------------------------------------
        # 新增：将配速添加到 HUD 元素列表
        # ----------------------------------------------------
        elements = [
            ("time", gpx_time_iso.strftime("%m.%d %H:%M:%S"), (255, 255, 255)),
            ("pace", pace_str, (255, 255, 255)), # 新增配速显示
            ("alt", f"{ele:.1f}m", (255, 255, 255)),
            ("hr", f"{hr:.0f}bpm", hr_color),
            ("cad", f"{cad*2:.0f}spm", (255, 255, 255)),
        ]

        element_widths = []
        icon_text_gap = int(10 * scale)

        # ... (后续的布局计算和绘图逻辑保持不变) ...
        for _, text, _ in elements:
            (text_w, _), _ = cv2.getTextSize(text, self.font, font_scale, thickness)
            element_total_width = icon_size[0] + icon_text_gap + text_w
            element_widths.append(element_total_width)

        total_elements_width = sum(element_widths)
        padding_px = int(w_frame * 0.03)
        available_width = w_frame - 2 * padding_px
        total_gap_width = available_width - total_elements_width
        
        if len(elements) > 1:
            avg_gap = total_gap_width / (len(elements) - 1)
        else:
            avg_gap = 0
        
        current_x = float(padding_px)
        for i, (icon_name, text, text_color) in enumerate(elements):
            icon_y = int(y_center_px - icon_size[1] / 2)
            self._overlay_icon(frame, icon_name, (int(current_x), icon_y), icon_size)

            (text_w, text_h), _ = cv2.getTextSize(text, self.font, font_scale, thickness)
            text_x = int(current_x) + icon_size[0] + icon_text_gap
            text_y = int(y_center_px + text_h / 2)
            
            cv2.putText(frame, text, (text_x, text_y), self.font, font_scale, text_color, thickness)
            
            if i < len(elements) - 1:
                current_x += element_widths[i] + avg_gap
        
        # 在 HUD 绘制完成后，叠加跑步路径
        frame = self._draw_path(frame, t_idx)
        
        return frame

    def find_nearest_index(self, t):
        times = self.gpx_data["times"]
        idx = np.searchsorted(times, t, side='left')
        
        if idx == 0:
            return 0
        if idx >= len(times):
            return len(times) - 1
        
        if abs(times[idx] - t) < abs(times[idx - 1] - t):
            return idx
        else:
            return idx - 1

    def process_video(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for frame_idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            
            t = frame_idx / fps
            idx = self.find_nearest_index(t)
            
            frame = self._draw_hud(frame, idx)
            out.write(frame)
            
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{frame_count} frames...")

        cap.release()
        out.release()
        print(f"\nProcessing complete. Video saved to: {output_path}")

if __name__ == "__main__":
    overlay_processor = GPXVideoOverlay(csv_path="aligned.csv")
    overlay_processor.process_video(
        "video/DJI_20250916215214_0376_D.MP4", "output/004.mp4")