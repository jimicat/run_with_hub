import cv2
from datetime import datetime, timezone, timedelta
import numpy as np
import os
import subprocess
import tempfile
import json  # 用于处理 ffprobe 输出
import gpxpy # 用于解析 GPX 文件

# 从 aligned.py 移植的函数
def _get_video_start_time(video_path: str) -> datetime:
    """
    获取视频文件的创建时间（UTC）。
    使用 ffprobe 读取 format_tags:creation_time。
    """
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_entries", "format_tags=creation_time", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    try:
        metadata = json.loads(result.stdout)
        creation_time_str = metadata.get("format", {}).get("tags", {}).get("creation_time")
        
        if creation_time_str:
            # 转换成 datetime 对象（UTC 时间），处理 ISO 格式中的 'Z'
            return datetime.fromisoformat(creation_time_str.replace("Z", "+00:00"))
        else:
            raise KeyError("creation_time tag not found in video metadata.")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error reading video metadata ({e}). Check if ffprobe is installed and video has time tag.")
        raise RuntimeError("Failed to determine video start time.") from e


def _parse_gpx(gpx_path: str):
    """解析 GPX 文件，返回 (time, lat, lon, ele, hr, cad) 列表"""
    with open(gpx_path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)

    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                hr, cad = 0.0, 0.0 # 默认值设为 0.0
                if point.extensions:
                    for ext in point.extensions:
                        for child in ext:
                            tag = child.tag.lower()
                            if tag.endswith("hr"): # 匹配 <gpxtpx:hr>
                                try:
                                    hr = float(child.text)
                                except (ValueError, TypeError):
                                    pass
                            elif tag.endswith("cad"): # 匹配 <gpxtpx:cad>
                                try:
                                    cad = float(child.text)
                                except (ValueError, TypeError):
                                    pass

                points.append((
                    point.time,
                    point.latitude,
                    point.longitude,
                    point.elevation,
                    hr,
                    cad
                ))
    return points


class GPXVideoOverlay:
    """
    A class for overlaying GPX data onto a video.
    """

    def __init__(self, gpx_path: str, video_path: str, offset_seconds: int = 0, 
                 map_position="topright", map_scale=0.2, layout="default"):
        
        # 1. 加载全量 GPX 轨迹点
        self.gpx_points_full = _parse_gpx(gpx_path) 
        
        # 2. 准备用于 HUD 的对齐数据，并确定视频在完整路径中的起始索引
        self.gpx_data_metrics, self.video_start_idx = self._prepare_gpx_data(
            video_path, offset_seconds)
            
        # 3. 存储全路径的经纬度数据，用于地图绘制
        self.full_lats = np.array([p[1] for p in self.gpx_points_full], dtype=np.float32)
        self.full_lons = np.array([p[2] for p in self.gpx_points_full], dtype=np.float32)

        self.icons = self._preload_icons()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.map_position = map_position
        self.map_scale = map_scale        
        self.layout = layout


    def _prepare_gpx_data(self, video_path: str, offset_seconds: int = 0):
        """
        加载并对齐 GPX 数据，只保留与视频时间重叠的部分用于计算指标。
        返回 (对齐的指标数据, 视频在完整路径中的起始索引)。
        """
        video_start = _get_video_start_time(video_path)
        gpx_points = self.gpx_points_full # 使用 __init__ 中加载的全量数据

        times, lats, lons, eles, hrs, cads, date_r = [], [], [], [], [], [], []
        video_start_idx = -1 # 视频在 full gpx points 中的起始索引

        for i, (t, lat, lon, ele, hr, cad) in enumerate(gpx_points):
            # 计算 GPX 时间相对于视频开始时间（加上偏移）的秒数
            delta = (t - video_start).total_seconds() + offset_seconds
            
            if delta >= 0:
                # 记录视频开始时对应的全路径索引
                if video_start_idx == -1:
                    video_start_idx = i

                times.append(delta)
                lats.append(lat)
                lons.append(lon)
                eles.append(ele)
                hrs.append(hr)
                cads.append(cad)

                # 将时间转换为 UTC+8 并存储
                dt = t.astimezone(timezone(timedelta(hours=8)))
                date_r.append(dt)

        # ----------------------------------------------------
        # 计算累计距离 (Cumulative Distance) - 基于对齐后的数据
        # ----------------------------------------------------
        distances = [0.0]
        # 这里的 lats/lons 已经是视频时间段内的子集
        lats_np = np.array(lats, dtype=np.float32)
        lons_np = np.array(lons, dtype=np.float32)

        for i in range(1, len(lats_np)):
            dist = self._haversine(lats_np[i-1], lons_np[i-1], lats_np[i], lons_np[i])
            distances.append(distances[-1] + dist)

        metrics_data = {
            "times": np.array(times, dtype=np.float32),
            "lats": lats_np,
            "lons": lons_np,
            "eles": np.array(eles, dtype=np.float32),
            "hrs": np.array(hrs, dtype=np.float32),
            "cads": np.array(cads, dtype=np.float32),
            "date_r": date_r,
            "distances": np.array(distances, dtype=np.float32),
        }
        
        return metrics_data, video_start_idx


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

        a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * \
            np.cos(phi2) * np.sin(delta_lambda / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c
    
    # ... (_draw_date, _preload_icons, _overlay_icon 保持不变)
    def _draw_date(self, frame, gpx_time_iso):
        """单独在左上角绘制日期和时间"""
        h_frame, w_frame = frame.shape[:2]

        base_width = 1280.0
        scale = w_frame / base_width
        icon_size = (int(40 * scale), int(40 * scale))
        font_scale = 1.0 * scale
        thickness = max(1, int(2 * scale))
        icon_text_gap = int(10 * scale)

        text = gpx_time_iso.strftime("%Y-%m-%d %H:%M:%S")

        margin = int(20 * scale)
        # 左上角基准点
        icon_x, icon_y = margin, margin

        # 绘制日期图标
        self._overlay_icon(frame, "time", (icon_x, icon_y), icon_size)

        # 绘制日期文本（紧随其右）
        (text_w, text_h), _ = cv2.getTextSize(text, self.font, font_scale, thickness)
        text_x = icon_x + icon_size[0] + icon_text_gap
        text_y = icon_y + icon_size[1] // 2 + text_h // 2

        cv2.putText(frame, text, (text_x, text_y),
                    self.font, font_scale, (255, 255, 255), thickness)

        return frame

    def _preload_icons(self):
        icon_paths = {
            "time": "icons/icon_02/time.png",
            "pace": "icons/icon_02/pace.png",
            "alt": "icons/icon_02/alt.png",
            "hr": "icons/icon_02/hr.png",
            "cad": "icons/icon_02/cad.png",
        }
        icons = {}
        for name, path in icon_paths.items():
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
            roi[:, :, c] = roi[:, :, c] * \
                (1 - mask) + resized_icon[:, :, c] * mask

        frame[y:y_end, x:x_end] = roi
        return frame
    # ...


    def _draw_path(self, frame, t_idx):
        """
        在视频画面上绘制跑步路径小地图。
        ***使用完整的 GPX 路径来确定地图边界和绘制轨迹线。***
        """
        h_frame, w_frame = frame.shape[:2]
        
        # 使用完整的经纬度数据来绘制全路径
        lats = self.full_lats
        lons = self.full_lons

        # ==== 1. 归一化坐标到小地图区域 (基于完整路径) ====
        min_lat, max_lat = np.min(lats), np.max(lats)
        min_lon, max_lon = np.min(lons), np.max(lons)

        # 小地图大小 (随视频宽度缩放)
        map_size = int(w_frame * self.map_scale)
        margin = int(20 * (w_frame / 1280))  # 跟随画面缩放

        # 根据 map_position 设置位置
        if self.map_position == "topleft":
            map_x1, map_y1 = margin, margin
        # ... (其他位置判断保持不变)
        elif self.map_position == "topright":
            map_x1, map_y1 = w_frame - map_size - margin, margin
        elif self.map_position == "bottomleft":
            map_x1, map_y1 = margin, h_frame - map_size - margin
        elif self.map_position == "bottomright":
            map_x1, map_y1 = w_frame - map_size - margin, h_frame - map_size - margin
        else:
            map_x1, map_y1 = w_frame - map_size - margin, margin

        map_x2, map_y2 = map_x1 + map_size, map_y1 + map_size

        # 避免除零
        lat_range = max(max_lat - min_lat, 1e-6)
        lon_range = max(max_lon - min_lon, 1e-6)

        # 将全路径的经纬度映射到小地图区域
        norm_x = (lons - min_lon) / lon_range
        norm_y = (lats - min_lat) / lat_range
        norm_y = 1.0 - norm_y  # Y 轴翻转

        pts = np.stack([
            map_x1 + norm_x * map_size,
            map_y1 + norm_y * map_size
        ], axis=-1).astype(int)

        # ==== 2. 绘制完整轨迹 ====
        # 使用完整的 pts 数组绘制轨迹
        for i in range(1, len(pts)):
            cv2.line(frame, tuple(pts[i-1]), tuple(pts[i]), (0, 255, 255), 3)

        # ==== 3. 绘制当前位置 ====
        # 将视频对齐后的索引 t_idx 映射到完整路径的索引
        # self.video_start_idx 是视频开始对应的全路径索引
        full_path_idx = self.video_start_idx + t_idx
        
        # 确保索引在范围内
        if full_path_idx < 0 or full_path_idx >= len(pts):
            # 如果视频帧跑到 GPX 结束之后，标记点会消失，这是可以接受的
            return frame 
            
        cur_pt = tuple(pts[full_path_idx])
        cv2.circle(frame, cur_pt, 6, (0, 0, 255), -1)  # 红色圆点

        return frame

    def _draw_hud(self, frame, t_idx):
        h_frame, w_frame = frame.shape[:2]
        base_width = 1280.0
        scale = w_frame / base_width
    
        icon_size = (int(40 * scale), int(40 * scale))
        font_scale = 1.0 * scale
        thickness = max(int(2 * scale), 1)
    
        # ***注意：这里必须使用 self.gpx_data_metrics（对齐后的数据）***
        data = self.gpx_data_metrics
        ele = data["eles"][t_idx]
        hr = data["hrs"][t_idx]
        cad = data["cads"][t_idx]
        gpx_time_iso = data["date_r"][t_idx]
        current_time_sec = data["times"][t_idx]
        current_dist_m = data["distances"][t_idx]
    
        # ---- 计算配速 (Pace) ----
        pace_str = "--:--"
        if current_dist_m > 10 and current_time_sec > 0.5:
            speed_mps = current_dist_m / current_time_sec
            if speed_mps > 0.01: 
                speed_kmh = speed_mps * 3.6 
                pace_sec_per_km = 3600 / speed_kmh
                pace_min = int(pace_sec_per_km // 60)
                pace_sec = int(pace_sec_per_km % 60)
                pace_str = f"{pace_min:02d}'{pace_sec:02d}\""
    
        elements = [
            ("pace", pace_str, (255, 255, 255)),
            ("alt", f"{ele:.1f}m", (255, 255, 255)),
            ("hr", f"{hr:.0f}bpm" if hr > 0 else "--", (0, 255, 255) if hr > 0 else (255, 255, 255)),
            ("cad", f"{cad*2:.0f}spm" if cad > 0 else "--", (0, 255, 255) if cad > 0 else (255, 255, 255)),
        ]
    
        if self.layout == "default":
            # ... (底部横向布局绘制逻辑保持不变)
            bg_height = int(80 * scale)
            overlay = np.zeros_like(frame, dtype=np.uint8)
            cv2.rectangle(overlay, (0, h_frame - bg_height),
                          (w_frame, h_frame), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
            y_center_px = h_frame - bg_height // 2
            icon_text_gap = int(10 * scale)
    
            element_widths = []
            for _, text, _ in elements:
                (text_w, _), _ = cv2.getTextSize(text, self.font, font_scale, thickness)
                element_total_width = icon_size[0] + icon_text_gap + text_w
                element_widths.append(element_total_width)
    
            total_elements_width = sum(element_widths)
            padding_px = int(w_frame * 0.03)
            available_width = w_frame - 2 * padding_px
            total_gap_width = available_width - total_elements_width
            avg_gap = total_gap_width / (len(elements) - 1) if len(elements) > 1 else 0
    
            current_x = float(padding_px)
            for i, (icon_name, text, text_color) in enumerate(elements):
                icon_y = int(y_center_px - icon_size[1] / 2)
                self._overlay_icon(frame, icon_name, (int(current_x), icon_y), icon_size)
                (text_w, text_h), _ = cv2.getTextSize(text, self.font, font_scale, thickness)
                text_x = int(current_x) + icon_size[0] + icon_text_gap
                text_y = int(y_center_px + text_h / 2)
                cv2.putText(frame, text, (text_x, text_y), self.font,
                            font_scale, text_color, thickness)
                if i < len(elements) - 1:
                    current_x += element_widths[i] + avg_gap
    
        elif self.layout == "grid9":
            # ... (九宫格布局绘制逻辑保持不变)
            cell_h = h_frame // 3
            x0, y0 = 0, 2 * cell_h 
    
            padding = int(10 * scale)
            cur_y = y0 + padding
    
            for icon_name, text, text_color in elements:
                self._overlay_icon(frame, icon_name, (x0 + padding, cur_y), icon_size)
                (text_w, text_h), _ = cv2.getTextSize(text, self.font, font_scale, thickness)
                text_x = x0 + padding + icon_size[0] + 10
                text_y = cur_y + icon_size[1] // 2 + text_h // 2
                cv2.putText(frame, text, (text_x, text_y), self.font,
                            font_scale, text_color, thickness)
                cur_y += icon_size[1] + padding
    
        # ---- 路径和日期保持绘制 ----
        frame = self._draw_path(frame, t_idx)
        frame = self._draw_date(frame, gpx_time_iso)
        return frame
    

    def find_nearest_index(self, t):
        """查找距离给定视频时间 t 最近的 对齐后 GPX 数据点索引"""
        # ***注意：这里必须使用 self.gpx_data_metrics["times"]***
        times = self.gpx_data_metrics["times"]
        idx = np.searchsorted(times, t, side='left')

        if idx == 0:
            return 0
        if idx >= len(times):
            return len(times) - 1

        if abs(times[idx] - t) < abs(times[idx - 1] - t):
            return idx
        else:
            return idx - 1

    def process_video(self, video_path, output_path, max_duration=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
    
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_file_path = tmp_file.name
        tmp_file.close()
    
        out = cv2.VideoWriter(tmp_file_path, fourcc, fps, (width, height))
    
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
        if max_duration is not None:
            max_frames = int(fps * max_duration)
            frame_count = min(frame_count, max_frames)
    
        print(f"Start rendering video at {fps} FPS. Total frames to process: {frame_count}")

        for frame_idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
    
            t = frame_idx / fps
            # idx 是对齐后数据（metrics）的索引
            idx = self.find_nearest_index(t)
    
            frame = self._draw_hud(frame, idx)
            out.write(frame)
    
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{frame_count} frames...")
    
        cap.release()
        out.release()
        print(f"Rendering complete. Merging audio from original video...")
    
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", tmp_file_path, 
                "-i", video_path,   
                "-c:v", "copy",      
                "-c:a", "aac",       
                "-map", "0:v:0",     
                "-map", "1:a:0",     
                "-shortest",         
                output_path
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Processing complete. Video saved to: {output_path}")
        except subprocess.CalledProcessError as e:
            print("Error merging audio. Check if ffmpeg is installed and accessible.")
            print("FFmpeg stdout:", e.stdout.decode())
            print("FFmpeg stderr:", e.stderr.decode())
            print(f"Unmerged video saved to temporary path: {tmp_file_path}") 
            return
        
        os.remove(tmp_file_path)


if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # 路径配置
    video_file = "video/DJI_20250916214343_0373_D.MP4"
    gpx_file = "gpx/activity_20403771520.gpx"
    output_file = "output/001_full_path_overlay.mp4"
    time_offset = 0  # 如果需要人工修正，可以在这里设置偏移量（秒）
    # ----------------------------------------------------------------------

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    try:
        # 实例化时进行数据加载、对齐和全路径准备
        overlay_processor = GPXVideoOverlay(
            gpx_path=gpx_file,
            video_path=video_file,
            offset_seconds=time_offset, 
            map_position="topright",
            map_scale=0.2,
            layout="grid9"
        )

        overlay_processor.process_video(
            video_file,
            output_file,
            max_duration=10  
        )
    except RuntimeError as e:
        print(f"\n--- Critical Error ---")
        print(e)
        print("请检查视频文件是否存在，是否有正确的时间元数据，以及 'ffprobe' 是否已安装。")
    except FileNotFoundError as e:
        print(f"\n--- File Not Found Error ---")
        print(f"错误: {e}")
        print(f"请检查路径 '{video_file}'、'{gpx_file}' 和 'icons/' 文件夹是否正确。")