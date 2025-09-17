import cv2
import csv
from datetime import datetime, timezone, timedelta
import numpy as np

class GPXVideoOverlay:
    """
    A class for overlaying GPX data onto a video.
    """

    def __init__(self, csv_path="aligned.csv"):
        self.gpx_data = self._load_aligned_csv(csv_path)
        self.icons = self._preload_icons()
        self.font = cv2.FONT_HERSHEY_SIMPLEX

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

        return {
            "times": np.array(times, dtype=np.float32),
            "lats": np.array(lats, dtype=np.float32),
            "lons": np.array(lons, dtype=np.float32),
            "eles": np.array(eles, dtype=np.float32),
            "hrs": np.array(hrs, dtype=np.float32),
            "cads": np.array(cads, dtype=np.float32),
            "date_r": date_r,
        }

    def _preload_icons(self):
        icon_paths = {
            "time": "icons/time.png",
            "alt": "icons/alt.png",
            "hr": "icons/hr.png",
            "cad": "icons/cad.png",
        }
        icons = {}
        for name, path in icon_paths.items():
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
        ele = data["eles"][t_idx]
        hr = data["hrs"][t_idx]
        cad = data["cads"][t_idx]
        gpx_time_iso = data["date_r"][t_idx]

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

        elements = [
            ("time", gpx_time_iso.strftime("%m.%d %H:%M:%S"), (255, 255, 255)),
            ("alt", f"{ele:.1f}m", (255, 255, 255)),
            ("hr", f"{hr:.0f}bpm", hr_color),
            ("cad", f"{cad*2:.0f}spm", (255, 255, 255)),
        ]

        element_widths = []
        icon_text_gap = int(10 * scale)

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
        "video_data/DJI_20250916214848_0375_D.MP4", "03.mp4")