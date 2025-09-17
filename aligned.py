import subprocess
import json
import gpxpy
import datetime
import csv

def get_video_start_time(video_path: str) -> datetime.datetime:
    """获取视频文件的创建时间（UTC）"""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_entries", "format_tags=creation_time", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    metadata = json.loads(result.stdout)
    creation_time_str = metadata["format"]["tags"]["creation_time"]
    # 转换成 datetime 对象（UTC 时间）
    return datetime.datetime.fromisoformat(creation_time_str.replace("Z", "+00:00"))

def parse_gpx(gpx_path: str):
    """解析 GPX 文件，返回 (time, lat, lon, ele, hr, cad) 列表"""
    with open(gpx_path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)

    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                hr, cad = None, None
                if point.extensions:
                    for ext in point.extensions:
                        for child in ext:
                            tag = child.tag.lower()
                            if "hr" in tag:
                                hr = int(child.text)
                            elif "cad" in tag:
                                cad = int(child.text)

                points.append((
                    point.time,
                    point.latitude,
                    point.longitude,
                    point.elevation,
                    hr,
                    cad
                ))
    return points


def align_gpx_with_video(video_path: str, gpx_path: str, offset_seconds: int = 0):
    video_start = get_video_start_time(video_path)
    gpx_points = parse_gpx(gpx_path)

    aligned_data = []
    for t, lat, lon, ele, hr, cad in gpx_points:
        # 同时保留原始 GPX 时间字符串（ISO 格式）
        gpx_time_iso = t.isoformat()
        # 计算相对视频的秒数
        delta = (t - video_start).total_seconds() + offset_seconds
        if delta >= 0:
            aligned_data.append((gpx_time_iso, delta, lat, lon, ele, hr, cad))
    return aligned_data

if __name__ == "__main__":
    video_file = "video/DJI_20250916215214_0376_D.MP4"
    gpx_file = "gpx/activity_20403771520.gpx"
    offset = 0  # 如果需要人工修正，可以改这里

    aligned = align_gpx_with_video(video_file, gpx_file, offset)


    with open("aligned.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gpx_time_iso","video_time_sec", "lat", "lon", "ele", "hr", "cad"])
        writer.writerows(aligned)

    print("对齐完成，已输出 aligned.csv")
