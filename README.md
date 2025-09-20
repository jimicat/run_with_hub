## 通过此脚本可以将运动数据叠加至视频上。

注意：此脚本还在开发中，不提供测试文档。
Custom font
-----------

You can pass a TrueType font file path and a base font size (pixels at 1280 width) when creating the `GPXVideoOverlay` instance:

```python
overlay_processor = GPXVideoOverlay(
	gpx_path=gpx_file,
	video_path=video_file,
	font_path="/path/to/your/font.ttf",
	base_font_size=28,
)
```

The script uses Pillow (PIL) when available to render the TTF font. If Pillow is not installed or the font cannot be loaded, the overlay falls back to OpenCV's built-in font rendering.

HUD Layouts
-----------

Two HUD layouts are available via the `layout` parameter when creating `GPXVideoOverlay`:

- `classic` (or `default`): Semi-transparent bottom bar with subtle border — conservative, high-contrast layout.
- `scifi`: Darker bar with per-element neon boxes, glow text and a futuristic look.

- `left` / `leftbottom`: A vertical stacked HUD at the left-bottom corner. Useful when you prefer a sidebar style HUD instead of a bottom bar.

Example:

```python
overlay_processor = GPXVideoOverlay(
	gpx_path=gpx_file,
	video_path=video_file,
	layout="scifi",
)
```