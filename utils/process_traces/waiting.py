#!/usr/bin/env python3
"""
将图片1叠加到图片2中央/全屏，输出新图片
用法: python overlay_images.py <背景图片> <叠加图片> <输出路径>
"""

import sys
from PIL import Image


def overlay_center(background_path: str, overlay_path: str, output_path: str, scale: float = 2.0):
    """将overlay图片居中叠加到background上

    Args:
        background_path: 背景图片路径
        overlay_path: 叠加图片路径
        output_path: 输出路径
        scale: 叠加图片缩放比例，默认2倍
    """
    # 打开图片
    background = Image.open(background_path).convert("RGBA")
    overlay = Image.open(overlay_path).convert("RGBA")

    # 放大叠加图片
    ov_w, ov_h = overlay.size
    new_size = (int(ov_w * scale), int(ov_h * scale))
    overlay = overlay.resize(new_size, Image.Resampling.LANCZOS)
    print(f"叠加图片已放大至: {new_size[0]}x{new_size[1]}")

    # 计算居中位置
    bg_w, bg_h = background.size
    ov_w, ov_h = overlay.size

    # 如果overlay比background大，缩放overlay
    if ov_w > bg_w or ov_h > bg_h:
        scale = min(bg_w / ov_w, bg_h / ov_h)
        new_size = (int(ov_w * scale), int(ov_h * scale))
        overlay = overlay.resize(new_size, Image.Resampling.LANCZOS)
        ov_w, ov_h = new_size
        print(f"叠加图片已缩放至: {ov_w}x{ov_h}")

    # 计算居中坐标
    x = (bg_w - ov_w) // 2
    y = (bg_h - ov_h) // 2

    # 叠加
    background.paste(overlay, (x, y), overlay)

    # 保存
    # 如果输出格式不支持RGBA，转为RGB
    if output_path.lower().endswith(('.jpg', '.jpeg')):
        background = background.convert("RGB")

    background.save(output_path)
    print(f"完成! 输出: {output_path}")


def overlay_fullscreen(background_path: str, overlay_path: str, output_path: str):
    """将overlay图片拉伸到全屏叠加到background上

    Args:
        background_path: 背景图片路径
        overlay_path: 叠加图片路径
        output_path: 输出路径
    """
    # 打开图片
    background = Image.open(background_path).convert("RGBA")
    overlay = Image.open(overlay_path).convert("RGBA")

    # 获取背景尺寸
    bg_w, bg_h = background.size
    print(f"背景尺寸: {bg_w}x{bg_h}")

    # 将overlay拉伸到全屏
    overlay = overlay.resize((bg_w, bg_h), Image.Resampling.LANCZOS)
    print(f"叠加图片已拉伸至全屏: {bg_w}x{bg_h}")

    # 全屏叠加（从0,0开始）
    background.paste(overlay, (0, 0), overlay)

    # 保存
    if output_path.lower().endswith(('.jpg', '.jpeg')):
        background = background.convert("RGB")

    background.save(output_path)
    print(f"完成! 输出: {output_path}")


if __name__ == "__main__":
    # 居中叠加示例
    overlay_center("../traces_new/taobao-16/taobao_tc1_EA_3.png",
                   "./waiting.png",
                   "../traces_new/taobao-16/taobao_tc1_EA-T_3.png")

    # 全屏叠加示例
    # overlay_fullscreen("../traces_new/douyin_20/douyin_tc2_b_1.png", "./ad.png",
    #                    "../traces_new/douyin_20/douyin_tc2_FO_1.png")
