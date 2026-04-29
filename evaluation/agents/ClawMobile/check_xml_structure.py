"""
详细检查 XML 文件内容
"""
import os

xml_path = '../../../data/douyin-20/douyin_tc1_b_1.xml'
if os.path.exists(xml_path):
    # 用二进制模式读取
    with open(xml_path, 'rb') as f:
        raw = f.read(1000)
    print(f"前 1000 bytes (hex):\n{raw[:100].hex()}")
    print(f"\n前 1000 bytes (repr):\n{repr(raw[:200])}")

    # 检测编码
    if raw.startswith(b'\xef\xbb\xbf'):
        print("\n检测到 UTF-8 BOM")
        encoding = 'utf-8-sig'
    elif raw.startswith(b'\xff\xfe'):
        print("\n检测到 UTF-16 LE BOM")
        encoding = 'utf-16-le'
    elif raw.startswith(b'\xfe\xff'):
        print("\n检测到 UTF-16 BE BOM")
        encoding = 'utf-16-be'
    else:
        encoding = 'utf-8'
        print(f"\n默认编码: {encoding}")

    # 用正确编码读取
    with open(xml_path, 'r', encoding=encoding) as f:
        content = f.read()

    print(f"\n文件大小: {len(content)} 字符")
    print(f"前 500 字符:\n{content[:500]}")

    # 查找第一个非空白字符
    first_non_ws = None
    for i, c in enumerate(content):
        if not c.isspace():
            first_non_ws = i
            break
    if first_non_ws is not None:
        print(f"\n第一个非空白字符在位置 {first_non_ws}: '{content[first_non_ws]}'")
        print(f"从该位置开始的内容:\n{content[first_non_ws:first_non_ws+200]}")
else:
    print(f"文件不存在: {xml_path}")
