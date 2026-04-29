"""
调试 AndroidStateProvider 如何处理 a11y_tree
"""
import asyncio
import os
import xml.etree.ElementTree as ET


async def test():
    from mobilerun.tools.filters import ConciseFilter
    from mobilerun.tools.formatters import IndexedFormatter

    # 读取 XML 文件
    xml_path = '../../../data/douyin-20/douyin_tc1_b_1.xml'
    with open(xml_path, 'r', encoding='utf-8') as f:
        xml_content = f.read()

    # 解析为字典树
    root = ET.fromstring(xml_content)

    def node_to_dict(node):
        result = dict(node.attrib)
        children = [node_to_dict(child) for child in node]
        if children:
            result["children"] = children
        return result

    a11y_tree = node_to_dict(root)
    print(f"a11y_tree 根节点 keys: {list(a11y_tree.keys())}")
    print(f"a11y_tree 有 children: {'children' in a11y_tree}")
    if 'children' in a11y_tree:
        print(f"children 数量: {len(a11y_tree['children'])}")
        print(f"第一个 child keys: {list(a11y_tree['children'][0].keys())[:5]}")

    # 创建 filter 和 formatter
    tree_filter = ConciseFilter()
    tree_formatter = IndexedFormatter()

    device_context = {
        "screen_bounds": {"width": 1080, "height": 2400}
    }

    phone_state = {
        "foreground_app": "com.ss.android.ugc.aweme",
        "orientation": "portrait"
    }

    # 测试 filter
    print("\n=== 测试 ConciseFilter ===")
    try:
        filtered = tree_filter.filter(a11y_tree, device_context)
        print(f"filtered 类型: {type(filtered)}")
        if isinstance(filtered, dict):
            print(f"filtered keys: {list(filtered.keys())[:10]}")
            if 'children' in filtered:
                print(f"filtered children 数量: {len(filtered['children'])}")
        elif isinstance(filtered, list):
            print(f"filtered 是列表，长度: {len(filtered)}")
    except Exception as e:
        print(f"filter 错误: {e}")
        import traceback
        traceback.print_exc()
        filtered = a11y_tree

    # 测试 formatter
    print("\n=== 测试 IndexedFormatter ===")
    tree_formatter.screen_width = 1080
    tree_formatter.screen_height = 2400
    try:
        formatted_text, focused_text, elements, phone_state_result = tree_formatter.format(filtered, phone_state)
        print(f"formatted_text 长度: {len(formatted_text)}")
        print(f"elements 数量: {len(elements)}")
        if elements:
            print(f"第一个 element: {elements[0]}")
        else:
            print("elements 为空!")
            # 打印前200字符的 formatted_text
            print(f"formatted_text 前200字符:\n{formatted_text[:200]}")
    except Exception as e:
        print(f"formatter 错误: {e}")
        import traceback
        traceback.print_exc()


asyncio.run(test())
