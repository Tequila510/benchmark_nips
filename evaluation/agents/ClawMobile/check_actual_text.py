"""
检查 XML 中有 text 内容的节点
"""
import asyncio
import sys
import xml.etree.ElementTree as ET

xml_path = '../../../data/douyin-20/douyin_tc1_b_1.xml'
with open(xml_path, 'r', encoding='utf-8') as f:
    xml_content = f.read()

root = ET.fromstring(xml_content)

# 找所有有 text 属性且非空的节点
text_nodes = []
for node in root.iter():
    text = node.attrib.get('text', '')
    if text:  # 非空
        text_nodes.append({
            'class': node.attrib.get('class', ''),
            'text': text,
            'clickable': node.attrib.get('clickable', ''),
            'bounds': node.attrib.get('bounds', ''),
        })

print(f"=== 有非空 text 的节点数: {len(text_nodes)} ===")
for i, node in enumerate(text_nodes[:15]):
    print(f"{i + 1}. class={node['class']}, text='{node['text'][:30]}', clickable={node['clickable']}")

# 检查这些节点在 filtered 中的位置
if 'offline_driver' in sys.modules:
    del sys.modules['offline_driver']

from offline_driver import OfflineDriver
from mobilerun.tools.filters import ConciseFilter
from mobilerun.tools.formatters import IndexedFormatter


async def test():
    driver = OfflineDriver(
        dataset_path='../../../data/douyin-20/checkpoint_douyin.json',
        output_path='/tmp/test.json',
        testcase_index=0,
    )

    result = await driver.get_ui_tree()
    a11y_tree = result.get('a11y_tree', {})

    # filter
    filter_obj = ConciseFilter()
    device_context = result.get('device_context', {})
    filtered = filter_obj.filter(a11y_tree, device_context)

    # format
    formatter = IndexedFormatter()
    formatter.screen_width = device_context['screen_bounds']['width']
    formatter.screen_height = device_context['screen_bounds']['height']

    formatted_text, focused_text, elements, phone_state = formatter.format(filtered, result.get('phone_state', {}))

    # 在 elements 中查找有实际 text 的（不是 resourceId/className）
    print("\n=== elements 中有实际 text 内容的 ===")
    count = 0
    for el in elements:
        text = el.get('text', '')
        className = el.get('className', '')
        resourceId = el.get('resourceId', '')

        # text 是实际内容（不是 fallback 到 className 或 resourceId）
        is_actual_text = text and text != className and (not resourceId or text != resourceId)

        if is_actual_text:
            count += 1
            if count <= 15:
                print(
                    f"{count}. index={el['index']}, className={className}, text='{text[:50]}', resourceId={resourceId}")

    print(f"\n总计: {count} 个")


asyncio.run(test())
