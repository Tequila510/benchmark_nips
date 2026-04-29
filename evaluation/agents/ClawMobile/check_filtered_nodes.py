"""
检查 filtered 数据中节点的属性
"""
import asyncio
import sys

if 'offline_driver' in sys.modules:
    del sys.modules['offline_driver']

from offline_driver import OfflineDriver
from mobilerun.tools.filters import ConciseFilter


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

    # 检查 filtered 的结构
    print("=== filtered 根节点 ===")
    print(f"keys: {list(filtered.keys())}")

    # 递归打印前几层节点的关键属性
    def print_node(node, depth=0, max_depth=3):
        if depth > max_depth:
            return

        indent = "  " * depth
        className = node.get('class', 'N/A')
        text = node.get('text', '')
        resourceId = node.get('resource-id', '')
        clickable = node.get('clickable', '')

        if text or clickable == 'true' or depth < 2:
            print(f"{indent}class={className}, text='{text[:30]}', resource-id={resourceId}, clickable={clickable}")

        for child in node.get('children', []):
            print_node(child, depth + 1, max_depth)

    print("\n=== 节点树结构 ===")
    print_node(filtered, max_depth=4)

    # 统计有 text 的节点
    def count_text_nodes(node, results):
        text = node.get('text', '')
        if text:
            results.append({
                'class': node.get('class', ''),
                'text': text,
                'clickable': node.get('clickable', ''),
            })
        for child in node.get('children', []):
            count_text_nodes(child, results)

    text_nodes = []
    count_text_nodes(filtered, text_nodes)
    print(f"\n=== 有 text 的节点数: {len(text_nodes)} ===")
    for i, node in enumerate(text_nodes[:10]):
        print(f"  {i + 1}. class={node['class']}, text='{node['text'][:30]}', clickable={node['clickable']}")


asyncio.run(test())
