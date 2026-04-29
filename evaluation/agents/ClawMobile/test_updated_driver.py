"""
使用更新后的 offline_driver.py 测试 filter
"""
import asyncio
import os
import sys

# 强制重新加载
if 'offline_driver' in sys.modules:
    del sys.modules['offline_driver']

from offline_driver import OfflineDriver


async def test():
    driver = OfflineDriver(
        dataset_path='../../../data/douyin-20/checkpoint_douyin.json',
        output_path='/tmp/test.json',
        testcase_index=0,
    )

    print(f"=== 测试 get_ui_tree ===")
    result = await driver.get_ui_tree()

    a11y_tree = result.get('a11y_tree', {})
    print(f"a11y_tree keys: {list(a11y_tree.keys())}")
    print(f"有 boundsInScreen: {'boundsInScreen' in a11y_tree}")
    if 'boundsInScreen' in a11y_tree:
        print(f"boundsInScreen: {a11y_tree['boundsInScreen']}")

    if 'children' in a11y_tree:
        first_child = a11y_tree['children'][0]
        print(f"\n第一个子节点 keys: {list(first_child.keys())[:10]}")
        print(f"有 boundsInScreen: {'boundsInScreen' in first_child}")
        if 'boundsInScreen' in first_child:
            print(f"boundsInScreen: {first_child['boundsInScreen']}")

    # 测试 filter
    from mobilerun.tools.filters import ConciseFilter
    from mobilerun.tools.formatters import IndexedFormatter

    filter_obj = ConciseFilter()
    formatter = IndexedFormatter()

    device_context = result.get('device_context', {})
    print(f"\ndevice_context: {device_context}")

    print("\n=== 测试 ConciseFilter ===")
    filtered = filter_obj.filter(a11y_tree, device_context)
    print(f"filtered 是否为 None: {filtered is None}")
    if filtered:
        print(f"filtered children 数量: {len(filtered.get('children', []))}")

    print("\n=== 测试 IndexedFormatter ===")
    formatter.screen_width = device_context['screen_bounds']['width']
    formatter.screen_height = device_context['screen_bounds']['height']

    formatted_text, focused_text, elements, phone_state = formatter.format(
        filtered or a11y_tree,
        result.get('phone_state', {})
    )

    print(f"elements 数量: {len(elements)}")
    if elements:
        print(f"前3个 elements:")
        for i, el in enumerate(elements[:3]):
            print(f"  {i + 1}. {el}")
    else:
        print("elements 为空!")
        print(f"formatted_text 前500字符:\n{formatted_text[:500]}")


asyncio.run(test())
