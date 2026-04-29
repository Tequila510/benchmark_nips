"""
检查有 text 内容的节点是否正确显示
"""
import asyncio
import sys

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

    # 打印有实际 text 的元素
    print("=== 有 text 的 elements ===")
    count = 0
    for el in elements:
        # text 不是 className 才是有实际内容
        if el.get('text') and el.get('text') != el.get('className'):
            count += 1
            if count <= 15:
                print(
                    f"{count}. index={el['index']}, className={el['className']}, text='{el['text'][:50]}', bounds={el['bounds']}")

    print(f"\n总计: {count} 个有 text 的元素")

    # 打印 formatted_text 的关键部分
    print("\n=== formatted_text 片段 ===")
    print(formatted_text[:1500])


asyncio.run(test())
