"""
直接执行 offline_driver.py 的代码并测试
"""
import sys
import os

# 直接读取并执行文件内容
with open('offline_driver.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 创建一个新的模块命名空间
module_ns = {}
exec(code, module_ns)

# 获取 OfflineDriver 类
OfflineDriver = module_ns['OfflineDriver']

# 测试
import asyncio


async def test():
    driver = OfflineDriver(
        dataset_path='../../../data/douyin-20/checkpoint_douyin.json',
        output_path='/tmp/test.json',
        testcase_index=0,
    )

    print(f'\n=== 检查方法 ===')
    print(f'Has _resolve_path: {hasattr(driver, "_resolve_path")}')
    print(f'Has _parse_xml_to_tree: {hasattr(driver, "_parse_xml_to_tree")}')
    print(f'Has get_ui_tree: {hasattr(driver, "get_ui_tree")}')

    if hasattr(driver, '_resolve_path'):
        print(f'\n=== 测试路径解析 ===')
        resolved = driver._resolve_path('./douyin_tc1_b_1.xml')
        print(f'Resolved path: {resolved}')
        print(f'File exists: {os.path.exists(resolved)}')

        if os.path.exists(resolved):
            with open(resolved, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f'XML content length: {len(content)}')

    print(f'\n=== 测试 get_ui_tree ===')
    result = await driver.get_ui_tree()
    print(f'Result keys: {list(result.keys())}')
    print(f'raw_xml length: {len(result.get("raw_xml", ""))}')
    print(f'a11y_tree has content: {bool(result.get("a11y_tree"))}')


asyncio.run(test())
