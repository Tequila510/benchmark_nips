"""
强制重新加载 offline_driver 模块并测试
"""
import sys
import os
import importlib.util

# 清除所有可能的缓存
for key in list(sys.modules.keys()):
    if 'offline_driver' in key:
        del sys.modules[key]

# 强制从文件加载
spec = importlib.util.spec_from_file_location('offline_driver', './offline_driver.py')
offline_driver_module = importlib.util.module_from_spec(spec)
sys.modules['offline_driver'] = offline_driver_module
spec.loader.exec_module(offline_driver_module)

# 现在导入
OfflineDriver = offline_driver_module.OfflineDriver

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
    print(f'a11y_tree keys: {list(result.get("a11y_tree", {}).keys()) if result.get("a11y_tree") else "empty"}')


asyncio.run(test())
