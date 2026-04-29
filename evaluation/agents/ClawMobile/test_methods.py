"""
检查所有方法
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

# 列出所有方法
print("OfflineDriver 类的所有方法:")
for name in dir(OfflineDriver):
    if not name.startswith('__'):
        print(f"  {name}")

print("\n=== 检查特定方法 ===")
driver = OfflineDriver(
    dataset_path='../../../data/douyin-20/checkpoint_douyin.json',
    output_path='/tmp/test.json',
    testcase_index=0,
)

methods_to_check = [
    '_parse_checkpoints',
    'set_testcase_index',
    'advance_checkpoint',
    '_save_results',
    '_get_current_checkpoint',
    '_resolve_path',
    'get_recorded_actions',
    '_record_action',
    '_parse_xml_to_tree',
    'get_ui_tree',
]

for method in methods_to_check:
    has_it = hasattr(driver, method)
    print(f"  {method}: {has_it}")
