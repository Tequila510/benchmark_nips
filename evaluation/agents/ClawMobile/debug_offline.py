"""
调试：检查 OfflineDriver 从哪里来
"""
import sys

# 检查是否有缓存的 offline_driver 模块
print("=== 检查 sys.modules ===")
for key in sys.modules:
    if 'offline' in key.lower():
        print(f"  {key}: {sys.modules[key]}")

# 在 exec 之前检查
print("\n=== 在 exec 之前检查 ===")
# 先导入 DeviceDriver 看看
from mobilerun.tools.driver.base import DeviceDriver
print(f"DeviceDriver 来自: {DeviceDriver.__module__}")
print(f"DeviceDriver 方法: {[m for m in dir(DeviceDriver) if not m.startswith('__')]}")

# 检查是否有其他 OfflineDriver
try:
    from mobilerun.tools.driver import OfflineDriver as OtherDriver
    print(f"\n发现 mobilerun 中的 OfflineDriver: {OtherDriver}")
    print(f"方法: {[m for m in dir(OtherDriver) if not m.startswith('_')]}")
except ImportError:
    print("\nmobilerun 中没有 OfflineDriver")

# 现在执行我们的代码
print("\n=== exec 我们的定义 ===")
with open('offline_driver.py', 'r', encoding='utf-8') as f:
    code = f.read()

module_ns = {'DeviceDriver': DeviceDriver}
exec(code, module_ns)

OfflineDriver = module_ns['OfflineDriver']
print(f"我们的 OfflineDriver: {OfflineDriver}")
print(f"OfflineDriver.__module__: {OfflineDriver.__module__}")

# 列出所有方法
print("\n所有方法:")
for name in dir(OfflineDriver):
    if not name.startswith('__'):
        # 检查这个方法是否在 __dict__ 中（即是否是我们的）
        in_dict = name in OfflineDriver.__dict__
        print(f"  {name}: {'定义在类中' if in_dict else '继承自基类'}")

# 创建实例
driver = OfflineDriver(
    dataset_path='../../../data/douyin-20/checkpoint_douyin.json',
    output_path='/tmp/test.json',
    testcase_index=0,
)

print("\n=== 实例方法检查 ===")
print(f"driver._resolve_path 存在: {hasattr(driver, '_resolve_path')}")
print(f"'_resolve_path' in driver.__class__.__dict__: {'_resolve_path' in driver.__class__.__dict__}")

# 检查实例的 __class__ 是什么
print(f"\ndriver.__class__: {driver.__class__}")
print(f"driver.__class__.__bases__: {driver.__class__.__bases__}")
