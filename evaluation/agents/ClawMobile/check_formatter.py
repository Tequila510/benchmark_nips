"""
检查 IndexedFormatter 如何提取属性
"""
from mobilerun.tools.formatters import IndexedFormatter
import inspect

print("=== IndexedFormatter.format 方法签名 ===")
print(inspect.signature(IndexedFormatter.format))

print("\n=== IndexedFormatter.format 源码 ===")
source = inspect.getsource(IndexedFormatter.format)
print(source[:3000])

# 检查它是如何处理节点的
print("\n\n=== 检查 _format_node 方法 ===")
try:
    source = inspect.getsource(IndexedFormatter._format_node)
    print(source[:2000])
except:
    print("_format_node 不存在，可能在别处")

# 打印所有方法
print("\n=== IndexedFormatter 所有方法 ===")
for name in dir(IndexedFormatter):
    if not name.startswith('_') or name in ['_format_node', '_element_to_str', '_process_node']:
        print(f"  {name}")
