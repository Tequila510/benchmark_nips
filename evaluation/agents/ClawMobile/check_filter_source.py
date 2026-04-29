"""
检查 ConciseFilter 的源码和期望格式
"""
from mobilerun.tools.filters import ConciseFilter
import inspect

print("=== ConciseFilter.filter 方法签名 ===")
print(inspect.signature(ConciseFilter.filter))

print("\n=== ConciseFilter 源码 ===")
source = inspect.getsource(ConciseFilter.filter)
print(source[:3000])

print("\n=== ConciseFilter 类的所有方法 ===")
for name in dir(ConciseFilter):
    if not name.startswith('_'):
        print(f"  {name}")
