"""
检查 _intersects_screen 和 _min_size 的具体逻辑
"""
from mobilerun.tools.filters import ConciseFilter
import inspect

print("=== _intersects_screen 源码 ===")
source = inspect.getsource(ConciseFilter._intersects_screen)
print(source)

print("\n=== _min_size 源码 ===")
source = inspect.getsource(ConciseFilter._min_size)
print(source)


# 手动解析 bounds
def parse_bounds(bounds_str):
    """解析 bounds 字符串 '[x1,y1][x2,y2]'"""
    import re
    match = re.match(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', bounds_str)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
    return None


# 测试 bounds 解析
bounds_str = '[0,0][720,1456]'
print(f"\n=== 解析 bounds '{bounds_str}' ===")
result = parse_bounds(bounds_str)
print(f"解析结果: {result}")

if result:
    x1, y1, x2, y2 = result
    screen_width, screen_height = 720, 1456
    print(f"x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    print(f"screen_width={screen_width}, screen_height={screen_height}")

    # 手动计算 _intersects_screen 的逻辑
    # 通常是检查是否与屏幕区域有交集
    intersects = not (x2 < 0 or x1 > screen_width or y2 < 0 or y1 > screen_height)
    print(f"\n手动计算 intersects: {intersects}")
    print(f"  x2 < 0: {x2 < 0}")
    print(f"  x1 > screen_width: {x1 > screen_width}")
    print(f"  y2 < 0: {y2 < 0}")
    print(f"  y1 > screen_height: {y1 > screen_height}")

# 检查子节点的 bounds
bounds_str2 = '[0,0][720,1600]'
print(f"\n=== 解析 bounds '{bounds_str2}' ===")
result2 = parse_bounds(bounds_str2)
print(f"解析结果: {result2}")
if result2:
    x1, y1, x2, y2 = result2
    print(f"x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    # y2=1600 > screen_height=1456，这可能就是问题！
    print(f"y2 ({y2}) > screen_height ({screen_height}): {y2 > screen_height}")
