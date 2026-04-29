"""
逐步调试 filter 流程
"""
from mobilerun.tools.filters import ConciseFilter
import xml.etree.ElementTree as ET

xml_path = '../../../data/douyin-20/douyin_tc1_b_1.xml'
with open(xml_path, 'r', encoding='utf-8') as f:
    xml_content = f.read()

root = ET.fromstring(xml_content)

def node_to_dict(node):
    result = dict(node.attrib)
    children = [node_to_dict(c) for c in node]
    if children:
        result["children"] = children
    return result

tree = node_to_dict(root)

# 添加 bounds
if 'bounds' not in tree:
    width = int(tree.get('width', 720))
    height = int(tree.get('height', 1456))
    tree['bounds'] = f'[0,0][{width},{height}]'

print(f"=== 根节点 ===")
print(f"有 bounds: {'bounds' in tree}")
print(f"bounds: {tree.get('bounds')}")
print(f"children 数量: {len(tree.get('children', []))}")

# 检查第一个子节点
if tree.get('children'):
    first_child = tree['children'][0]
    print(f"\n=== 第一个子节点 ===")
    print(f"有 bounds: {'bounds' in first_child}")
    print(f"bounds: {first_child.get('bounds')}")
    print(f"class: {first_child.get('class')}")

# 手动调用 filter 内部方法
filter_obj = ConciseFilter()
screen_bounds = {"width": 720, "height": 1456}
filtering_params = {}

print("\n=== 手动测试 _intersects_screen ===")
result = filter_obj._intersects_screen(tree, 720, 1456)
print(f"根节点 _intersects_screen: {result}")

print("\n=== 手动测试 _min_size ===")
result = filter_obj._min_size(tree, 5)
print(f"根节点 _min_size: {result}")

# 如果第一个子节点有 bounds，也测试它
if tree.get('children') and 'bounds' in tree['children'][0]:
    first = tree['children'][0]
    print(f"\n第一个子节点 _intersects_screen: {filter_obj._intersects_screen(first, 720, 1456)}")
    print(f"第一个子节点 _min_size: {filter_obj._min_size(first, 5)}")

# 测试完整的 filter
print("\n=== 完整 filter 测试 ===")
device_context = {
    "screen_bounds": {"width": 720, "height": 1456},
    "filtering_params": {}
}
filtered = filter_obj.filter(tree, device_context)
print(f"filter 结果是否为 None: {filtered is None}")
if filtered:
    print(f"filtered children 数量: {len(filtered.get('children', []))}")
