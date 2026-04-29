"""
检查 ConciseFilter._filter_node 方法
"""
from mobilerun.tools.filters import ConciseFilter
import inspect

print("=== ConciseFilter._filter_node 源码 ===")
try:
    source = inspect.getsource(ConciseFilter._filter_node)
    print(source)
except Exception as e:
    print(f"获取源码失败: {e}")

# 直接调用测试
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


a11y_tree = node_to_dict(root)

# 测试 filter
filter_obj = ConciseFilter()
device_context = {
    "screen_bounds": {"width": 720, "height": 1456}
}

print("\n=== 调用 filter ===")
result = filter_obj.filter(a11y_tree, device_context)
print(f"结果类型: {type(result)}")
print(f"结果是否为 None: {result is None}")

if result is not None:
    print(f"结果 keys: {list(result.keys())[:10]}")
    if 'children' in result:
        print(f"结果 children 数量: {len(result['children'])}")
else:
    print("filter 返回了 None!")

# 检查 filtering_params
print("\n=== 尝试添加 filtering_params ===")
device_context_with_params = {
    "screen_bounds": {"width": 720, "height": 1456},
    "filtering_params": {}
}
result2 = filter_obj.filter(a11y_tree, device_context_with_params)
print(f"结果类型: {type(result2)}")
print(f"结果是否为 None: {result2 is None}")
