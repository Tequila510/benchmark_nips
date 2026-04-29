"""
检查 XML 解析后的 a11y_tree 结构
"""
import os
import xml.etree.ElementTree as ET

xml_path = '../../../data/douyin-20/douyin_tc1_b_1.xml'

with open(xml_path, 'r', encoding='utf-8') as f:
    xml_content = f.read()

print(f"XML 总长度: {len(xml_content)}")

# 解析
root = ET.fromstring(xml_content)
print(f"\n根节点: {root.tag}")
print(f"根节点属性: {root.attrib}")

# 统计所有节点
all_nodes = list(root.iter())
print(f"\n总节点数: {len(all_nodes)}")

# 找有 clickable=true 的节点
clickable_nodes = [n for n in all_nodes if n.attrib.get('clickable') == 'true']
print(f"可点击节点数: {len(clickable_nodes)}")

# 找有 text 的节点
text_nodes = [n for n in all_nodes if n.attrib.get('text')]
print(f"有 text 的节点数: {len(text_nodes)}")

# 打印前几个节点的信息
print("\n前10个节点:")
for i, node in enumerate(all_nodes[:10]):
    attrs = dict(list(node.attrib.items())[:5])
    print(f"  {i+1}. {node.tag}: {attrs}")

# 检查根节点的直接子节点
print(f"\n根节点的直接子节点数: {len(root)}")
for i, child in enumerate(root[:3]):
    print(f"  子节点 {i+1}: {child.tag}, 属性: {dict(list(child.attrib.items())[:3])}")
    print(f"    子节点的子节点数: {len(child)}")

# 转换为字典树（模拟 _parse_xml_to_tree）
def node_to_dict(node):
    result = dict(node.attrib)
    children = [node_to_dict(c) for c in node]
    if children:
        result["children"] = children
    return result

a11y_tree = node_to_dict(root)
print(f"\na11y_tree 字典:")
print(f"  keys: {list(a11y_tree.keys())}")
print(f"  有 children: {'children' in a11y_tree}")
if 'children' in a11y_tree:
    print(f"  children 数量: {len(a11y_tree['children'])}")
    if a11y_tree['children']:
        print(f"  第一个 child keys: {list(a11y_tree['children'][0].keys())}")
