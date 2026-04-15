import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont

def extract_clickable_nodes(xml_path):
    """
    从 uiautomator XML 中读取 clickable=true 的节点，并解析 bounds。
    返回: [(index, (x1, y1, x2, y2))]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    clickable_nodes = []
    idx = 1

    def dfs(node):
        nonlocal idx
        # 判断 clickable
        if node.attrib.get("clickable") == "true":
            bounds = node.attrib.get("bounds")
            if bounds:
                # 解析 "[x1,y1][x2,y2]"
                try:
                    parts = bounds.replace("[", "").replace("]", ",").split(",")
                    x1, y1, x2, y2 = map(int, parts[:4])
                    clickable_nodes.append((idx, (x1, y1, x2, y2)))
                    idx += 1
                except Exception as e:
                    print("Bounds parse error:", bounds, e)

        # DFS 遍历子节点
        for child in node:
            dfs(child)

    dfs(root)
    return clickable_nodes


def draw_clickable_nodes(image_path, xml_path, output_path="output_with_boxes.png"):
    """
    在截图上绘制 clickable 元素的 bounding boxes 和编号。
    """
    # 加载图像
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # 字体设置（Mac 上常用）
    try:
        font = ImageFont.truetype("Arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    nodes = extract_clickable_nodes(xml_path)

    for idx, (x1, y1, x2, y2) in nodes:
        # 绘制矩形
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        # 绘制序号
        draw.text((x1 + 5, y1 + 5), str(idx), fill="red", font=font)
        # 显示 bounds 坐标
        print(f"idx: {idx}, bbox: [{x1},{y1},{x2},{y2}]")

    img.save(output_path)
    print("Saved:", output_path)


if __name__ == "__main__":
    # 示例
    draw_clickable_nodes(
        image_path="../traces_new/taobao-16/taobao_tc1_EO_3.png",
        xml_path="../traces_new/taobao-16/taobao_tc1_EO_3.xml",
        output_path="annotated.png"
    )