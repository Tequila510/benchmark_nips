"""
检查 Portal 返回的 a11y_tree 格式
"""
from mobilerun.tools.ui.provider import AndroidStateProvider
import inspect

print("=== AndroidStateProvider 源码片段 ===")
# 找到 get_state 方法中如何处理 a11y_tree
source = inspect.getsource(AndroidStateProvider.get_state)
print(source[:2000])

# 检查 Portal 返回的数据格式
print("\n\n=== 检查 Portal 格式 ===")
# 查找 portal_client 或相关代码
try:
    from mobilerun.tools.android.portal_client import PortalClient
    portal_source = inspect.getsource(PortalClient.get_ui_tree)
    print(portal_source[:2000])
except Exception as e:
    print(f"PortalClient 错误: {e}")
