# 千问本机控制台

地址：<http://127.0.0.1:8765/>。页面可以收藏。它只监听本机，不能通过其他设备访问。

当前用户登录 Windows 时，启动文件夹中的“Qianwen-Control-Panel”快捷方式会启动控制台，**不会自动启动助手**。页面的启动与关闭操作分别调用 `scripts/start_local.ps1` 和 `scripts/start_local.ps1 -Restart -StopOnly`；关闭不会停止共享 Ollama 服务或卸载模型。服务状态是端口连通性，不代表 QQ 已连接或模型推理可用。日志可能包含聊天内容，常见凭据格式会被脱敏，但并不保证过滤所有私人信息。

显存看板使用 `nvidia-smi` 显示整张 NVIDIA 显卡的显存总占用，并非仅限千问；读取失败时显示“暂不可用”。

安装开机启动：`powershell -NoProfile -ExecutionPolicy Bypass -File scripts/install_dashboard.ps1`。取消开机启动：删除 Windows 当前用户启动文件夹里的“Qianwen-Control-Panel”快捷方式。手动启动：使用 AstrBot 环境的 `pythonw.exe` 运行 `dashboard/server.py`。如果 8765 已被其他程序占用，应检查占用者，不要终止未知进程。
