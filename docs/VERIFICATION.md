# 0.1.0 验证记录

- 全量测试：122 passed。
- Ruff 指定的语法／名称错误规则检查通过（不是全面风格或安全审计）。
- 成功构建源代码包和 `py3-none-any` wheel。
- 将 wheel 安装到独立虚拟环境后，再次通过 122 项测试；导入位置确认来自安装包，而非源码路径。
- 安装后的 `agipot-predict --demo` 生成严格 JSON 报告，输入标记为合成数据。
- 对公开源码、测试、文档进行私有导入、常见密钥格式、账户配置和本机路径扫描，未发现匹配项。

本机运行时为 Python 3.12。当前发布凭证缺少 GitHub OAuth `workflow` 权限，因此自动 CI 尚未启用。附带的 [GitHub Actions 模板](github-actions-tests.yml)可由有权限的维护者放入 `.github/workflows/tests.yml`，届时将对 Python 3.11／3.12／3.13 执行安装、测试、演示和构建。这些其他版本的 CI 尚未运行。

上述检查验证程序行为与打包可用性，不构成预测精度、投资收益或正式统计显著性的验证。
