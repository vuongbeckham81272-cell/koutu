# 硬边抠图

一键抠图 · 硬边输出（应用名：硬边抠图）。本地离线 AI 抠图桌面工具：拖入产品图片，一键输出透明背景 PNG，边缘为无羽化的硬边（类似 Photoshop 钢笔工具的锐利边缘）。

> 注：macOS/Windows 的打包工具链要求安装包名为 ASCII，因此安装包文件名为 `koutu_*`；应用运行后窗口标题、界面显示均为「硬边抠图」。

- 完全免费、完全离线：不调用任何付费 API，不上传图片，断网可用
- 推理模型（BiRefNet_lite）随安装包分发，首次启动不需要联网下载
- 支持 Windows / macOS（Apple Silicon）

## 使用方法

1. 启动应用，等待右上角显示「模型就绪」
2. 把图片拖入虚线框，或点击虚线框选择文件
3. 等待推理完成（CPU 约 8~15 秒），结果以棋盘格背景展示
4. 点击「下载透明 PNG」，保存为 `原文件名-cutout.png`

选项说明：

- **硬边模式**（默认开启）：alpha 只有 0/255 两个值，放大 400% 无半透明灰边；关闭后为模型原始软边效果
- **原图 / 结果**：切换查看抠图前后的对比
- **批量处理**：选择输入文件夹和输出文件夹，逐张处理并保存

## 各平台安装

### Windows

1. 下载 `koutu_0.1.0_x64-setup.exe`（NSIS 安装器）
2. 双击运行，按提示完成安装
3. 首次运行如出现 SmartScreen 提示（未签名应用的正常现象），点击「更多信息」→「仍要运行」

### macOS（Apple Silicon，M1/M2/M3/M4 芯片）

1. 下载 `koutu_0.1.0_aarch64.dmg`
2. 打开 DMG，把「硬边抠图.app」拖入 Applications
3. 首次打开如提示「无法验证开发者」（未签名应用的正常现象），在终端执行：

   ```bash
   xattr -cr /Applications/硬边抠图.app
   ```

   之后正常双击打开即可

## 常见问题

**首次启动需要联网吗？**
不需要。模型文件已打包进安装包（约 224MB），所有推理均在本地 CPU 完成。

**应用图标是默认占位图？**
本版本使用 Tauri 默认图标占位，后续版本会替换为正式图标。

## 开发

```bash
npm install
npm run tauri dev    # 开发模式
npm run tauri build  # 本地打包（产物见下方目录）
```

> 注意：模型文件（224MB）不存放在 git 仓库中。如果是 `git clone` 下来的代码，
> 首次运行前需手动下载模型放到 `src-tauri/models/BiRefNet_lite.onnx`
> （HuggingFace: `onnx-community/BiRefNet_lite-ONNX` 的 `onnx/model.onnx`）；
> 通过 Release 安装包安装的最终用户无需此步骤（CI 打包时已自动下载并打入安装包）。

打 tag 发布（自动构建 Windows 与 macOS Apple Silicon 安装包并上传到 GitHub Release）：

```bash
git tag v0.1.0
git push origin v0.1.0
```

## 本地构建产物目录

- Windows：`src-tauri/target/release/bundle/nsis/`
- macOS：`src-tauri/target/release/bundle/dmg/`

## 技术栈

Tauri 2（Rust 后端）+ React + Vite + TypeScript；推理使用 `ort` crate 加载 BiRefNet_lite ONNX 模型，本地 CPU 运行；硬边后处理为纯 Rust 手写的阈值化 + 3×3 形态学 opening。
