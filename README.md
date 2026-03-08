# EMG 手势识别实验项目

这是一个基于表面肌电（EMG）串口采集的四分类手势识别原型项目。  
项目通过 Arduino 串口实时读取多通道数据，提取时域/频域/小波特征，使用随机森林完成训练与在线预测。

## 功能概览

- 串口实时采集 EMG 数据（默认 `COM6`，`115200`）
- 四类动作采样与标签构建（前 / 后 / 左 / 右）
- 固定窗口分段（每窗 200 点）
- 多特征融合：
  - 时域：`RMS`、`MAV`、`WL`、`ZC`、`SSC`、`VAR`、`WA`、`MCV`
  - 频域：`MPF`、`MF`、`SM2`、`ME`
  - 参数模型：`AR(7)`
  - 时频：`WPT`（小波包）
- 模型训练与评估：`RandomForestClassifier`
- 在线循环推理输出预测类别

## 项目结构

- `main.py`：数据采集、特征拼接、模型训练、在线预测主流程
- `feature_utils.py`：特征工程函数库

## 运行环境

建议 Python 3.9+，依赖包括：

- `pyserial`
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `pywavelets`
- `statsmodels`
- `tqdm`

可使用以下命令安装：

```bash
pip install pyserial numpy pandas scipy scikit-learn pywavelets statsmodels tqdm
```

## 使用说明

1. 将 EMG 设备连接到串口，并确认串口参数（代码默认 `COM6`）。
2. 按需要修改 `main.py` 中的串口名（macOS/Linux 通常是 `/dev/tty.*` 或 `/dev/ttyUSB*`）。
3. 运行：

```bash
python main.py
```

4. 按终端提示依次采集四类动作样本。程序会完成训练并持续输出在线预测结果。

## 注意事项

- 当前脚本是实验原型，流程较长且为阻塞式采集。
- `RandomForestClassifier(max_features='auto')` 在新版本 scikit-learn 可能产生兼容性警告，可按版本调整。
- 建议后续加入：配置文件、数据保存、异常处理、可视化与模型持久化。
