# Front tip / heatmap_sigma 消融（快速版）

对照基线：`front_fold0`（nose_tip=**3.0**, sigma=16, 已完整训完）

| Run | nose_tip | sigma | epochs | patience | 说明 |
|-----|---------:|------:|-------:|---------:|------|
| A1 | 6.0 | 16 | 30 | 10 | **新配置**（权重与基线不同，不能跳过） |
| A2 | 6.0 | 12 | 30 | 10 | 更尖峰 |
| A3 | 6.0 | 20 | 30 | 10 | 更宽峰 |

主指标：val/test **tip_mae ↓、PCK@20 ↑**；辅看 official mIoU / body / windshield。

状态：`outputs/train/TIP_ABLATION_STATUS.txt`

## Test 结果（fold_0，n=38，已完成）

| Run | tip 权重 / σ | official mIoU | tip MAE↓ | PCK@20↑ |
|-----|-------------|--------------:|---------:|--------:|
| 基线 front_fold0 | 3 / 16 | **0.739** | **17.3** | **0.76** |
| A1 | 6 / 16 | 0.707 | 21.0 | 0.74 |
| A2 | 6 / 12 | 0.698 | 27.2 | 0.71 |
| A3 | 6 / 20 | 0.702 | 18.4 | 0.74 |

结论（本轮快速消融）：加重 tip 到 6 **未优于**基线 w=3；三组里 **A3（σ=20）tip 最好但仍略差于基线**；**A2（σ=12）最差**。短训下更宽峰略稳，单纯加 tip 权重无效。下一步若继续提 tip，优先做 peak-boost / soft-argmax 坐标损失，而不是继续拧 w/σ。
