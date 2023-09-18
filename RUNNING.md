# 运行方法

TimeSformer 位置：`~/TimeSformer`
MViT 位置：`~/SlowFast`

1. `conda activate timesformer` 进入环境。（如果是 MViT，使用 `conda activate slowfast`）
2. 最终的测试配置在 `configs/Kinetics/mid_term`（space-only）,`configs/Kinetics/divST_benchmarks`（divST）文件夹中, `configs/SSv2/` (SSv2 数据集)，命名规则：
    1. `_my_cache_only` 为缓存阶段（如果需要从头缓存和计算 SSIM 值，需要删除 `PATH_TO_DATA_DIR` 指定文件夹内的 `test_ssim_?x?.dat` SSIM 值和缓存文件夹 `PATH_CACHE_DIR` 指定的文件夹，如果不需要缓存，在配置文件中将 `CACHE` 设置为 `False`，否则正式运行时也会先运行缓存阶段，目前所有的 SSIM 值是齐全的没有删除）。
    2. `_my(_baseline)` 为基准。
    3. `_my_0.000_e_0.0_min_0` 为 SSIM 实验，参数请参照表格，第一个值为 alpha，第二个值为 epsilon，第三个值为 m。
3. yaml 文件运行方法：`python tools/run_net.py --cfg path/to/config`，一般运行顺序：缓存->基准->SSIM 组->删除缓存。（SSv2 数据集不需要缓存阶段，MViT 不需要缓存阶段，但是也应当使用数据预处理阶段计算 SSIM）缓存需要留足够的空间，TimeSformer-L (K600) 需要 1.4T 的硬盘空间。

config 中新增的键可以查阅 `timesformer/config/defaults.py` 文件中的注释。
`OUTPUT_DIR` 将会指定 log 的位置。
