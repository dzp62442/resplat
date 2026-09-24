# 命令清单：通过 python scripts/train_queue.py 启动或恢复，不要直接 bash train.sh。

# 依次执行 Init 训练 → Init 最终 mini 评估 → Refine 训练 → Refine 最终 mini 评估。
bash scripts/omniscene_view6_112x200_base_init.sh _1
bash scripts/omniscene_view6_112x200_base_refine.sh _1

bash scripts/omniscene_view6_112x200_base_init.sh _2
bash scripts/omniscene_view6_112x200_base_refine.sh _2
