node_rank=$1
torchrun --nnodes 2 --node-rank $node_rank --nproc_per_node=2 --master-addr 1.1.1.1 --master_port=11111 ddp_pp_train.py --config config.yaml
