activate PYTHON36
python src/gen_frozen_pb.py --checkpoint=C:\GANHands\HandPoseEstimationForMobile\trained\zq71_cpm_tiny\log\zq71_cpm_batch-20_lr-0.001_gpus-1_256x256_experiments-zq71_cpm\model-10000 --output_graph=models\zq71-model-10000.pb --size=224 --model=zq62_cpm --output_node_names=CPM/stage_1_out

python src/benchmark.py --anno_json_path=../GANHands_test.json --img_path=C:/GaneratedHandsForReal_TIME/data/GANeratedHands_Release --output_node_name="CPM/stage_1_out" --frozen_pb_path=models\zq71-model-10000.pb 