# HandPoseEstimationForMobile
tensorflow训练手21点，heatmap

在此项目上修改而来https://github.com/edvardHua/PoseEstimationForMobile

# 数据

下载GANeratedDataset: https://handtracker.mpi-inf.mpg.de/projects/GANeratedHands/GANeratedDataset.htm ,解压之后目录为 GANeratedHands_Release

GANeratedHands_Release与training、trained同一目录


下载用于本工程的annotation: https://pan.baidu.com/s/1P09wcWukd6QMTyby-70d2g 

GANHands_test.json 、 GANHands_train.json 放在与training、 trained同一目录

其中如果要自行生成json文件，可以参考anno_to_json.py, 输入文件格式是 https://github.com/Ninebell/GaneratedHandsForReal_TIME 里的RegNet\train_path_list.ini

# 训练 

我在window 10系统，用anaconda2装的python3.6, tensorflow 1.9.0, 及其他(根据缺少提示一点一点装) 

**（1）修改cfg脚本**

如果你不确定解压的目录是否正确，打开training/experiments/zq71_cpm.cfg 
	
	修改imgpath: 'GANHAND_ROOT' （注意GANDHAND_ROOT为你解压后的目录）
	
	修改checkpoint、modelpath、logpath（此三者相同）

**（2）训练**

用命令行运行：
	
	cd training
	
	python src\train.py experiments/zq71_cpm.cfg
	
**（3）模型导出成pb**

根据实际情况修改如下命令：
	
	python src/gen_frozen_pb.py --checkpoint=..\trained\zq71_cpm_tiny\log\zq71_cpm_batch-20_lr-0.001_gpus-1_256x256_experiments-zq71_cpm\model-10000 --output_graph=models\zq71-model-10000.pb --size=256 --model=zq71_cpm --output_node_names=CPM/stage_1_out
	
**（4）运行测试集**

根据实际情况修改如下命令：
	
	python src/benchmark.py --anno_json_path=../GANHands_test.json --img_path=../GANeratedHands_Release --output_node_name="CPM/stage_1_out" --frozen_pb_path=models\zq71-model-10000.pb 


	