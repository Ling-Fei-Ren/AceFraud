
It is without any information about author name and organizaion name.
It is only used for Reproducibility checking.

Requirements：
torch-cluster==1.5.7
torch-geometric==1.6.3
torch-scatter==2.0.6
torch==1.6.0
scikit-learn==0.21.3
numpy==1.17.2
hyperopt==0.2.5
python==3.7.4

Instructions to run the experiment


Step 1. Run the search process, given different random seeds. (The yelpchi dataset is used as an example)

python train_search.py  --data yelp --fix_last True  --epochs 20


Step 2. Fine tune the searched architectures. You need specify the arch_filename with the resulting filename from Step 1.

python fine_tune.py --data yelp  --fix_last True   --hyper_epoch 50  --arch_filename exp_res/yelp.txt   
Step 2 is a coarse-graind tuning process, and the results are saved in a picklefile in the directory tuned_res.

