# cd DESIGN
# python main.py # train target rec model
# python inference_result.py # get rec item list
# python shadow_dataset_preprocessing.py # generate shadow dataset based on the rec result and shadow graph
# python main.py --is_shadow True # train shadow rec model
# python gen_mia_dataset.py # generate mia training and test dataset
# python mia_attack.py --agg_mtd uemb # perform mia attack

dataset='ciao'
# dataset='flickr'

model_name='DESIGN'
trained_name='DESIGN-def-er-2.pth'
# trained_name='DESIGN-def-dp-0.1.pth'
# trained_name='DESIGN-ciao-final.pth'
# trained_name='DESIGN-def-l1v3-0.02.pth'

# trained_name='DESIGN-flickr-final.pth'
# trained_name='DESIGN-flickr-def-l1v3-0.02.pth'
# trained_name='DESIGN-flickr-def-er-2.pth'
# trained_name='DESIGN-flickr-def-dp-2.pth'

# model_name='DiffNet'
# trained_name='DiffNet-ciao-final.pth'
# trained_name='DiffNet-ciao-def-l1v3-0.04.pth'
# trained_name='DiffNet-ciao-def-dp-1.pth'
# trained_name='DiffNet-ciao-def-er-1.5.pth'

# trained_name='DiffNet-flickr-final.pth'
# trained_name='DiffNet-flickr-def-l1v3-0.1.pth'
# trained_name='DiffNet-flickr-def-dp-1.pth'
# trained_name='DiffNet-flickr-def-er-1.5.pth'


output_path=log/$dataset/mia_attack_$trained_name.txt
# echo $output_path
gpu_id=0
shadow_ratio=0.1
rec_lens=30
base_attack='enminer'

echo -e '\n[INFO] inference result ...'
python inference_result.py --trained_name $trained_name --dataset $dataset --is_shadow False --gpu_id $gpu_id --shadow_ratio $shadow_ratio --rec_lens $rec_lens --model_name $model_name

echo -e '\n[INFO] generate shadow dataset ...'
python shadow_dataset_preprocessing.py --trained_name $trained_name --dataset $dataset --gpu_id $gpu_id --shadow_ratio $shadow_ratio --rec_lens $rec_lens --model_name $model_name

echo -e '\n[INFO] train shadow model use shadow dataset ...'
python main_shadow.py --is_shadow True --trained_name $trained_name --num_epoch 10 --dataset $dataset --gpu_id $gpu_id --shadow_ratio $shadow_ratio --rec_lens $rec_lens --model_name $model_name

echo -e '\n[INFO] generate mia dataset ...'
python gen_mia_dataset.py --trained_name $trained_name --dataset $dataset --gpu_id $gpu_id --shadow_ratio $shadow_ratio --rec_lens $rec_lens --model_name $model_name

# 改这个参数--agg_mtd来perform attack baselines
echo -e '\n[INFO] perform mia attack ...'
# python mia_attack.py --trained_name $trained_name --dataset $dataset --gpu_id $gpu_id --shadow_ratio $shadow_ratio --rec_lens $rec_lens --model_name $model_name --agg_mtd $base_attack >> $output_path
python mia_enhance_attack.py --trained_name $trained_name --dataset $dataset --gpu_id $gpu_id --shadow_ratio $shadow_ratio --rec_lens $rec_lens >> $output_path