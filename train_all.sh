# python train.py --config configs/iphone_blur_dataset/paper-windmill.txt
# python train.py --config configs/iphone_blur_dataset/backpack.txt
# python train.py --config configs/iphone_blur_dataset/mochi-high-five.txt
# python train.py --config configs/iphone_blur_dataset/spin.txt
# python train.py --config configs/iphone_blur_dataset/sriracha-tree.txt
# python train.py --config configs/iphone_blur_dataset/teddy.txt
python train.py --config configs/iphone_blur_dataset/space-out.txt
python train.py --config configs/iphone_blur_dataset/wheel.txt
# ABOVE traning has failed


# python train.py --config configs/iphone_blur_dataset/block.txt
# python train.py --config configs/iphone_blur_dataset/creeper.txt
# ABOVE scenes are removed

# python train.py --config configs/iphone_blur_dataset/apple.txt
# python train.py --config configs/iphone_blur_dataset/handwavy.txt
# python train.py --config configs/iphone_blur_dataset/pillow.txt

python evaluation_iphone.py --config configs/iphone_blur_dataset/space-out.txt >> results/space-out.log
python evaluation_iphone.py --config configs/iphone_blur_dataset/wheel.txt >> results/wheel.log
# python train.py --config configs/iphone_blur_dataset/wheel.txt
