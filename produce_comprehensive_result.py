import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse  # 导入argparse模块

# 设置argparse，用于从命令行接收参数
parser = argparse.ArgumentParser(description='Plot performance metrics for various datasets.')
parser.add_argument('--tail', type=str, required=True, help='A tail number to specify dataset version.')
args = parser.parse_args()

# 使用args.tail获取传入的tail值
tail = args.tail

datasets = ['yelp-chi', 'texas', 'wisconsin', 'chameleon', 'squirrel', 'cornell']

for i, dataset in enumerate(datasets):
    B = np.load(f'utils_data/ourModel_{dataset}_{tail}/best_val_metric_list.npy')
    mlp_base = list(np.load(f'utils_data/mlp_{dataset}_{tail}/best_val_metric_list.npy')) * len(B)
    GCN_base = list(np.load(f'utils_data/GCN_{dataset}_{tail}/best_val_metric_list.npy')) * len(B)
    plt.plot(mlp_base, 'o-', label='mlp_base')
    plt.plot(B, 'o-', label='ourModel')
    plt.plot(GCN_base, 'o-', label='GCN_base')
    plt.legend()
    plt.title(f'{dataset}')
    plt.ylabel('Metric')
    plt.xlabel('Iteration')
    plt.grid(True)
    plt.savefig(f'imgs/{dataset}_{tail}.png')
    plt.clf()  # 清除当前图形

image_list = [f'{dataset}_{tail}.png' for dataset in datasets]

img_sample = plt.imread(os.path.join('imgs', image_list[0]))
(max_height, max_width, _) = img_sample.shape

images = [mpimg.imread(os.path.join('imgs', path)) for path in image_list]

plt.figure(figsize=(max_width * 3 / 100, max_height * 2 / 100))

for i, image in enumerate(images):
    plt.subplot(2, 3, i+1)
    plt.imshow(image)
    plt.axis('off') 
    
plt.tight_layout()

plt.show()
plt.savefig(f'imgs/Composite_{tail}.png')
print(f'imgs/Composite_{tail}.png')

