import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold


def get_fold():
    root = "/home/data_normal/abiz/wuzhiqiang/wzq/data/shopee-product-matching"
    df = pd.read_csv(root + '/train.csv')
    #      posting_id,                               image,      image_phash,                     title,   label_group
    # train_129225211  0000a68812bc7e98c42888dfb1c07da0.jpg  94974f937d4c2433  Paper Bag Victoria Secret    249114794
    # [34250 x 5]
    df['image_path'] = df['image'].apply(lambda x: os.path.join(root, 'train_images', x))
    
    tmp = df.groupby(['label_group'])['posting_id'].unique().to_dict()
    df['targets'] = df['label_group'].map(tmp)
    df['targets'] = df['targets'].apply(lambda x: " ".join(x))
    df = pd.concat([df, pd.DataFrame(columns=['fold'])], sort=False)

    kf = GroupKFold(n_splits=5)
    for fold, (train_index, test_index) in enumerate(kf.split(df, df['label_group'], df['label_group'])):
        print(f"fold {fold}: {len(test_index)}")
        df.loc[test_index, 'fold'] = fold
    
    print(df.head())
    df.to_csv("folds.csv", index=False)


if __name__ == "__main__":
    get_fold()