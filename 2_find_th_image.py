from model import *
from common import *
from dataset import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def do_valid(net, valid_loader):
    valid_label = []
    valid_probability = []

    start_timer = timer()
    with torch.no_grad():
        net.eval()
        valid_num = 0
        for t, batch in enumerate(valid_loader):
            index = batch['index']
            label = batch['label'].cuda()
            image = batch['image'].cuda()
            logit = net(image, label)
            p = torch.softmax(logit, dim=1)
            valid_num += len(index)

            valid_label.append(label.data.cpu().numpy())
            valid_probability.append(p.data.cpu().numpy())

            print('\r %8d / %d  %s' % (valid_num,
                                       len(valid_loader.dataset),
                                       time_to_str(timer() - start_timer, 'sec')),
                                       end='',
                                       flush=True)
        assert (valid_num == len(valid_loader.dataset))
    
    label = np.concatenate(valid_label)
    probability = np.concatenate(valid_probability)
    predict = probability.argmax(-1)

    acc = np_metric_accuracy(predict, label)
    loss = np_loss_cross_entropy(probability, label)
    return [loss, acc]


def extract_feat(net, valid_loader):
    features = []
    start_timer = timer()
    with torch.no_grad():
        net.eval()
        valid_num = 0
        for t, batch in enumerate(valid_loader):
            index = batch['index']
            image = batch['image'].cuda()
            feat = net(image)
            features += [feat.detach().cpu()]
            valid_num += len(index)
            print("\r %8d / %d  %s" % (valid_num,
                                       len(valid_loader.dataset),
                                       time_to_str(timer() - start_timer, 'sec')),
                                       end='',
                                       flush=True)
        assert (valid_num == len(valid_loader.dataset))
    print("\n")
    features = torch.cat(features).cpu().numpy()
    return features


def find_matches(posting_ids, threshold, features, n_batches, min_indices=1):
    assert len(posting_ids) == len(features)
    sim_threshold = threshold / 100
    y_pred = []
    n_rows = features.shape[0]
    bs = n_rows // n_batches
    batches = []
    for i in range(n_batches):
        left = bs * i
        right = bs * (i + 1)
        if i == n_batches - 1:
            right = n_rows
        batches.append(features[left: right, :])
    for batch in batches:
        dot_product = batch @ features.T  # 计算余弦距离
        selection = dot_product > sim_threshold
        for j in range(len(selection)):
            IDX = selection[j]  # 阈值之内的相似图片
            if np.sum(IDX) < min_indices:  # 如果相似的图片个数小于最少的个数，获得相似度最大的
                IDX = np.argsort(dot_product[j])[-min_indices:]
            y_pred.append(posting_ids[IDX].tolist())
    assert len(y_pred) == len(posting_ids)
    return y_pred


def find_threshold(df, features, thresholds=np.arange(30, 100, 5),):
    features = F.normalize(torch.from_numpy(features)).numpy()  # 对特征进行归一化
    best_score = 0
    best_threshold = -1
    best_y_pred = []
    posting_ids = df['posting_id']
    targets = df['targets'].apply(lambda x: x.split(" "))
    for threshold in thresholds:  # 计算动态阈值
        y_pred = find_matches(posting_ids, threshold, features, n_batches=10)
        scores = f1_score(targets, y_pred)
        precisions, recalls = precision_recall(targets, y_pred)
        df['score'] = scores
        df['precision'] = precisions
        df['recall'] = recalls
        selected_score = df['score'].mean()
        _p_mean = df['precision'].mean()
        _r_mean = df['recall'].mean()
        print(
            f"----------- valid f1: {selected_score} precision: {_p_mean} recall: {_r_mean} threshold: {threshold} ------------")
        if selected_score > best_score:
            best_score = selected_score
            best_threshold = threshold
            best_y_pred = y_pred
    return best_score, best_threshold, best_y_pred


def run_train():
    
    # config
    fold = 0
    margin = 0.8
    batch_size = 4
    arch = 'eca_nfnet_l1'
    initial_checkpoint = "/home/data_normal/abiz/wuzhiqiang/wzq/shopee/code_v3/result/image/0608/nfnet_margin/fold-0/checkpoint/00062000_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #-------------dataset-----------------------
    root = "/home/data_normal/abiz/wuzhiqiang/wzq/data/shopee-product-matching"
    df = pd.read_csv('folds.csv')
    df['image_path'] = df['image'].apply(lambda x: os.path.join(root, 'train_images', x))

    encoder = LabelEncoder()
    # df['label_group'] = encoder.fit_transform(df['label_group'])
    df_train = df[df['fold']!=fold].reset_index(drop=True)
    df_valid = df[df['fold']==fold].reset_index(drop=True)
    df_train['label_group'] = encoder.fit_transform(df_train['label_group'])
    df_valid['label_group'] = encoder.fit_transform(df_valid['label_group'])

    train_dataset = ShopeeDataset(df_train)
    valid_dataset = ShopeeDataset(df_valid)

    train_loader = data.DataLoader(
        train_dataset,
        sampler     = RandomSampler(train_dataset),
        batch_size  = batch_size,
        num_workers = 4,
        collate_fn  = collate_fn,
    )

    valid_loader = data.DataLoader(
        valid_dataset,
        shuffle     = False,
        batch_size  = batch_size,
        num_workers = 4,
        collate_fn  = collate_fn,
    )

    
    num_classes = len(train_dataset.df['label_group'].unique())
    print(f"train num of classes: {num_classes}\n")
    if arch == 'efficientnet_b3':
        net = ShopeeImgNet(num_classes=num_classes)
    elif arch == "eca_nfnet_l1":
        net = NFNet(num_classes=num_classes, margin=margin)
    net.to(device)

    if initial_checkpoint is not None:
        f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
        start_iteration = f['iteration']
        start_epoch = f['epoch']
        state_dict  = f['state_dict']
        net.load_state_dict(state_dict,strict=False)  #True
    else:
        start_iteration = 0
        start_epoch = 0


    iteration = start_iteration

    print(f"find threshold for iteration: {iteration}\n")
    features = extract_feat(net, valid_loader)
    find_threshold(df_valid, features)
    


if __name__ == '__main__':
    run_train()