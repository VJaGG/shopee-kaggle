from model import *
from common import *
from dataset import *
import os
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
            input_ids = batch['input_ids'].cuda()
            # token_type_ids = batch['token_type_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            feat, _ = net(input_ids, attention_mask)
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
        dot_product = batch @ features.T
        selection = dot_product > sim_threshold
        for j in range(len(selection)):
            IDX = selection[j]  # 阈值之内的相似图片
            if np.sum(IDX) < min_indices:
                IDX = np.argsort(dot_product[j])[-min_indices:]
            y_pred.append(posting_ids[IDX].tolist())
    assert len(y_pred) == len(posting_ids)
    return y_pred


def find_threshold(df, features, log, thresholds=np.arange(40, 100, 5),):
    features = F.normalize(torch.from_numpy(features)).numpy()
    best_score = 0
    best_threshold = -1
    best_y_pred = []
    posting_ids = df['posting_id']
    targets = df['targets'].apply(lambda x: x.split(" "))
    for threshold in thresholds:
        y_pred = find_matches(posting_ids, threshold, features, n_batches=10)
        scores = f1_score(targets, y_pred)
        precisions, recalls = precision_recall(targets, y_pred)
        df['score'] = scores
        df['precision'] = precisions
        df['recall'] = recalls
        selected_score = df['score'].mean()
        _p_mean = df['precision'].mean()
        _r_mean = df['recall'].mean()
        log.write(
            f"----------- valid f1: {selected_score} precision: {_p_mean} recall: {_r_mean} threshold: {threshold} ------------\n")
        if selected_score > best_score:
            best_score = selected_score
            best_threshold = threshold
            best_y_pred = y_pred
    return best_score, best_threshold, best_y_pred


def run_train(config):
    
    # config
    initial_checkpoint = config['initial_checkpoint']
    # weight_decay = config['weight_decay']
    start_lr = config['start_lr']
    fold = config['fold']
    batch_size = config['batch_size']
    out_dir = config['out_dir'] + 'fold-%d' % fold
    SCHEDULER = config['scheduler']
    arch = config['arch']
    dynamic_margin = config['dynamic_margin']
    margin = config['margin']


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for f in ['checkpoint', 'valid']:
        os.makedirs(out_dir + '/' + f, exist_ok=True)
    log = Logger()
    log.open(out_dir + '/log.train.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\t__file__ = %s\n' % __file__)
    log.write('\tout_dir  = %s\n' % out_dir)
    log.write('\n')

    #-------------dataset-----------------------
    root = "/home/data_normal/abiz/wuzhiqiang/wzq/data/shopee-product-matching"
    df = pd.read_csv('folds.csv')
    df['image_path'] = df['image'].apply(lambda x: os.path.join(root, 'train_images', x))

    encoder = LabelEncoder()

    df_train = df[df['fold']!=fold].reset_index(drop=True)
    df_valid = df[df['fold']==fold].reset_index(drop=True)
    df_train['label_group'] = encoder.fit_transform(df_train['label_group'])
    df_valid['label_group'] = encoder.fit_transform(df_valid['label_group'])

    train_dataset = ShopeeTitleDataset(df_train)

    train_loader = data.DataLoader(
        train_dataset,
        sampler     = RandomSampler(train_dataset),
        batch_size  = batch_size,
        num_workers = 0,
        collate_fn  = title_collate_fn,
    )


    log.write('root = %s\n' % str(root))
    log.write('train_dataset : \n%s\n' % (train_dataset))
    # log.write('valid_dataset : \n%s\n' % (valid_dataset))
    log.write('\n')


    ## net ----------------------------------------
    log.write('** net setting **\n')
    num_classes = len(train_dataset.df['label_group'].unique())
    log.write(f"train num of classes: {num_classes}\n")
    log.write(f'margin: {margin}\n')
    log.write(f'dynamic margin: {dynamic_margin}\n')

    if arch == 'bert':
        net = ShopeeTextNet(num_classes=num_classes)
    elif arch == 'bert_margin':
        net = BERTMargin(num_classes=num_classes)
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

    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    log.write('\n')


    optimizer = None
    if config['layer']:
        bert_params = list(map(id, net.bert_model.parameters()))
        head_params = filter(lambda p: id(p) not in bert_params, net.parameters())
        params = [{'params': net.bert_model.parameters()}, {'params': head_params, 'lr': start_lr * 10}]
    else:
        params = net.parameters()

    optimizer = None
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(params, lr=start_lr)

    scheduler = None
    if SCHEDULER == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         factor=factor, patience=patience,
                                                         verbose=True, eps=eps)
    elif SCHEDULER == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=min_lr,
                                                         last_epoch=-1)
    elif SCHEDULER == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                   T_0=T_0, 
                                                                   T_mult=1,
                                                                   eta_min=min_lr,
                                                                   last_epoch=-1)

    num_iteration = 80000 * 1000
    iter_log   = 500  # print log
    iter_valid = 500  # validation
    iter_save  = list(range(0, num_iteration, 2000))

    log.write('optimizer\n  %s\n' % (optimizer))
    log.write('\n')


    # -------------------start training here!----------------------
    log.write('** start training here **\n')
    log.write('   batch_size = %d\n' % (batch_size))
    log.write('   experiment = %s\n' % str(__file__.split('/')[-2:]))
    log.write('                               |----- VALID -----|----- TRAIN/BATCH -----\n')
    log.write('rate     margin   iter   epoch | loss    acc     |  loss0  | time        \n')
    log.write('-------------------------------------------------------------------------\n')


    def message(mode='print'):
        if mode == ('print'):
            asterisk = ' '
            loss = batch_loss
        
        if mode == ('log'):
            asterisk = '*' if iteration in iter_save else ' '
            loss = train_loss
        
        text = \
            '%0.5f  %0.5f  %5.2f%s %4.2f  | ' % (rate, margin, iteration / 10000, asterisk, epoch,) +\
            '%6.3f  %6.3f  | ' % (*valid_loss, ) + \
            '%6.3f  | ' % (*loss, ) + \
            '%s' % (time_to_str(timer() - start_timer, 'min'))
        return text 

    # ---------
    valid_loss = np.zeros(2, np.float32)
    train_loss = np.zeros(1, np.float32)
    batch_loss = np.zeros_like(train_loss)
    sum_train_loss = np.zeros_like(train_loss)
    sum_train = 0

    start_timer = timer()
    epoch = start_epoch
    iteration = start_iteration
    rate = 0

    criteria = nn.CrossEntropyLoss()

    while iteration < num_iteration:

        # features = extract_feat(net, valid_loader)
        # find_threshold(df_valid, features, log)

        for t, batch in enumerate(train_loader):

            # if iteration % iter_valid == 0:
            #     valid_loss = do_valid(net, valid_loader)
            
            if iteration % iter_log == 0:
                print('\r', end='', flush=True)
                log.write(message(mode='log') + '\n')
            
            if iteration in iter_save:
                if iteration != start_iteration:
                    torch.save({
                        'state_dict': net.state_dict(),
                        'iteration': iteration,
                        'epoch': epoch,
                    }, out_dir + '/checkpoint/%08d_model.pth' % (iteration))
                    pass
            
            # learning rate schduler 
            rate = get_learning_rate(optimizer)

            # one iteration update
            batch_size = len(batch['index'])
            label = batch['label'].cuda()
            # image = batch['image'].cuda()
            input_ids = batch['input_ids'].cuda()
            token_type_ids = batch['token_type_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()

            net.train()
            optimizer.zero_grad()
            if dynamic_margin:
                margin =  (iteration // 10000) * 0.1 + 0.2
                logit = net(input_ids, attention_mask, label, margin)
                loss0 = criteria(logit, label)
            else:
                logit = net(input_ids, attention_mask, label, margin)
                loss0 = criteria(logit, label)
            (loss0).backward()
            optimizer.step()
            
            epoch += 1 / len(train_loader)
            iteration += 1
            batch_loss = np.array([loss0.item()])
            sum_train_loss += batch_loss
            sum_train += 1
            if iteration % 100 == 0:
                train_loss = sum_train_loss / (sum_train + 1e-12)
                sum_train_loss[...] = 0
                sum_train = 0
            
            print('\r', end='', flush=True)
            print(message(mode='print'), end='', flush=True)
        if scheduler is not None:
            scheduler.step()
    log.write('\n')


if __name__ == '__main__':
    with open('./scripts/0text_bert_margin.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    print(config)
    run_train(config)