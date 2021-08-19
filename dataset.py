from common import *
from transformers import AutoTokenizer


width, height = 640, 640
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def collate_fn(batch):
    batch_size = len(batch)
    image = []
    label = []
    index = []
    input_ids = []
    token_type_ids = []
    attention_mask = []
    for r in batch:
        image.append((r['image'] / 255.0 - mean) / std)
        label.append(r['label'])
        index.append(r['index'])
        input_ids.append(r['items']['input_ids'])
        token_type_ids.append(r['items']['token_type_ids'])
        attention_mask.append(r['items']['attention_mask'])
    
    label = np.stack(label)
    image = np.stack(image)
    input_ids = torch.stack(input_ids)  # 返回的tokens
    token_type_ids = torch.stack(token_type_ids)  
    attention_mask = torch.stack(attention_mask)  # 返回的attention_mask

    image = image.transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).contiguous().float()
    label = torch.from_numpy(label).contiguous().long()

    return {
        'index': index,
        'label': label,
        'image': image,
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
    }


def title_collate_fn(batch):
    batch_size = len(batch)
    label = []
    index = []
    input_ids = []
    token_type_ids = []
    attention_mask = []
    for r in batch:
        label.append(r['label'])
        index.append(r['index'])
        input_ids.append(r['items']['input_ids'])
        token_type_ids.append(r['items']['token_type_ids'])
        attention_mask.append(r['items']['attention_mask'])
    
    label = np.stack(label)
    input_ids = torch.stack(input_ids)  # 返回的tokens
    token_type_ids = torch.stack(token_type_ids)  
    attention_mask = torch.stack(attention_mask)  # 返回的attention_mask

    label = torch.from_numpy(label).contiguous().long()

    return {
        'index': index,
        'label': label,
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
    }


null_augment = A.Compose(
    [A.Resize(width, height)]
)

train_augment = A.Compose([
    A.Resize(width, height),
    A.HorizontalFlip(p=0.5),
    # A.VerticalFlip(p=0.5),
    ])


def make_fold(mode='train-0'):
    if 'train' in mode:
        df = pd.read_csv('./folds.csv')
        df['fold'] = df['fold'].astype(int)

        fold = int(mode[-1])
        df_train = df[df.fold != fold].reset_index(drop=True)  # train data
        df_valid = df[df.fold == fold].reset_index(drop=True)  # valid data
        return df_train, df_valid


class ShopeeDataset(data.Dataset):
    def __init__(self, df, augment=null_augment, bert_model_arch="bert-base-multilingual-uncased"):
        self.df = df
        self.augment = augment
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_arch)
        texts = df['title'].fillna("NaN").tolist()  # 所有标题生成的文本列表
        self.encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
        )


    def __getitem__(self, index):
        posting_info = self.df.iloc[index]
        # title = posting_info.title
        image_path = posting_info.image_path

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)

        label = posting_info.label_group
        items = {k: torch.tensor(v[index]) for k, v in self.encodings.items()}
        
        if self.augment:
            augmented = self.augment(image=image)
            image = augmented['image']

        r = {
            'index': index,
            'image': image,
            'label': label,
            'items': items
        }
        return r

    def __len__(self):
        return self.df.shape[0]

    def __str__(self):
        string = ''
        string += '\tlen     = %d\n' % len(self)
        string += '\tlabel   = %d\n' % len(self.df['label_group'].unique())
        return string


class ShopeeTitleDataset(data.Dataset):
    def __init__(self, df, bert_model_arch="bert-base-multilingual-uncased"):
        # 文本有增广吗？
        # <todo>
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_arch)
        texts = df['title'].fillna("NaN").tolist()  # 所有标题生成的文本列表
        self.encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
        )


    def __getitem__(self, index):
        posting_info = self.df.iloc[index]
        label = posting_info.label_group
        items = {k: torch.tensor(v[index]) for k, v in self.encodings.items()}
        r = {
            'index': index,
            'label': label,
            'items': items
        }
        return r

    def __len__(self):
        return self.df.shape[0]

    def __str__(self):
        string = ''
        string += '\tlen     = %d\n' % len(self)
        string += '\tlabel   = %d\n' % len(self.df['label_group'].unique())
        return string


def run_check_dataset():
    df_train, df_valid = make_fold('train-0')
    encoder = LabelEncoder()
    df_train['label_group'] = encoder.fit_transform(df_train['label_group'])
    df_valid['label_group'] = encoder.fit_transform(df_valid['label_group'])
    train_data = ShopeeDataset(df_train)
    valid_data = ShopeeDataset(df_valid)
    print(train_data)
    print(valid_data)
    for i in range(100):
        r = train_data[i]
        print(r['index'])
        print(r['label'])
        print(r['items'])
        print('image : ')
        print('\t', r['image'].shape)
        print('')
    
    train_loader = data.DataLoader(
        train_data,
        sampler = RandomSampler(train_data),
        batch_size = 8,
        drop_last = True,
        num_workers = 0,
        collate_fn  = collate_fn,
    )
    for t, batch in enumerate(train_loader):
        if t > 30:
            break
        print(t, "---------------")
        print(batch['image'].shape, batch['image'].is_contiguous())
        print(batch['label'].shape, batch['label'].is_contiguous())
        # print(batch[''])
        print(batch['input_ids'].shape, batch['input_ids'].is_contiguous())
        print(batch['token_type_ids'].shape, batch['token_type_ids'].is_contiguous())
        print(batch['attention_mask'].shape, batch['attention_mask'].is_contiguous())
        print(' ')

def run_check_augment():
    df_train, df_valid = make_fold('train-0')
    encoder = LabelEncoder()
    df_train['label_group'] = encoder.fit_transform(df_train['label_group'])
    df_valid['label_group'] = encoder.fit_transform(df_valid['label_group'])
    train_data = ShopeeDataset(df_train, train_augment)
    print(train_data)

    for i in range(20):
        i = np.random.choice(len(train_data))
        r = train_data[i]
        image = r['image']
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"./augmented/{i}.png", image)

def run_check_title_dataset():
    df_train, df_valid = make_fold('train-0')
    encoder = LabelEncoder()
    df_train['label_group'] = encoder.fit_transform(df_train['label_group'])
    df_valid['label_group'] = encoder.fit_transform(df_valid['label_group'])
    train_data = ShopeeTitleDataset(df_train)
    valid_data = ShopeeTitleDataset(df_valid)
    print(train_data)
    print(valid_data)
    for i in range(100):
        r = train_data[i]
        print(r['index'])
        print(r['label'])
        print(r['items'])
    
    train_loader = data.DataLoader(
        train_data,
        sampler = RandomSampler(train_data),
        batch_size = 8,
        drop_last = True,
        num_workers = 0,
        collate_fn  = title_collate_fn,
    )
    for t, batch in enumerate(train_loader):
        if t > 30:
            break
        print(t, "---------------")
        print(batch['label'].shape, batch['label'].is_contiguous())
        # print(batch[''])
        print(batch['input_ids'].shape, batch['input_ids'].is_contiguous())
        print(batch['token_type_ids'].shape, batch['token_type_ids'].is_contiguous())
        print(batch['attention_mask'].shape, batch['attention_mask'].is_contiguous())
        print(' ')

if __name__ == "__main__":
    # run_check_dataset()
    run_check_title_dataset()
    # run_check_augment()