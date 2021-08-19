def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    
    # assert (len(lr) == 1)
    lr = lr[0]
    return lr