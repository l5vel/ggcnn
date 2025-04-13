def get_dataset(dataset_name):
    if dataset_name == 'cornell':
        from .cornell_data import CornellDataset
        return CornellDataset
    elif dataset_name == 'jacquard':
        from .jacquard_data import JacquardDataset
        return JacquardDataset
    elif dataset_name == 'nbmod':
        from .nbmod_data import NBModDataset
        return NBModDataset
    else:
        raise NotImplementedError('Dataset Type {} is Not implemented'.format(dataset_name))

if __name__ == '__main__':

    dataset_path = '/home/data/maa1446/nbmod/a_bunch_of_bananas'
    Dataset = get_dataset('nbmod')
    train_dataset = Dataset(dataset_path, start=0.0, end=0.9, ds_rotate=0,
                            random_rotate=True, random_zoom=True,
                            include_depth=1, include_rgb=0)