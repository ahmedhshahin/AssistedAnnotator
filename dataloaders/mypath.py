
class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'pascal':
            return '/home/ahmed/Documents/VOC2012'  # folder that contains VOCdevkit/.

        elif database == 'sbd'or database == 'lv':
            return '/home/ahmed/Documents'  # folder with img/, inst/, cls/, etc.
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def models_dir():
        return 'models/'
