from os import path
from tqdm import tqdm


class Dataloader:

    def __init__(self, data_path, scene):
        self.data_path = data_path
        self.scene = scene
        self.scene_folder = path.join(data_path, scene)
        self.gt_lines = []
        self.test_lines = []
        self.camera = None

        self.load_data()

    def load_data(self):
        raise NotImplementedError

    def get_train(self):
        for line in tqdm(self.gt_lines):
            yield self.line2data(line)

    def get_test(self):
        for line in tqdm(self.gt_lines):
            yield self.line2data(line)

    def line2data(self, line):
        raise NotImplementedError
