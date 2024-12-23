Datasets
========

This module contains classes for loading the `7-Scenes <https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/>`_
and `Cambridge Landmarks <https://www.repository.cam.ac.uk/items/53788265-cb98-42ee-b85b-7a0cbc8eddb3>`_ datasets,
as well as instructions on how to download the datasets.

Download
------------

The datasets can be downloaded from the following command from the `FaVoR` repository:

```bash
bash scripts/download_datasets.sh
```

This script will create the folder ``datasets`` and download the datasets,
the `NetVLAD matches <https://cvg-data.inf.ethz.ch/pixloc_CVPR2021/>`_, and
the `COLMAP ground truth <https://github.com/tsattler/visloc_pseudo_gt_limitations/tree/main>`_ for the 7-Scenes dataset.

Dataloader Base Class
---------------------

.. automodule:: lib.data_loaders.Dataloader
    :members:
    :undoc-members:
    :show-inheritance:

Dataloader for Cambridge Landmarks Dataset
------------------------------------------

The class :class:`lib.data_loaders.CambridgeLandmarksDataloader` is a subclass of :class:`lib.data_loaders.Dataloader` and is used to load the Cambridge Landmarks Dataset.

Dataloader for 7Scenes Dataset
-------------------------------

The class :class:`lib.data_loaders.SevenScenesDataloader` is a subclass of :class:`lib.data_loaders.Dataloader` and is used to load the 7Scenes Dataset.
