Extractors
==========

This module contains the feature extractors used in the project. The extractors are used to extract features from images. The extracted features are used to match images and to estimate the pose of the camera.


Base Extractor
--------------

.. automodule:: lib.feature_extractors.base_extractor
    :members:
    :show-inheritance:

ALIKE
-----

This extractor is based on the ALIKE feature extractor. This class is a subclass of the :class:`lib.feature_extractors.base_extractor.BaseExtractor` class.

SuperPoint
----------

This extractor is based on the SuperPoint feature extractor. This class is a subclass of the :class:`lib.feature_extractors.base_extractor.BaseExtractor` class.