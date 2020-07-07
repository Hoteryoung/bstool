# from abc import ABCMeta, abstractmethod


# class BaseInstanceBuilding(metaclass=ABCMeta):

#     @abstractmethod
#     def rescale(self, scale, interpolation='nearest'):
#         pass

#     @abstractmethod
#     def resize(self, out_shape, interpolation='nearest'):
#         pass

#     @abstractmethod
#     def flip(self, flip_direction='horizontal'):
#         pass

#     @abstractmethod
#     def pad(self, out_shape, pad_val):
#         pass

#     @abstractmethod
#     def crop(self, bbox):
#         pass

#     @abstractmethod
#     def crop_and_resize(self,
#                         bboxes,
#                         out_shape,
#                         inds,
#                         interpolation='bilinear'):
#         pass

#     @abstractmethod
#     def expand(self, expanded_h, expanded_w, top, left):
#         pass

#     @property
#     @abstractmethod
#     def areas(self):
#         pass

#     @abstractmethod
#     def to_ndarray(self):
#         pass

#     @abstractmethod
#     def to_tensor(self, dtype, device):
#         pass


# class PolygonBuildings(BaseInstanceBuilding):
#     def __init__(self, buildings):
#         assert isinstance(buildings, )
#         if len(masks) > 0:
#             assert isinstance(masks[0], list)
#             assert isinstance(masks[0][0], np.ndarray)

#         self.height = height
#         self.width = width
#         self.masks = masks

#     def __getitem__(self, index):
#         if isinstance(index, np.ndarray):
#             index = index.tolist()
#         if isinstance(index, list):
#             masks = [self.masks[i] for i in index]
#         else:
#             try:
#                 masks = self.masks[index]
#             except Exception:
#                 raise ValueError(
#                     f'Unsupported input of type {type(index)} for indexing!')
#         if isinstance(masks[0], np.ndarray):
#             masks = [masks]  # ensure a list of three levels
#         return PolygonMasks(masks, self.height, self.width)

#     def __iter__(self):
#         return iter(self.masks)

#     def __repr__(self):
#         s = self.__class__.__name__ + '('
#         s += f'num_masks={len(self.masks)}, '
#         s += f'height={self.height}, '
#         s += f'width={self.width})'
#         return s

#     def __len__(self):
#         return len(self.masks)



    


# def polygon_to_bitmap(polygons, height, width):
#     """Convert masks from the form of polygons to bitmaps.

#         Args:
#             polygons (list[ndarray]): masks in polygon representation
#             height (int): mask height
#             width (int): mask width

#         Return:
#             ndarray: the converted masks in bitmap representation
#     """
#     rles = maskUtils.frPyObjects(polygons, height, width)
#     rle = maskUtils.merge(rles)
#     bitmap_mask = maskUtils.decode(rle).astype(np.bool)
#     return bitmap_mask
