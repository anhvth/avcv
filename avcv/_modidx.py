# Autogenerated by nbdev

d = { 'settings': { 'branch': 'main',
                'doc_baseurl': '/avcv/',
                'doc_host': 'https://anhvth.github.io',
                'git_url': 'https://github.com/anhvth/avcv/tree/nbdev-convert/',
                'lib_path': 'avcv'},
  'syms': { 'avcv.all': {},
            'avcv.cli': {'avcv.cli.convert_image': ('cli.html#convert_image', 'avcv/cli.py')},
            'avcv.coco': { 'avcv.coco.AvCOCO': ('coco_dataset.html#avcoco', 'avcv/coco.py'),
                           'avcv.coco.AvCOCO.__init__': ('coco_dataset.html#avcoco.__init__', 'avcv/coco.py'),
                           'avcv.coco.AvCOCO.createIndex': ('coco_dataset.html#avcoco.createindex', 'avcv/coco.py'),
                           'avcv.coco.AvCOCO.loadRes': ('coco_dataset.html#avcoco.loadres', 'avcv/coco.py'),
                           'avcv.coco.CocoDataset': ('coco_dataset.html#cocodataset', 'avcv/coco.py'),
                           'avcv.coco.CocoDataset.__init__': ('coco_dataset.html#cocodataset.__init__', 'avcv/coco.py'),
                           'avcv.coco.CocoDataset.__len__': ('coco_dataset.html#cocodataset.__len__', 'avcv/coco.py'),
                           'avcv.coco.CocoDataset.__str__': ('coco_dataset.html#cocodataset.__str__', 'avcv/coco.py'),
                           'avcv.coco.CocoDataset.evaluate': ('coco_dataset.html#cocodataset.evaluate', 'avcv/coco.py'),
                           'avcv.coco.CocoDataset.get_anns': ('coco_dataset.html#cocodataset.get_anns', 'avcv/coco.py'),
                           'avcv.coco.CocoDataset.get_image_path': ('coco_dataset.html#cocodataset.get_image_path', 'avcv/coco.py'),
                           'avcv.coco.CocoDataset.get_pair_img_anns': ('coco_dataset.html#cocodataset.get_pair_img_anns', 'avcv/coco.py'),
                           'avcv.coco.CocoDataset.image_paths': ('coco_dataset.html#cocodataset.image_paths', 'avcv/coco.py'),
                           'avcv.coco.CocoDataset.img_ids_with_ann': ('coco_dataset.html#cocodataset.img_ids_with_ann', 'avcv/coco.py'),
                           'avcv.coco.CocoDataset.imread': ('coco_dataset.html#cocodataset.imread', 'avcv/coco.py'),
                           'avcv.coco.CocoDataset.load_anns': ('coco_dataset.html#cocodataset.load_anns', 'avcv/coco.py'),
                           'avcv.coco.CocoDataset.visualize': ('coco_dataset.html#cocodataset.visualize', 'avcv/coco.py'),
                           'avcv.coco.DiagnoseCoco': ('coco_dataset.html#diagnosecoco', 'avcv/coco.py'),
                           'avcv.coco.DiagnoseCoco.find_false_samples': ( 'coco_dataset.html#diagnosecoco.find_false_samples',
                                                                          'avcv/coco.py'),
                           'avcv.coco._f': ('coco_dataset.html#_f', 'avcv/coco.py'),
                           'avcv.coco.bbox_expand': ('coco_dataset.html#bbox_expand', 'avcv/coco.py'),
                           'avcv.coco.check_save_coco_dict': ('coco_dataset.html#check_save_coco_dict', 'avcv/coco.py'),
                           'avcv.coco.concat_coco': ('coco_dataset.html#concat_coco', 'avcv/coco.py'),
                           'avcv.coco.concat_coco_v2': ('coco_dataset.html#concat_coco_v2', 'avcv/coco.py'),
                           'avcv.coco.extract_coco': ('coco_dataset.html#extract_coco', 'avcv/coco.py'),
                           'avcv.coco.get_bboxes': ('coco_dataset.html#get_bboxes', 'avcv/coco.py'),
                           'avcv.coco.get_overlap_rate': ('coco_dataset.html#get_overlap_rate', 'avcv/coco.py'),
                           'avcv.coco.split_coco': ('coco_dataset.html#split_coco', 'avcv/coco.py'),
                           'avcv.coco.to_jpg': ('coco_dataset.html#to_jpg', 'avcv/coco.py'),
                           'avcv.coco.v2c': ('coco_dataset.html#v2c', 'avcv/coco.py'),
                           'avcv.coco.video_to_coco': ('coco_dataset.html#video_to_coco', 'avcv/coco.py')},
            'avcv.debug': { 'avcv.debug.dpython': ('debug.html#dpython', 'avcv/debug.py'),
                            'avcv.debug.make_mini_coco': ('debug.html#make_mini_coco', 'avcv/debug.py')},
            'avcv.dist_utils': { 'avcv.dist_utils.all_gather': ('dist_utils.html#all_gather', 'avcv/dist_utils.py'),
                                 'avcv.dist_utils.all_gatherv': ('dist_utils.html#all_gatherv', 'avcv/dist_utils.py'),
                                 'avcv.dist_utils.all_reduce': ('dist_utils.html#all_reduce', 'avcv/dist_utils.py'),
                                 'avcv.dist_utils.gather_grad': ('dist_utils.html#gather_grad', 'avcv/dist_utils.py'),
                                 'avcv.dist_utils.get_rank': ('dist_utils.html#get_rank', 'avcv/dist_utils.py'),
                                 'avcv.dist_utils.get_world_size': ('dist_utils.html#get_world_size', 'avcv/dist_utils.py'),
                                 'avcv.dist_utils.primary': ('dist_utils.html#primary', 'avcv/dist_utils.py'),
                                 'avcv.dist_utils.rank0_to_all': ('dist_utils.html#rank0_to_all', 'avcv/dist_utils.py'),
                                 'avcv.dist_utils.reduce_loss_dict': ('dist_utils.html#reduce_loss_dict', 'avcv/dist_utils.py'),
                                 'avcv.dist_utils.reduce_sum': ('dist_utils.html#reduce_sum', 'avcv/dist_utils.py'),
                                 'avcv.dist_utils.setup_distributed': ('dist_utils.html#setup_distributed', 'avcv/dist_utils.py'),
                                 'avcv.dist_utils.synchronize': ('dist_utils.html#synchronize', 'avcv/dist_utils.py')},
            'avcv.utils': { 'avcv.utils.TimeLoger': ('utils.html#timeloger', 'avcv/utils.py'),
                            'avcv.utils.TimeLoger.__init__': ('utils.html#timeloger.__init__', 'avcv/utils.py'),
                            'avcv.utils.TimeLoger.__str__': ('utils.html#timeloger.__str__', 'avcv/utils.py'),
                            'avcv.utils.TimeLoger.start': ('utils.html#timeloger.start', 'avcv/utils.py'),
                            'avcv.utils.TimeLoger.update': ('utils.html#timeloger.update', 'avcv/utils.py'),
                            'avcv.utils.VideoReader': ('utils.html#videoreader', 'avcv/utils.py'),
                            'avcv.utils.VideoReader.__getitem__': ('utils.html#videoreader.__getitem__', 'avcv/utils.py'),
                            'avcv.utils.VideoReader.__init__': ('utils.html#videoreader.__init__', 'avcv/utils.py'),
                            'avcv.utils.VideoReader.__len__': ('utils.html#videoreader.__len__', 'avcv/utils.py'),
                            'avcv.utils.av_i2v': ('utils.html#av_i2v', 'avcv/utils.py'),
                            'avcv.utils.find_contours': ('utils.html#find_contours', 'avcv/utils.py'),
                            'avcv.utils.generate_tmp_filename': ('utils.html#generate_tmp_filename', 'avcv/utils.py'),
                            'avcv.utils.get_files': ('utils.html#get_files', 'avcv/utils.py'),
                            'avcv.utils.get_md5': ('utils.html#get_md5', 'avcv/utils.py'),
                            'avcv.utils.get_name': ('utils.html#get_name', 'avcv/utils.py'),
                            'avcv.utils.identify': ('utils.html#identify', 'avcv/utils.py'),
                            'avcv.utils.images_to_video': ('utils.html#images_to_video', 'avcv/utils.py'),
                            'avcv.utils.imemoize': ('utils.html#imemoize', 'avcv/utils.py'),
                            'avcv.utils.is_interactive': ('utils.html#is_interactive', 'avcv/utils.py'),
                            'avcv.utils.md5_from_str': ('utils.html#md5_from_str', 'avcv/utils.py'),
                            'avcv.utils.memoize': ('utils.html#memoize', 'avcv/utils.py'),
                            'avcv.utils.mkdir': ('utils.html#mkdir', 'avcv/utils.py'),
                            'avcv.utils.np_memmap_loader': ('utils.html#np_memmap_loader', 'avcv/utils.py'),
                            'avcv.utils.np_memmap_saver': ('utils.html#np_memmap_saver', 'avcv/utils.py'),
                            'avcv.utils.printc': ('utils.html#printc', 'avcv/utils.py'),
                            'avcv.utils.put_text': ('utils.html#put_text', 'avcv/utils.py'),
                            'avcv.utils.self_memoize': ('utils.html#self_memoize', 'avcv/utils.py'),
                            'avcv.utils.v2i': ('utils.html#v2i', 'avcv/utils.py'),
                            'avcv.utils.video_to_images': ('utils.html#video_to_images', 'avcv/utils.py')},
            'avcv.visualize': { 'avcv.visualize.Board': ('visualize.html#board', 'avcv/visualize.py'),
                                'avcv.visualize.Board.__call__': ('visualize.html#board.__call__', 'avcv/visualize.py'),
                                'avcv.visualize.Board.__init__': ('visualize.html#board.__init__', 'avcv/visualize.py'),
                                'avcv.visualize.Board.clear': ('visualize.html#board.clear', 'avcv/visualize.py'),
                                'avcv.visualize.Board.draw': ('visualize.html#board.draw', 'avcv/visualize.py'),
                                'avcv.visualize.Board.img_concat': ('visualize.html#board.img_concat', 'avcv/visualize.py'),
                                'avcv.visualize.Board.lazy_img_concat': ('visualize.html#board.lazy_img_concat', 'avcv/visualize.py'),
                                'avcv.visualize.Board.set_line_text': ('visualize.html#board.set_line_text', 'avcv/visualize.py'),
                                'avcv.visualize.Board.show': ('visualize.html#board.show', 'avcv/visualize.py'),
                                'avcv.visualize._detect_format': ('visualize.html#_detect_format', 'avcv/visualize.py'),
                                'avcv.visualize._to_bchw': ('visualize.html#_to_bchw', 'avcv/visualize.py'),
                                'avcv.visualize.bbox_visualize': ('visualize.html#bbox_visualize', 'avcv/visualize.py'),
                                'avcv.visualize.imshow': ('visualize.html#imshow', 'avcv/visualize.py'),
                                'avcv.visualize.plot_images': ('visualize.html#plot_images', 'avcv/visualize.py'),
                                'avcv.visualize.tensor2imgs': ('visualize.html#tensor2imgs', 'avcv/visualize.py'),
                                'avcv.visualize.tensor_to_image': ('visualize.html#tensor_to_image', 'avcv/visualize.py')}}}
