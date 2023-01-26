"""EfficientDet Configurations

Adapted from official impl at https://github.com/google/automl/tree/master/efficientdet

TODO use a different config system (OmegaConfig -> Hydra?), separate model from train specific hparams
"""

from omegaconf import OmegaConf
from copy import deepcopy


def default_detection_model_configs():
    """Returns a default detection configs."""
    h = OmegaConf.create()

    # model name.
    h.name = 'tf_efficientdet_d1'

    h.backbone_name = 'tf_efficientnet_b1'
    h.backbone_args = None  # FIXME sort out kwargs vs config for backbone creation

    # model specific, input preprocessing parameters
    h.image_size = (640, 640)

    # dataset specific head parameters
    h.num_classes = 90

    # feature + anchor config
    h.min_level = 3
    h.max_level = 7
    h.num_levels = h.max_level - h.min_level + 1
    h.num_scales = 3
    h.aspect_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    # ratio w/h: 2.0 means w=1.4, h=0.7. Can be computed with k-mean per dataset.
    #h.aspect_ratios = [1.0, 2.0, 0.5]
    h.anchor_scale = 4.0

    # FPN and head config
    h.pad_type = 'same'  # original TF models require an equivalent of Tensorflow 'SAME' padding
    h.act_type = 'swish'
    h.norm_layer = None  # defaults to batch norm when None
    h.norm_kwargs = dict(eps=.001, momentum=.01)
    h.box_class_repeats = 3
    h.fpn_cell_repeats = 3
    h.fpn_channels = 88
    h.separable_conv = True
    h.apply_bn_for_resampling = True
    h.conv_after_downsample = False
    h.conv_bn_relu_pattern = False
    h.use_native_resize_op = False
    h.pooling_type = None
    h.redundant_bias = True  # original TF models have back to back bias + BN layers, not necessary!
    h.head_bn_level_first = False  # change order of BN in head repeat list of lists, True for torchscript compat

    h.fpn_name = None
    h.fpn_config = None
    h.fpn_drop_path_rate = 0.  # No stochastic depth in default. NOTE not currently used, unstable training

    # classification loss (used by train bench)
    h.alpha = 0.25
    h.gamma = 1.5
    h.label_smoothing = 0.  # only supported if new_focal == True
    h.new_focal = False  # use new focal loss (supports label smoothing but uses more mem, less optimal w/ jit script)
    h.jit_loss = False  # torchscript jit for loss fn speed improvement, can impact stability and/or increase mem usage

    # localization loss (used by train bench)
    h.delta = 0.1
    h.box_loss_weight = 50.0

    return h


efficientdet_model_param_dict = dict(
    # Models with PyTorch friendly padding and my PyTorch pretrained backbones, training TBD
    efficientdet_d2=dict(
        name='efficientdet_d2',
        backbone_name='efficientnet_b2',
        image_size=(768, 768),
        fpn_channels=112,
        fpn_cell_repeats=5,
        box_class_repeats=3,
        pad_type='',
        redundant_bias=False,
        backbone_args=dict(drop_path_rate=0.2),
        url='',  # no pretrained weights yet
    )
)


def get_efficientdet_config(model_name='tf_efficientdet_d1'):
    """Get the default config for EfficientDet based on model name."""
    h = default_detection_model_configs()
    h.update(efficientdet_model_param_dict[model_name])
    h.num_levels = h.max_level - h.min_level + 1
    return deepcopy(h)  # may be unnecessary, ensure no references to param dict values
