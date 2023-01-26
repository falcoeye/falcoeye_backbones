torch-model-archiver --model-name detectwaste --version 1.0 --model-file ./model.py --serialized-file ./detect-waste.pth --force --handler ./detectwaste_handler.py --extra-files "soft_nms.py,model_config.py,fpn_config.py,config_utils.py,bench.py,matcher.py,box_list.py,region_similarity_calculator.py,box_coder.py,argmax_matcher.py,anchors.py,target_assigner.py"