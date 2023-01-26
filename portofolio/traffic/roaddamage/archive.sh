torch-model-archiver --model-name roaddamage-${1} --version 1.0 --model-file ./model.py --serialized-file ${2} --force --handler ./roaddamage_handler.py --extra-files "transform.py,soft_nms.py,model_config.py,bench.py,anchors.py"