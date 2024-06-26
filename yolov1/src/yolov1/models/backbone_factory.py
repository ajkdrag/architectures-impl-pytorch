from typing import Dict, Type

import timm


class BackboneFactory:
    registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str):
        def inner_wrapper(wrapped_class: Type):
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create_backbone(cls, name: str, **kwargs):
        return cls.registry[name](**kwargs)


@BackboneFactory.register("resnet50")
def resnet50(pretrained: bool = True, **kwargs):
    model = timm.create_model("resnet50", pretrained=pretrained, **kwargs)
    model.num_features = model.feature_info[-1]["num_chs"]
    model.scale_down_factor = model.feature_info[-1]["reduction"]
    return model


@BackboneFactory.register("resnet34")
def resnet34(pretrained: bool = True, **kwargs):
    model = timm.create_model("resnet34", pretrained=pretrained, **kwargs)
    model.num_features = model.feature_info[-1]["num_chs"]
    model.scale_down_factor = model.feature_info[-1]["reduction"]
    return model


@BackboneFactory.register("resnet18")
def resnet18(pretrained: bool = True, **kwargs):
    model = timm.create_model("resnet18", pretrained=pretrained, **kwargs)
    model.num_features = model.feature_info[-1]["num_chs"]
    model.scale_down_factor = model.feature_info[-1]["reduction"]
    return model
