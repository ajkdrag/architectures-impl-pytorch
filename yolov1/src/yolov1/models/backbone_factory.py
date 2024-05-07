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


@BackboneFactory.register("resnet34")
def resnet34(pretrained: bool = True, **kwargs):
    model = timm.create_model("resnet34", pretrained=pretrained, **kwargs)
    model.num_features = 512
    return model


@BackboneFactory.register("resnet18")
def resnet18(pretrained: bool = True, **kwargs):
    model = timm.create_model("resnet18", pretrained=pretrained, **kwargs)
    model.num_features = 512
    return model
