from typing import Optional
from dataclasses import dataclass

@dataclass
class Config_MobileViT_v1_XXS:
    block_1_1_dims: int = 16
    block_1_2_dims: int = 16

    block_2_1_dims: int = 24
    block_2_2_dims: int = 24
    block_2_3_dims: int = 24

    block_3_1_dims: int = 48
    block_3_2_dims: int = 48

    block_4_1_dims: int = 64
    block_4_2_dims: int = 64

    block_5_1_dims: int = 80
    block_5_2_dims: int = 80

    final_conv_dims: int = 320

    tf_block_3_dims: int = 64
    tf_block_4_dims: int = 80
    tf_block_5_dims: int = 96

    tf_block_3_repeats: int = 2
    tf_block_4_repeats: int = 4
    tf_block_5_repeats: int = 3

    depthwise_expansion_factor: int = 2


def get_mobile_vit_v1_configs(model_type: str = "S", updates: Optional[dict] = None):
    #remove model type argument as only XXS is supported
    base_config = Config_MobileViT_v1_XXS #removed configs for s and xs

    if updates:
        return base_config(**updates)

    return base_config