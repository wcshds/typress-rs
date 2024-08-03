import json
from transformers import VisionEncoderDecoderModel

model = VisionEncoderDecoderModel.from_pretrained('paran3xus/typst_eq_ocr')

#################### encoder ####################
with open("./deit_model.json") as f:
    data = json.load(f)

data["item"]["embed"]["patch_embed"]["proj"]["weight"]["param"]["bytes"] = list(model.encoder.embeddings.patch_embeddings.projection.weight.data.numpy().tobytes())
data["item"]["embed"]["patch_embed"]["proj"]["bias"]["param"]["bytes"] = list(model.encoder.embeddings.patch_embeddings.projection.bias.data.numpy().tobytes())
data["item"]["embed"]["cls_token"]["param"]["bytes"] = list(model.encoder.embeddings.cls_token.data.numpy().tobytes())
data["item"]["embed"]["distillation_token"]["param"]["bytes"] = list(model.encoder.embeddings.distillation_token.data.numpy().tobytes())
data["item"]["embed"]["position_embed"]["param"]["bytes"] = list(model.encoder.embeddings.position_embeddings.data.numpy().tobytes())

for i in range(len(model.encoder.encoder.layer)):
    data["item"]["encoder"]["layers"][i]["attention"]["query"]["weight"]["param"]["bytes"] = list(model.encoder.encoder.layer[i].attention.attention.query.weight.data.transpose(0, 1).numpy().tobytes())
    data["item"]["encoder"]["layers"][i]["attention"]["query"]["bias"]["param"]["bytes"] = list(model.encoder.encoder.layer[i].attention.attention.query.bias.data.numpy().tobytes())
    data["item"]["encoder"]["layers"][i]["attention"]["key"]["weight"]["param"]["bytes"] = list(model.encoder.encoder.layer[i].attention.attention.key.weight.data.transpose(0, 1).numpy().tobytes())
    data["item"]["encoder"]["layers"][i]["attention"]["key"]["bias"]["param"]["bytes"] = list(model.encoder.encoder.layer[i].attention.attention.key.bias.data.numpy().tobytes())
    data["item"]["encoder"]["layers"][i]["attention"]["value"]["weight"]["param"]["bytes"] = list(model.encoder.encoder.layer[i].attention.attention.value.weight.data.transpose(0, 1).numpy().tobytes())
    data["item"]["encoder"]["layers"][i]["attention"]["value"]["bias"]["param"]["bytes"] = list(model.encoder.encoder.layer[i].attention.attention.value.bias.data.numpy().tobytes())
    data["item"]["encoder"]["layers"][i]["attention"]["output"]["weight"]["param"]["bytes"] = list(model.encoder.encoder.layer[i].attention.output.dense.weight.data.transpose(0, 1).numpy().tobytes())
    data["item"]["encoder"]["layers"][i]["attention"]["output"]["bias"]["param"]["bytes"] = list(model.encoder.encoder.layer[i].attention.output.dense.bias.data.numpy().tobytes())

    data["item"]["encoder"]["layers"][i]["intermediate"]["dense"]["weight"]["param"]["bytes"] = list(model.encoder.encoder.layer[i].intermediate.dense.weight.data.transpose(0, 1).numpy().tobytes())
    data["item"]["encoder"]["layers"][i]["intermediate"]["dense"]["bias"]["param"]["bytes"] = list(model.encoder.encoder.layer[i].intermediate.dense.bias.data.numpy().tobytes())

    data["item"]["encoder"]["layers"][i]["output"]["dense"]["weight"]["param"]["bytes"] = list(model.encoder.encoder.layer[i].output.dense.weight.data.transpose(0, 1).numpy().tobytes())
    data["item"]["encoder"]["layers"][i]["output"]["dense"]["bias"]["param"]["bytes"] = list(model.encoder.encoder.layer[i].output.dense.bias.data.numpy().tobytes())

    data["item"]["encoder"]["layers"][i]["layernorm_before"]["gamma"]["param"]["bytes"] = list(model.encoder.encoder.layer[i].layernorm_before.weight.data.numpy().tobytes())
    data["item"]["encoder"]["layers"][i]["layernorm_before"]["beta"]["param"]["bytes"] = list(model.encoder.encoder.layer[i].layernorm_before.bias.data.numpy().tobytes())

    data["item"]["encoder"]["layers"][i]["layernorm_after"]["gamma"]["param"]["bytes"] = list(model.encoder.encoder.layer[i].layernorm_after.weight.data.numpy().tobytes())
    data["item"]["encoder"]["layers"][i]["layernorm_after"]["beta"]["param"]["bytes"] = list(model.encoder.encoder.layer[i].layernorm_after.bias.data.numpy().tobytes())

data["item"]["layernorm"]["gamma"]["param"]["bytes"] = list(model.encoder.layernorm.weight.data.numpy().tobytes())
data["item"]["layernorm"]["beta"]["param"]["bytes"] = list(model.encoder.layernorm.bias.data.numpy().tobytes())

with open("./deit_model.json", 'w') as f:
    json.dump(data, f)

#################### decoder ####################

with open("./decoder.json") as f:
    data = json.load(f)

data["item"]["model"]["embed_tokens"]["embed"]["weight"]["param"]["bytes"] = list(model.decoder.model.decoder.embed_tokens.weight.data.numpy().tobytes())
data["item"]["model"]["embed_positions"]["embed"]["weight"]["param"]["bytes"] = list(model.decoder.model.decoder.embed_positions.weight.data.numpy().tobytes())
data["item"]["model"]["layernorm_embed"]["gamma"]["param"]["bytes"] = list(model.decoder.model.decoder.layernorm_embedding.weight.data.numpy().tobytes())
data["item"]["model"]["layernorm_embed"]["beta"]["param"]["bytes"] = list(model.decoder.model.decoder.layernorm_embedding.bias.data.numpy().tobytes())

for i in range(len(model.decoder.model.decoder.layers)):
    # self_attn
    data["item"]["model"]["layers"][i]["self_attn"]["q_proj"]["weight"]["param"]["bytes"] = list(model.decoder.model.decoder.layers[i].self_attn.q_proj.weight.data.transpose(0, 1).numpy().tobytes())
    data["item"]["model"]["layers"][i]["self_attn"]["q_proj"]["bias"]["param"]["bytes"] = list(model.decoder.model.decoder.layers[i].self_attn.q_proj.bias.data.numpy().tobytes())
    data["item"]["model"]["layers"][i]["self_attn"]["k_proj"]["weight"]["param"]["bytes"] = list(model.decoder.model.decoder.layers[i].self_attn.k_proj.weight.data.transpose(0, 1).numpy().tobytes())
    data["item"]["model"]["layers"][i]["self_attn"]["k_proj"]["bias"]["param"]["bytes"] = list(model.decoder.model.decoder.layers[i].self_attn.k_proj.bias.data.numpy().tobytes())
    data["item"]["model"]["layers"][i]["self_attn"]["v_proj"]["weight"]["param"]["bytes"] = list(model.decoder.model.decoder.layers[i].self_attn.v_proj.weight.data.transpose(0, 1).numpy().tobytes())
    data["item"]["model"]["layers"][i]["self_attn"]["v_proj"]["bias"]["param"]["bytes"] = list(model.decoder.model.decoder.layers[i].self_attn.v_proj.bias.data.numpy().tobytes())
    data["item"]["model"]["layers"][i]["self_attn"]["out_proj"]["weight"]["param"]["bytes"] = list(model.decoder.model.decoder.layers[i].self_attn.out_proj.weight.data.transpose(0, 1).numpy().tobytes())
    data["item"]["model"]["layers"][i]["self_attn"]["out_proj"]["bias"]["param"]["bytes"] = list(model.decoder.model.decoder.layers[i].self_attn.out_proj.bias.data.numpy().tobytes())
    # self_attn_layer_norm
    data["item"]["model"]["layers"][i]["self_attn_layer_norm"]["gamma"]["param"]["bytes"] = list(model.decoder.model.decoder.layers[i].self_attn_layer_norm.weight.data.numpy().tobytes())
    data["item"]["model"]["layers"][i]["self_attn_layer_norm"]["beta"]["param"]["bytes"] = list(model.decoder.model.decoder.layers[i].self_attn_layer_norm.bias.data.numpy().tobytes())
    # encoder_attn
    data["item"]["model"]["layers"][i]["encoder_attn"]["q_proj"]["weight"]["param"]["bytes"] = list(model.decoder.model.decoder.layers[i].encoder_attn.q_proj.weight.data.transpose(0, 1).numpy().tobytes())
    data["item"]["model"]["layers"][i]["encoder_attn"]["q_proj"]["bias"]["param"]["bytes"] = list(model.decoder.model.decoder.layers[i].encoder_attn.q_proj.bias.data.numpy().tobytes())
    data["item"]["model"]["layers"][i]["encoder_attn"]["k_proj"]["weight"]["param"]["bytes"] = list(model.decoder.model.decoder.layers[i].encoder_attn.k_proj.weight.data.transpose(0, 1).numpy().tobytes())
    data["item"]["model"]["layers"][i]["encoder_attn"]["k_proj"]["bias"]["param"]["bytes"] = list(model.decoder.model.decoder.layers[i].encoder_attn.k_proj.bias.data.numpy().tobytes())
    data["item"]["model"]["layers"][i]["encoder_attn"]["v_proj"]["weight"]["param"]["bytes"] = list(model.decoder.model.decoder.layers[i].encoder_attn.v_proj.weight.data.transpose(0, 1).numpy().tobytes())
    data["item"]["model"]["layers"][i]["encoder_attn"]["v_proj"]["bias"]["param"]["bytes"] = list(model.decoder.model.decoder.layers[i].encoder_attn.v_proj.bias.data.numpy().tobytes())
    data["item"]["model"]["layers"][i]["encoder_attn"]["out_proj"]["weight"]["param"]["bytes"] = list(model.decoder.model.decoder.layers[i].encoder_attn.out_proj.weight.data.transpose(0, 1).numpy().tobytes())
    data["item"]["model"]["layers"][i]["encoder_attn"]["out_proj"]["bias"]["param"]["bytes"] = list(model.decoder.model.decoder.layers[i].encoder_attn.out_proj.bias.data.numpy().tobytes())
    # encoder_attn_layer_norm
    data["item"]["model"]["layers"][i]["encoder_attn_layer_norm"]["gamma"]["param"]["bytes"] = list(model.decoder.model.decoder.layers[i].encoder_attn_layer_norm.weight.data.numpy().tobytes())
    data["item"]["model"]["layers"][i]["encoder_attn_layer_norm"]["beta"]["param"]["bytes"] = list(model.decoder.model.decoder.layers[i].encoder_attn_layer_norm.bias.data.numpy().tobytes())
    # fc1
    data["item"]["model"]["layers"][i]["fc1"]["weight"]["param"]["bytes"] = list(model.decoder.model.decoder.layers[i].fc1.weight.data.transpose(0, 1).numpy().tobytes())
    data["item"]["model"]["layers"][i]["fc1"]["bias"]["param"]["bytes"] = list(model.decoder.model.decoder.layers[i].fc1.bias.data.numpy().tobytes())
    # fc2
    data["item"]["model"]["layers"][i]["fc2"]["weight"]["param"]["bytes"] = list(model.decoder.model.decoder.layers[i].fc2.weight.data.transpose(0, 1).numpy().tobytes())
    data["item"]["model"]["layers"][i]["fc2"]["bias"]["param"]["bytes"] = list(model.decoder.model.decoder.layers[i].fc2.bias.data.numpy().tobytes())
    # final_layer_norm
    data["item"]["model"]["layers"][i]["final_layer_norm"]["gamma"]["param"]["bytes"] = list(model.decoder.model.decoder.layers[i].final_layer_norm.weight.data.numpy().tobytes())
    data["item"]["model"]["layers"][i]["final_layer_norm"]["beta"]["param"]["bytes"] = list(model.decoder.model.decoder.layers[i].final_layer_norm.bias.data.numpy().tobytes())

data["item"]["output_projection"]["weight"]["param"]["bytes"] = list(model.decoder.output_projection.weight.data.transpose(0, 1).numpy().tobytes())

with open("./decoder.json", 'w') as f:
    json.dump(data, f)
