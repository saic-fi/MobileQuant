import numpy as np
import torch, onnx, re
import os.path as osp
from copy import deepcopy


from aimet_torch.onnx_utils import OnnxSaver
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.qc_quantize_op import QcQuantizeWrapper


def to_device(x, device):
    return x.to(device) if isinstance(x, torch.Tensor) else tuple(to_device(y, device) for y in x)


def dump_onnx_and_encoding(sim_model, sim_sample, onnx_path, input_names):
    OnnxSaver.create_onnx_model_with_pytorch_layer_names(onnx_path, QuantizationSimModel.get_original_model(sim_model.model.cpu()).cpu(), to_device(sim_sample, 'cpu'), False, {}, {'opset_version': 9, 'input_names': input_names, 'output_names': ['output', 'k_out', 'v_out']})
    onnx_node_to_io_tensor_map, valid_param_set = OnnxSaver.get_onnx_node_to_io_tensor_names_map(onnx.load(onnx_path))
    QuantizationSimModel._export_encodings_to_files(sim_model.model, osp.dirname(onnx_path), osp.splitext(osp.basename(onnx_path))[0], onnx_node_to_io_tensor_map, valid_param_set, sim_model._excluded_layer_names, propagate_encodings=True, quantizer_args=sim_model.quant_args)


def update_qcfg_sim(sim_model, sixteen_bit_input_activations, sixteen_bit_output_activations, config, num_blocks=None, new_bitwidth=16):
    if num_blocks is None:
        num_blocks = config.n_layer
    for name, module in sim_model.model.named_modules():
        if isinstance(module, QcQuantizeWrapper):
            for i in range(len(module.input_quantizers)):
                module.input_quantizers[i].enabled = True
            if any(substring in name for substring in sixteen_bit_input_activations):
                for i in range(len(module.input_quantizers)):
                    module.input_quantizers[i].bitwidth = new_bitwidth
            if any(substring in name for substring in sixteen_bit_output_activations):
                for i in range(len(module.output_quantizers)):
                    module.output_quantizers[i].bitwidth = new_bitwidth
            if name.startswith('module_mul') and name != 'module_mul':
                ind = name.split('_')[-1]
                assert ind.isdigit()
                ind = int(ind)
                # if config.impl_sym_pch_as_slinear:
                #     if ind % 8 in [6, 7] or ind == num_blocks * 8 + 1:
                #         for i in range(len(module.input_quantizers)):
                #             module.input_quantizers[i].bitwidth = new_bitwidth
                #         for i in range(len(module.output_quantizers)):
                #             module.output_quantizers[i].bitwidth = new_bitwidth
                # else:
                #     if ind % 7 in [6] or ind == num_blocks * 7 + 1:
                #         for i in range(len(module.input_quantizers)):
                #             module.input_quantizers[i].bitwidth = new_bitwidth
                #         for i in range(len(module.output_quantizers)):
                #             module.output_quantizers[i].bitwidth = new_bitwidth
                if config.impl_sym_pch_as_slinear:
                    if ind % 8 in [7] or ind == num_blocks * 8 + 1:
                        for i in range(len(module.input_quantizers)):
                            module.input_quantizers[i].bitwidth = new_bitwidth
                        for i in range(len(module.output_quantizers)):
                            module.output_quantizers[i].bitwidth = new_bitwidth
            # if name == "module_matmul":
            #     module.output_quantizers[0].bitwidth = new_bitwidth
            if name.startswith('module_matmul') and name != 'module_matmul':
                ind = name.split('_')[-1]
                assert ind.isdigit()
                ind = int(ind)
                if ind % 2 == 1:
                    # softmax output
                    module.input_quantizers[0].bitwidth = new_bitwidth
                else:
                    module.output_quantizers[0].bitwidth = new_bitwidth

            if name.startswith('module_add') and name != 'module_add':
                ind = name.split('_')[-1]
                assert ind.isdigit()
                ind = int(ind)
                if ind % 5 in [2, 3, 4]:
                    # skip connection
                    for i in range(len(module.input_quantizers)):
                        module.input_quantizers[i].bitwidth = new_bitwidth
                    for i in range(len(module.output_quantizers)):
                        module.output_quantizers[i].bitwidth = new_bitwidth
            if name.startswith('norm.module_mul'):
                # layernorm before lm_head
                for i in range(len(module.output_quantizers)):
                    module.output_quantizers[i].bitwidth = new_bitwidth
                # for i in range(len(module.input_quantizers)):
                #     module.input_quantizers[i].enabled = False
                # for i in range(len(module.output_quantizers)):
                #     module.input_quantizers[i].enabled = False
            # if name.startswith('norm.module') or "lm_head" in name:
            #     for i in range(len(module.input_quantizers)):
            #         module.input_quantizers[i].enabled = False
            #     for i in range(len(module.output_quantizers)):
            #         module.input_quantizers[i].enabled = False
            if "lm_head" in name:
                for i in list(module.param_quantizers.keys()):
                    module.param_quantizers[i].bitwidth = 8

            
    return sim_model


def disable_quant_sim(sim_model):
    for name, module in sim_model.model.named_modules():
        if isinstance(module, QcQuantizeWrapper):
            for i in range(len(module.input_quantizers)):
                module.input_quantizers[i].enabled = False
            for i in range(len(module.output_quantizers)):
                module.output_quantizers[i].enabled = False
            for i in list(module.param_quantizers.keys()):
                module.param_quantizers[i].enabled = False
    return sim_model


def name_to_initializer(onnx_model):
    name_to_initializer = {}
    for ini in onnx_model.graph.initializer:
        name_to_initializer[ini.name] = ini
    return name_to_initializer


def name_to_node(onnx_model):
    name_to_node = {}
    for node in onnx_model.graph.node:
        name_to_node[node.name] = node
    return name_to_node


def output_to_node(onnx_model):
    output_to_node = {}
    for node in onnx_model.graph.node:
        for output in node.output:
            output_to_node[output] = node
    return output_to_node


def align_onnx_model(onnx_direct_path, onnx_aimet_path, output_path):
    # the key assumption here is that the outputs of the nodes in both onnx models are the same
    onnx_model_direct = onnx.load(onnx_direct_path)
    onnx_model_aimet = onnx.load(onnx_aimet_path)

    direct_name_to_initializer = name_to_initializer(onnx_model_direct)
    aimet_name_to_initializer = name_to_initializer(onnx_model_aimet)

    direct_output_to_node = output_to_node(onnx_model_direct)
    aimet_output_to_node = output_to_node(onnx_model_aimet)

    for i in range(len(onnx_model_direct.graph.node)):
        direct_node = onnx_model_direct.graph.node[i]
        output_name = direct_node.output[0]
        if output_name in aimet_output_to_node:
            aimet_node = aimet_output_to_node[output_name]
        elif 'reshape' in output_name and output_name.replace('Constant', 'Concat') in aimet_output_to_node:
            aimet_node = aimet_output_to_node[output_name.replace('Constant', 'Concat')]
        else:
            print('{} is not in aimet_onnx'.format(output_name))
        direct_node.name = aimet_node.name
        # for nn.Linear
        if direct_node.op_type == 'MatMul' and len(direct_node.input) == 2 and direct_node.input[1] in direct_name_to_initializer:
            assert(aimet_node.op_type == direct_node.op_type and all(n1 == n2 for n1, n2 in zip(aimet_node.output, direct_node.output)) and aimet_node.input[0] == direct_node.input[0])
            old_ini_name = direct_node.input[1]
            old_ini = direct_name_to_initializer[old_ini_name]
            old_ini.name = aimet_node.input[1]
            direct_node.input[1] = aimet_node.input[1]
    
    save_as_external_data = onnx_model_direct.ByteSize() >= onnx.checker.MAXIMUM_PROTOBUF
    onnx.save(onnx_model_direct, output_path, save_as_external_data=save_as_external_data)


def incorporate_l2norm(onnx_path, output_path, n_layer=28):
    #####################################################################
    # Incorporate L2 norm
    print('Incorporating LpNormalization...')
    onnx_model_origin = onnx.load(onnx_path)
    onnx_model_l2norm = deepcopy(onnx_model_origin)
    name_to_node_map = name_to_node(onnx_model_origin)

    OP_TYPES = ['Abs', 'Constant', 'Pow', 'ReduceSum', 'Clip', 'Shape', 'Expand', 'Div']
    # OP_TYPES = ['ReduceL2', 'Clip', 'Shape', 'Expand', 'Div']
    IN_TYPE, OUT_TYPE = 'Shape', 'Div'

    #####################################################################
    print('Collecting nodes related to L2 norm...')
    rms_nodes = {}
    for node_name in list(name_to_node_map.keys()):
        node = name_to_node_map[node_name]
        if 'module_normalize' in node_name and node.op_type in OP_TYPES:
            rms_nodes[node.name] = node

    def find_node(prefix, op_type):
        for k in list(rms_nodes.keys()):
            if prefix in k and rms_nodes[k].op_type == op_type:
                return rms_nodes[k]
        return None

    print('Creating nodes for L2 norm...')
    l2norm_nodes = []
    for i in range(n_layer * 2 + 1):
        prefix = 'module_normalize_{}'.format(i) if i > 0 else 'module_normalize'
        input_node, output_node = find_node(prefix, IN_TYPE), find_node(prefix, OUT_TYPE)
        if input_node is None or output_node is None:
            continue
        new_node_name = prefix + '_l2norm'
        node = onnx.helper.make_node(name=new_node_name, op_type='LpNormalization', inputs=input_node.input, outputs=output_node.output, axis=-1, p=2)
        l2norm_nodes.append(node)

    # delete the old nodes
    print('Delete old nodes...')
    for i in range(len(onnx_model_l2norm.graph.node)-1, -1, -1):
        node = onnx_model_l2norm.graph.node[i]
        onnx_model_l2norm.graph.node.remove(node)
    
    # add new nodes in topological order
    print('Add new nodes in topological order...')
    for i in range(len(onnx_model_origin.graph.node)):
        node = onnx_model_origin.graph.node[i]
        if node.name not in rms_nodes:
            onnx_model_l2norm.graph.node.append(deepcopy(node))
        elif node.op_type == OUT_TYPE:
            onnx_model_l2norm.graph.node.append(l2norm_nodes.pop(0))
    
    print('Exporting the l2norm onnx model...')
    save_as_external_data = onnx_model_l2norm.ByteSize() >= onnx.checker.MAXIMUM_PROTOBUF
    onnx.save(onnx_model_l2norm, output_path, save_as_external_data=save_as_external_data)
    return onnx_model_l2norm


qnn_data_type_to_np = {
    0x0008: np.int8,
    0x0016: np.int16,
    0x0032: np.int32,
    0x0064: np.int64,

    0x0108: np.uint8,
    0x0116: np.uint16,
    0x0132: np.uint32,
    0x0164: np.uint64,

    0x0216: np.float16,
    0x0232: np.float32,

    0x0308: np.int8,
    0x0316: np.int16,
    0x0332: np.int32,

    0x0408: np.uint8,
    0x0416: np.uint16,
    0x0432: np.uint32,

    0x0508: np.bool_
}


# qnn_data_type_to_np = {
#     8: np.int8,
#     22: np.int16,
#     50: np.int32,
#     100: np.int64,

#     264: np.uint8,
#     278: np.uint16,
#     306: np.uint32,
#     356: np.uint64,

#     534: np.float16,
#     562: np.float32,

#     776: np.int8,
#     790: np.int16,
#     818: np.int32,

#     1032: np.uint8,
#     1046: np.uint16,
#     1074: np.uint32,

#     1288: np.bool_
# }



def update_encodings_from_min_max(fmin, fmax, encoding, field):
    bitwidth = int(encoding[field]["bitwidth"])
    qmax = 2 ** bitwidth - 1
    scale = (fmax-fmin)/qmax
    offset = int((fmin*qmax)/(fmax-fmin))
    encoding[field]["max"], encoding[field]["min"], encoding[field]["scale"], encoding[field]["offset"] = fmax, fmin, scale, offset
    return encoding


def prefix_match_linear(dictionary, prefix):
    matches = []
    for key in dictionary.keys():
        if key.startswith(prefix):
            matches.append(key)
    assert(len(matches) == 1)
    return matches


def override_encoding(src_dict, tgt_dict, src_name, tgt_name, src_field, tgt_field, tgt_subfield, factor=1.0):
    assert(src_name in src_dict)
    assert(tgt_name in tgt_dict)
    fmin, fmax = src_dict[src_name][src_field]
    encoding = tgt_dict[tgt_name][tgt_field]
    fmin, fmax = fmin * factor, fmax * factor
    tgt_dict[tgt_name][tgt_field] = update_encodings_from_min_max(fmin, fmax, encoding, tgt_subfield)
    return tgt_dict


def update_encodings(ori_encodings, new_act_dict, num_blocks, q_proj_factor, config):
    ori_act_dict = ori_encodings["activation_encodings"]

    all_keys = set(ori_act_dict.keys())

    print("num of nodes (before)", len(all_keys))

    for i in range(num_blocks):
        # input norm
        tgt_name = prefix_match_linear(ori_act_dict, f"layers.{i}.input_layernorm.module_normalize")[0]
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.input_layernorm", tgt_name, "input", "input", "0")
        all_keys.discard(tgt_name)

        tgt_name = prefix_match_linear(ori_act_dict, f"layers.{i}.input_layernorm.module_mul")[0]
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.input_layernorm", tgt_name, "output", "output", "0")
        all_keys.discard(tgt_name)

        # post norm
        tgt_name = prefix_match_linear(ori_act_dict, f"layers.{i}.post_attention_layernorm.module_normalize")[0]
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.post_attention_layernorm", tgt_name, "input", "input", "0")
        all_keys.discard(tgt_name)

        tgt_name = prefix_match_linear(ori_act_dict, f"layers.{i}.post_attention_layernorm.module_mul")[0]
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.post_attention_layernorm", tgt_name, "output", "output", "0")
        all_keys.discard(tgt_name)


        # attn proj
        # q_proj
        tgt_name = prefix_match_linear(ori_act_dict, f"layers.{i}.self_attn.q_proj")[0]
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.input_layernorm", tgt_name, "output", "input", "0")
        tgt_name = prefix_match_linear(ori_act_dict, f"layers.{i}.self_attn.q_proj")[0]
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.q_proj", tgt_name, "output", "output", "0", q_proj_factor)
        all_keys.discard(tgt_name)

        # k_proj
        tgt_name = prefix_match_linear(ori_act_dict, f"layers.{i}.self_attn.k_proj")[0]
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.input_layernorm", tgt_name, "output", "input", "0")
        tgt_name = prefix_match_linear(ori_act_dict, f"layers.{i}.self_attn.k_proj")[0]
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.k_proj", tgt_name, "output", "output", "0")
        all_keys.discard(tgt_name)

        # v_proj
        tgt_name = prefix_match_linear(ori_act_dict, f"layers.{i}.self_attn.v_proj")[0]
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.input_layernorm", tgt_name, "output", "input", "0")
        tgt_name = prefix_match_linear(ori_act_dict, f"layers.{i}.self_attn.v_proj")[0]
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.v_proj", tgt_name, "output", "output", "0")
        all_keys.discard(tgt_name)

        # o_proj
        tgt_name = prefix_match_linear(ori_act_dict, f"layers.{i}.self_attn.o_proj")[0]
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.o_proj", tgt_name, "output", "output", "0")
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.pv_bmm", tgt_name, "output", "input", "0")
        all_keys.discard(tgt_name)
        
        # mlp
        tgt_name = prefix_match_linear(ori_act_dict, f"layers.{i}.mlp.w1")[0]
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.post_attention_layernorm", tgt_name, "output", "input", "0")
        tgt_name = prefix_match_linear(ori_act_dict, f"layers.{i}.mlp.w1")[0]
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.mlp.w1", tgt_name, "output", "output", "0")
        all_keys.discard(tgt_name)

        tgt_name = prefix_match_linear(ori_act_dict, f"layers.{i}.mlp.w3")[0]
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.post_attention_layernorm", tgt_name, "output", "input", "0")
        tgt_name = prefix_match_linear(ori_act_dict, f"layers.{i}.mlp.w3")[0]
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.mlp.w3", tgt_name, "output", "output", "0")
        all_keys.discard(tgt_name)


        # act
        tgt_name = prefix_match_linear(ori_act_dict, f"layers.{i}.mlp.act.sigmoid")
        if len(tgt_name) > 0:
            tgt_name = tgt_name[0]
            ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.mlp.w1", tgt_name, "output", "input", "0")
            ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.mlp.act_fn", tgt_name, "input2", "output", "0")
            all_keys.discard(tgt_name)

            tgt_name = prefix_match_linear(ori_act_dict, f"layers.{i}.mlp.act.mul")[0]
            ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.mlp.w1", tgt_name, "output", "input", "0")
            ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.mlp.act_fn", tgt_name, "input2", "input", "1")
            ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.mlp.act_fn", tgt_name, "output", "output", "0")
            # ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.mlp.w2", tgt_name, "input", "output", "0")
            all_keys.discard(tgt_name)
            
        if config.impl_sym_pch_as_slinear:
            tgt_name = prefix_match_linear(ori_act_dict, f"layers.{i}.mlp.w2.linear")[0]
            ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.mlp.w2", tgt_name, "input", "input", "0")
        else:
            tgt_name = prefix_match_linear(ori_act_dict, f"layers.{i}.mlp.w2")[0]
            ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.mlp.w2", tgt_name, "input", "input", "0")
            ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.mlp.w2", tgt_name, "output", "output", "0")
        all_keys.discard(tgt_name)


        ######################################################################################################################################
        # qk_bmm and pv_bmm

        tgt_name = f"module_matmul" if i == 0 else f"module_matmul_{2*i}"
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.qk_bmm", tgt_name, "input", "input", "0", q_proj_factor)
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.qk_bmm", tgt_name, "input2", "input", "1")
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.qk_bmm", tgt_name, "output", "output", "0", q_proj_factor)
        all_keys.discard(tgt_name)

        tgt_name = f"module_matmul_{2*i+1}"
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.pv_bmm", tgt_name, "input", "input", "0")
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.pv_bmm", tgt_name, "input2", "input", "1")
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.pv_bmm", tgt_name, "output", "output", "0")
        all_keys.discard(tgt_name)

        #######################################################################################################################################
        # softmax
        tgt_name = prefix_match_linear(ori_act_dict, f"layers.{i}.self_attn.softmax")[0]
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.pv_bmm", tgt_name, "input", "output", "0")
        all_keys.discard(tgt_name)

        #######################################################################################################################################
        # add
        # three additions are not overrided: add_mask, and two in ROPE

        tgt_name = f"module_add_{5 * i + 3}"
        # ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.input_layernorm", tgt_name, "input", "input", "0")
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.o_proj", tgt_name, "output", "input", "1")
        # ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.post_attention_layernorm", tgt_name, "input", "output", "0")
        all_keys.discard(tgt_name)

        tgt_name = f"module_add_{5 * i + 4}"
        # ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.post_attention_layernorm", tgt_name, "input", "input", "0")
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.mlp.w2", tgt_name, "output", "input", "1")
        all_keys.discard(tgt_name)
        # if i > 0:
        #     tgt_name = f"module_add_{5 * (i - 1) + 4}"
        #     ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.input_layernorm", tgt_name, "input", "output", "0")

        #######################################################################################################################################
        # reshape
        tgt_name = f"layers.{i}.self_attn.module_reshape" if i == 0 else f"layers.{i}.self_attn.module_reshape_{6*i}"
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.q_proj", tgt_name, "output", "input", "0", q_proj_factor)
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.q_proj", tgt_name, "output", "output", "0", q_proj_factor)
        all_keys.discard(tgt_name)

        tgt_name = f"layers.{i}.self_attn.module_reshape_{6*i+1}"
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.k_proj", tgt_name, "output", "input", "0")
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.k_proj", tgt_name, "output", "output", "0")
        all_keys.discard(tgt_name)

        tgt_name = f"layers.{i}.self_attn.module_reshape_{6*i+2}"
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.v_proj", tgt_name, "output", "input", "0")
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.v_proj", tgt_name, "output", "output", "0")
        all_keys.discard(tgt_name)

        tgt_name = f"layers.{i}.self_attn.module_reshape_{6*i+3}"
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.k_proj", tgt_name, "output", "input", "0")
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.k_proj", tgt_name, "output", "output", "0")
        all_keys.discard(tgt_name)

        tgt_name = f"layers.{i}.self_attn.module_reshape_{6*i+4}"
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.v_proj", tgt_name, "output", "input", "0")
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.v_proj", tgt_name, "output", "output", "0")
        all_keys.discard(tgt_name)

        tgt_name = f"layers.{i}.self_attn.module_reshape_{6*i+5}"
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.pv_bmm", tgt_name, "output", "input", "0")
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.pv_bmm", tgt_name, "output", "output", "0")
        all_keys.discard(tgt_name)
        #######################################################################################################################################
        # transpose
        tgt_name = f"layers.{i}.self_attn.module_transpose" if i == 0 else f"layers.{i}.self_attn.module_transpose_{5*i}"
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.q_proj", tgt_name, "output", "input", "0", q_proj_factor)
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.q_proj", tgt_name, "output", "output", "0", q_proj_factor)
        all_keys.discard(tgt_name)

        tgt_name = f"layers.{i}.self_attn.module_transpose_{5*i+1}"
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.k_proj", tgt_name, "output", "input", "0")
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.k_proj", tgt_name, "output", "output", "0")
        all_keys.discard(tgt_name)

        tgt_name = f"layers.{i}.self_attn.module_transpose_{5*i+2}"
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.v_proj", tgt_name, "output", "input", "0")
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.v_proj", tgt_name, "output", "output", "0")
        all_keys.discard(tgt_name)

        tgt_name = f"layers.{i}.self_attn.module_transpose_{5*i+3}"
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.k_proj", tgt_name, "output", "input", "0")
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.k_proj", tgt_name, "output", "output", "0")
        all_keys.discard(tgt_name)

        tgt_name = f"layers.{i}.self_attn.module_transpose_{5*i+4}"
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.pv_bmm", tgt_name, "output", "input", "0")
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.pv_bmm", tgt_name, "output", "output", "0")
        all_keys.discard(tgt_name)

        #######################################################################################################################################
        # concat
        tgt_name = "module_cat" if i == 0 else f"module_cat_{2*i}"
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.q_proj", tgt_name, "output", "input", "0", q_proj_factor)
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.q_proj", tgt_name, "output", "input", "1", q_proj_factor)
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.q_proj", tgt_name, "output", "output", "0", q_proj_factor)
        all_keys.discard(tgt_name)

        tgt_name = f"module_cat_{2*i+1}"
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.k_proj", tgt_name, "output", "input", "0")
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.k_proj", tgt_name, "output", "input", "1")
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.k_proj", tgt_name, "output", "output", "0")
        all_keys.discard(tgt_name)

        #######################################################################################################################################
        # mul
        extra_mul = 1 if config.impl_sym_pch_as_slinear else 0

        tgt_name = f"module_mul_{(7+extra_mul)*i+1}"
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.q_proj", tgt_name, "output", "input", "0", q_proj_factor)
        all_keys.discard(tgt_name)
        
        tgt_name = f"module_mul_{(7+extra_mul)*i+2}"
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.q_proj", tgt_name, "output", "input", "0", q_proj_factor)
        all_keys.discard(tgt_name)

        tgt_name = f"module_mul_{(7+extra_mul)*i+3}"
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.k_proj", tgt_name, "output", "input", "0")
        all_keys.discard(tgt_name)
        
        tgt_name = f"module_mul_{(7+extra_mul)*i+4}"
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.self_attn.k_proj", tgt_name, "output", "input", "0")
        all_keys.discard(tgt_name)

        tgt_name = f"module_mul_{(7+extra_mul)*i+6}"
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.mlp.act_fn", tgt_name, "output", "input", "0")
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.mlp.w3", tgt_name, "output", "input", "1")
        ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.mlp.w2", tgt_name, "input", "output", "0")
        all_keys.discard(tgt_name)

        if config.impl_sym_pch_as_slinear:
            tgt_name = f"module_mul_{(7+extra_mul)*i+7}"
            ori_act_dict = override_encoding(new_act_dict, ori_act_dict, f"model.layers.{i}.mlp.w2", tgt_name, "output", "output", "0")
            all_keys.discard(tgt_name)


        
       
    print("num of nodes (after)", len(all_keys))
    for x in all_keys:
        print(f"not overriding {x}")

    ori_encodings["activation_encodings"] = ori_act_dict
    return ori_encodings


_qnn_parentheses = re.compile(r'\(.*?\)')


def norm_unit(value, unit):
    ''' Convert to milliseconds
    '''
    if unit == 'ms':
        return value
    elif unit == 's':
        return value * 1000
    elif unit == 'us':
        return value / 1000
    elif unit == 'ns':
        return value / 1000000
    elif unit in ['cycles', 'count', 'inf/sec']:
        return value
    elif unit == 'k':
        return value * 1e3
    elif unit == 'M':
        return value * 1e6
    elif unit == 'G':
        return value * 1e9
    elif unit == 'T':
        return value * 1e12
    elif not unit:
        return value
    else:
        raise ValueError(f'Unknown time unit: {unit!r}')


def parse_profile_viewer(out):
    section = None
    layers_timing = {}
    parsed = {}
    for line in out.splitlines():
        line = line.rstrip()
        if not line:
            continue
        if line.startswith('        '):
            # Layer info
            if section != 'Execute':
                print(f'Found 2nd indentation level in section other than Execute ({section}), ignoring....')
                continue
            line = line[8:]
            name, timing = tuple(l.strip() for l in line.rsplit(':', maxsplit=1))
            num, unit = tuple(l.strip() for l in timing.split(' ') if l)
            timing = norm_unit(int(num), unit)
            name = _qnn_parentheses.sub('', name)
            if ':' in name:
                name = name.split(':')[0]
            if ' ' in name:
                name = name.split(' ')[0]
            layers_timing[name] = timing
        elif line.startswith('    '):
            line = line[4:]
            name, timing = tuple(l.strip() for l in line.rsplit(':', maxsplit=1))
            num, unit = tuple(l.strip() for l in timing.split(' ') if l)
            timing = norm_unit(float(num), unit)

            if name == 'NetRun':
                sub = 'Total'
            else:
                if name.startswith('Backend ('):
                    sub = name[name.find('(')+1:name.rfind(')')]
                else:
                    sub = name

            sub = _qnn_parentheses.sub('', sub)
            parsed.setdefault(section, {})[sub] = timing
        else:
            if line == 'Init Stats:':
                section = 'Init'
            elif line == 'Compose Graphs Stats:':
                section = 'Build Graph'
            elif line == 'Finalize Stats:':
                section = 'Finalize'
            elif line == 'De-Init Stats:':
                section = 'De-Init'
            elif line == 'Total Inference Time:':
                section = 'Execute'

    parsed['Layer Times'] = layers_timing
    return parsed