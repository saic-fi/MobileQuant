import numpy as np
import os.path as osp
from tqdm import tqdm
from pathlib import Path
from glob import glob
import onnx, onnxruntime
from copy import deepcopy
from functools import partial
from datasets import load_dataset
import torch, os, argparse, logging, gc, subprocess, shutil
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, PreTrainedModel


from mobilellm.model.sim_model import SimConfig, SimModel, create_conv_model, Sim_Head, Sim_QNN

from mobilellm.utils.bench import print_model_size
from mobilellm.utils.io import json_load, json_save
from device.utils import incorporate_l2norm, update_qcfg_sim, to_device, dump_onnx_and_encoding, qnn_data_type_to_np, parse_profile_viewer

from aimet_torch.model_preparer import prepare_model
from aimet_torch.quantsim import QuantizationSimModel, load_encodings_to_sim
from aimet_common.defs import QuantScheme, QuantizationDataType
from aimet_common.utils import AimetLogger
AimetLogger.set_level_for_all_areas(logging.ERROR)


from mobilellm.model.hf_config import HFConfig
from mobilellm.model.hf_model import HFForCausalLM
AutoConfig.register("hfmodel", HFConfig)
AutoModelForCausalLM.register(HFConfig, HFForCausalLM)


parser = argparse.ArgumentParser()

parser.add_argument('--hf_path', type=str, default=None, help='path of the hf model')
parser.add_argument('--model_name', type=str, default=None)
parser.add_argument('--max_length', type=int, default=1024, help='max seq len for the samples')
parser.add_argument('--device_path',    type=str, default='/data/local/tmp/mobilequant')
parser.add_argument('--num_blocks',     type=int, default=None)
parser.add_argument("--output_dir",     type=str, default='results/sim_{}_qnn')
parser.add_argument('--default_config', type=str, default='assets/aimet_config.json', help='the default config file')
parser.add_argument('--quant_encoding', type=str, default=None, help='the encoding config file')
parser.add_argument('--kv_encoding',    type=str, default=None, help='the encoding for the kv cache')
parser.add_argument('--kv_cache',       default=False, action="store_true")
parser.add_argument('--detailed',       default=False, action="store_true")
parser.add_argument("--quant_config",   nargs='+', default=[8,8,32], type=int)
parser.add_argument('--use_conv', default=False, action="store_true")
parser.add_argument('--per_channel', default=False, action="store_true")
parser.add_argument('--outlier_act_bitwidth', type=int, default=16)


args = parser.parse_args()
assert(args.hf_path is not None)
if args.hf_path.endswith('/'):
    args.hf_path = args.hf_path[:-1]
args.model_name = osp.basename(args.hf_path)
args.model_path = osp.join(args.hf_path, f"sim_{args.model_name}.pth")
args.output_dir = args.output_dir.format(args.model_name)


if args.per_channel:
    args.default_config = "assets/aimet_per_channel_config.json"


if args.output_dir.endswith('/'):
    args.output_dir = args.output_dir[:-1]


if args.kv_cache:
    args.output_dir += '_with_kv'
else:
    args.output_dir += '_no_kv'


assert args.quant_encoding is not None
if args.kv_cache:
    assert args.kv_encoding is not None


seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn


def main():
    #####################################################################
    # sim model
    config = SimConfig.from_name(args.model_name)
    config.block_size = args.max_length
    model_ori = SimModel(config)
    for x in model_ori.parameters(): 
        x.requires_grad = False
    ckpt = torch.load(args.model_path, map_location='cpu')
    msg = model_ori.load_state_dict(ckpt, strict=True)
    print(msg)
    if args.use_conv:
        model_ori = create_conv_model(model_ori)

    #####################################################################
    # sub modules
    if args.num_blocks is None:
        args.num_blocks = config.n_layer
    model_head = Sim_Head.from_sim(model_ori).cuda()
    model_body = Sim_QNN.from_sim(model_ori, args.num_blocks).cuda()
    device = next(model_body.parameters()).device
    model_head.eval()
    model_body.eval()
    del model_ori
    gc.collect()
    torch.cuda.empty_cache()

    #####################################################################
    rand_ids        = torch.randint(3, config.vocab_size, size=(args.max_length,), dtype=torch.int32)
    position_ids    = torch.arange(0, args.max_length, dtype=torch.long)
    attention_mask  = SimModel._make_causal_mask(args.max_length, args.max_length, args.max_length)
    attention_mask  = config.neg_inf * attention_mask
    rand_sample     = (rand_ids, attention_mask, position_ids)

    with torch.no_grad():
        gpu_sample = model_head(*to_device(rand_sample, device))

    qnn_inp = ' '.join([
        'input_feats:='     + osp.join(args.device_path, "0-input.raw"),
        'attention_mask:='  + osp.join(args.device_path, "1-input.raw"),
        'cos:='             + osp.join(args.device_path, "2-input.raw"),
        'sin:='             + osp.join(args.device_path, "3-input.raw")
    ])

    cpu_inp = ' '.join([
        'input_feats:='     + osp.abspath(osp.join(args.output_dir, "0-input.raw")),
        'attention_mask:='  + osp.abspath(osp.join(args.output_dir, "1-input.raw")),
        'cos:='             + osp.abspath(osp.join(args.output_dir, "2-input.raw")),
        'sin:='             + osp.abspath(osp.join(args.output_dir, "3-input.raw")),
    ])

    os.makedirs(args.output_dir, exist_ok=True)

    if args.kv_cache:
        kv_enc = json_load(args.kv_encoding)
        k_min, k_max, v_min, v_max = kv_enc['k_cache']['min'], kv_enc['k_cache']['max'], kv_enc['v_cache']['min'], kv_enc['v_cache']['max']
        k_cache = torch.rand(args.num_blocks, config.n_kv_head, config.block_size-1, config.head_dim, dtype=torch.float32).to(device)
        v_cache = torch.rand(args.num_blocks, config.n_kv_head, config.block_size-1, config.head_dim, dtype=torch.float32).to(device)
        k_cache, v_cache = k_cache * (k_max-k_min) + k_min, v_cache * (v_max-v_min) + v_min
        k_cache[:, :, :, 0], k_cache[:, :, :, 1], v_cache[:, :, :, 0], v_cache[:, :, :, 1] = k_min, k_max, v_min, v_max
        gpu_sample = (gpu_sample[0][0].unsqueeze(0), gpu_sample[1][:, 0].unsqueeze(1), gpu_sample[2][0].unsqueeze(0), gpu_sample[3][0].unsqueeze(0))
        gpu_sample = (*gpu_sample, k_cache, v_cache)

        gpu_sample[4].cpu().data.numpy().astype(np.float32).tofile(osp.join(args.output_dir, "4-input.raw"))
        gpu_sample[5].cpu().data.numpy().astype(np.float32).tofile(osp.join(args.output_dir, "5-input.raw"))

        qnn_inp = ' '.join([
            'input_feats:='     + osp.join(args.device_path, "0-input.raw"),
            'attention_mask:='  + osp.join(args.device_path, "1-input.raw"),
            'cos:='             + osp.join(args.device_path, "2-input.raw"),
            'sin:='             + osp.join(args.device_path, "3-input.raw"),
            'k_cache:='         + osp.join(args.device_path, "4-input.raw"),
            'v_cache:='         + osp.join(args.device_path, "5-input.raw"),
        ])

        cpu_inp = ' '.join([
            'input_feats:='     + osp.abspath(osp.join(args.output_dir, "0-input.raw")),
            'attention_mask:='  + osp.abspath(osp.join(args.output_dir, "1-input.raw")),
            'cos:='             + osp.abspath(osp.join(args.output_dir, "2-input.raw")),
            'sin:='             + osp.abspath(osp.join(args.output_dir, "3-input.raw")),
            'k_cache:='         + osp.abspath(osp.join(args.output_dir, "4-input.raw")),
            'v_cache:='         + osp.abspath(osp.join(args.output_dir, "5-input.raw")),
        ])

    gpu_sample[0].cpu().data.numpy().astype(np.float32).tofile(osp.join(args.output_dir, "0-input.raw"))
    gpu_sample[1].cpu().data.numpy().astype(np.float32).tofile(osp.join(args.output_dir, "1-input.raw"))
    gpu_sample[2].cpu().data.numpy().astype(np.float32).tofile(osp.join(args.output_dir, "2-input.raw"))
    gpu_sample[3].cpu().data.numpy().astype(np.float32).tofile(osp.join(args.output_dir, "3-input.raw"))
    
    with open(osp.join(args.output_dir, "htp_file.txt"), 'w') as fid:
        fid.write(qnn_inp)
    with open(osp.join(args.output_dir, "cpu_file.txt"), 'w') as fid:
        fid.write(cpu_inp)

    #####################################################################
    # prepare model
    if args.kv_cache:
        model_fp = prepare_model(model_body)
    else:
        model_fp = prepare_model(model_body, concrete_args={'k_cache': None, 'v_cache': None})
    #####################################################################

    del model_body, model_head
    gc.collect()
    torch.cuda.empty_cache()

    #####################################################################
    ## Sim model
    with torch.no_grad():
        fp_outputs = model_fp(*gpu_sample)[0]
        # print(fp_outputs.shape)
    
    sim = QuantizationSimModel(
        model=model_fp, 
        quant_scheme=QuantScheme.post_training_tf, 
        dummy_input=gpu_sample,
        rounding_mode='nearest',
        in_place=True,
        config_file=args.default_config,
        default_output_bw=args.quant_config[1],
        default_param_bw=args.quant_config[0],
        default_data_type=QuantizationDataType.int,
    )

    #####################################################################
    ## 16-bit
    sixteen_bit_output_activations = ['module_normalize', 'o_proj', 'w2', 'lm_head', 'softmax']
    sixteen_bit_input_activations = ['module_normalize', 'norm.module_mul', 'w2', 'lm_head', 'softmax']
    sim = update_qcfg_sim(sim, sixteen_bit_input_activations, sixteen_bit_output_activations, config, args.num_blocks, args.outlier_act_bitwidth)
    
    #####################################################################
    ## Load or compute the encodings
    load_encodings_to_sim(sim, args.quant_encoding)
    #####################################################################
    with torch.no_grad():
        sim_outputs, sim_k, sim_v = sim.model(*gpu_sample)
        # print('shape', sim_outputs.shape, sim_k.shape, sim_v.shape)
    #################################################################
    aimet_dir = osp.join(args.output_dir, 'aimet')
    os.makedirs(aimet_dir, exist_ok=True)
    onnx_aimet_path  = osp.join(aimet_dir, 'model_aimet.onnx')

    l2norm_dir = osp.join(args.output_dir, 'l2norm')
    os.makedirs(l2norm_dir, exist_ok=True)
    onnx_l2norm_path = osp.join(l2norm_dir, 'model_l2norm.onnx') 
    ###########################################################################################
    input_names = ['input_feats', 'attention_mask', 'cos', 'sin']
    if args.kv_cache:
        input_names.extend(['k_cache', 'v_cache'])
    print('Exporting the model to an intermediate ONNX model using aimet...')
    dump_onnx_and_encoding(sim, gpu_sample, onnx_aimet_path, input_names=input_names)
    #####################################################################
    # Incorporate L2 norm
    incorporate_l2norm(onnx_aimet_path, onnx_l2norm_path, args.num_blocks)
    # Amend the encodings (hard coding)
    print('Exporting the l2norm encodings...')
    encodings = json_load(onnx_aimet_path.replace('.onnx', '.encodings'))
    activation_encodings = encodings['activation_encodings']

    for k in list(activation_encodings.keys()):
        if 'module_normalize' in k and 'Div' not in k and 'Mul' not in k:
            activation_encodings.pop(k)

    encodings['activation_encodings'] = activation_encodings
    #####################################################################
    # add the encoding for Slice, Unsqueeze, Cache, rm the encoding for Gather
    if args.kv_cache:
        qmax = 2 ** args.quant_config[1] - 1
        k_scale, v_scale = (k_max-k_min)/qmax, (v_max-v_min)/qmax
        k_enc = [{"bitwidth": args.quant_config[1], "dtype": "int", "is_symmetric": "False", "max": k_max, "min": k_min, "offset": int(k_min/k_scale), "scale": k_scale}]
        v_enc = [{"bitwidth": args.quant_config[1], "dtype": "int", "is_symmetric": "False", "max": v_max, "min": v_min, "offset": int(v_min/v_scale), "scale": v_scale}]
        activation_encodings['k_cache'], activation_encodings['v_cache'], activation_encodings['k_out'], activation_encodings['v_out'] = k_enc, v_enc, k_enc, v_enc
    onnx_model = onnx.load(onnx_l2norm_path)
    for i in range(len(onnx_model.graph.node)):
        node = onnx_model.graph.node[i]
        # if node.op_type in ['Slice', 'Unsqueeze', 'Squeeze']:
        if node.op_type in ['Slice', 'Squeeze']:
            assert(len(node.input) == 1 and len(node.output) == 1)
            # print('input', node.input[0])
            # print('output', node.output[0])
            assert(node.input[0] in activation_encodings)
            if node.output[0] not in activation_encodings:
                activation_encodings[node.output[0]] = deepcopy(activation_encodings[node.input[0]])
        elif node.op_type in ['Gather']:
            if node.output[0] in activation_encodings:
                # This is important: overriding the encoding of the Gather op leads to an error in the QNN context generation
                activation_encodings.pop(node.output[0])
    encodings["activation_encodings"] = activation_encodings
    json_save(onnx_l2norm_path.replace('.onnx', '.encodings'), encodings)
    del onnx_model
    gc.collect()
    torch.cuda.empty_cache()
    
    #####################################################################
    # On-device

    del sim, model_fp
    gc.collect()
    torch.cuda.empty_cache()

    env = os.environ.copy()
    qnn_dir = osp.abspath(osp.join(args.output_dir, 'device'))
    os.makedirs(qnn_dir, exist_ok=True)
    qnn_sdk_root = env["QNN_SDK_ROOT"]
    qnn_bin_dir = osp.join(qnn_sdk_root, "bin", "x86_64-linux-clang")
    qnn_lib_dir = osp.join(qnn_sdk_root, "lib", "x86_64-linux-clang")
    qnn_py_dir = osp.join(qnn_sdk_root, "lib", "python")


    def prepend(value, key):
        current = env.get(key)
        if current:
            value = f'{value}:{current}'
        env[key] = value

    prepend(qnn_bin_dir, 'PATH')
    prepend(qnn_py_dir, 'PYTHONPATH')
    prepend(qnn_lib_dir, 'LD_LIBRARY_PATH')

    #################################################################################################
    # convert
    qnn_command = [
        "qnn-onnx-converter", 
        "--input_network", "{}".format(onnx_l2norm_path), 
        "--output_path", osp.join(qnn_dir, "model.cpp"),
        "--input_list", osp.join(args.output_dir, "cpu_file.txt"),
        "--weight_bw", str(args.quant_config[0]), 
        "--act_bw", str(args.quant_config[1]), 
        "--bias_bw", str(args.quant_config[2]), 
        "--float_bw", "32",
        '--quantization_overrides', onnx_l2norm_path.replace('.onnx', '.encodings'), 
        "--input_layout", 'attention_mask', 'NFC',   
    ]
    if args.per_channel:
        if args.use_conv:
            qnn_command.extend(["--use_per_channel_quantization", "--param_quantizer_schema", "symmetric"])
        else:
            qnn_command.extend(["--use_per_row_quantization", "--param_quantizer_schema", "symmetric"])
    if args.kv_cache:
        qnn_command.extend(["--input_layout", 'k_cache', 'NHWC', "--input_layout", 'v_cache', 'NHWC'])

    result = subprocess.run(qnn_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=None)
    # print(result.stdout)
    # print(result.stderr)

    #################################################################################################
    # generate library
    qnn_command = [
        "qnn-model-lib-generator", 
        "-c", osp.join(qnn_dir, "model.cpp"),
        "-t", "aarch64-android", "x86_64-linux-clang",
        "-l", "qnn_model",
        "-o", qnn_dir, 
        "-b", osp.join(qnn_dir, "model.bin"),
    ]

    result = subprocess.run(qnn_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=None)
    # print(result.stdout)
    # print(result.stderr)

    #################################################################################################
    # generate context
    shutil.copy(str(Path(__file__).parent.parent.joinpath('assets', 'sm8650_htp_basic_config.json')), qnn_dir)
    shutil.copy(str(Path(__file__).parent.parent.joinpath('assets', 'sm8650_htp_ext_config.json')), qnn_dir)

    qnn_command = [
        "qnn-context-binary-generator", 
        "--model", osp.join(qnn_dir, "x86_64-linux-clang", "libqnn_model.so"),
        "--backend", osp.join(qnn_sdk_root, "lib/x86_64-linux-clang/libQnnHtp.so") ,
        "--binary_file", "qnn_model",
        "--output_dir", qnn_dir, 
        "--config_file", "sm8650_htp_basic_config.json",
    ]
    result = subprocess.run(qnn_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=qnn_dir)
    print(result.stdout)
    print(result.stderr)
    #################################################################################################

    #################################################################################################
    # push files to device
    print('Pushing input files...')
    msg = subprocess.run(['adb', 'shell', 'mkdir', '-p', args.device_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(msg.stdout.decode('utf-8'))
    for x in glob(os.path.join(args.output_dir, '*.raw')):
        msg = subprocess.run(['adb', 'push', x, args.device_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(msg.stdout.decode('utf-8'))
    msg = subprocess.run(['adb', 'push', osp.join(args.output_dir, 'htp_file.txt'), args.device_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print('Pushing config files...')
    msg = subprocess.run(['adb', 'push', osp.join(qnn_dir, 'sm8650_htp_basic_config.json'), args.device_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    msg = subprocess.run(['adb', 'push', osp.join(qnn_dir, 'sm8650_htp_ext_config.json'), args.device_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(msg.stdout.decode('utf-8'))
    print('Pushing lib files...')
    for x in ["libQnnHtp.so", "libQnnHtpNetRunExtensions.so", "libQnnHtpPrepare.so", "libQnnHtpV75Stub.so"]:
        msg = subprocess.run(['adb', 'push', osp.join(qnn_sdk_root, "lib", "aarch64-android", x), args.device_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(msg.stdout.decode('utf-8'))
    msg = subprocess.run(['adb', 'push', osp.join(qnn_sdk_root, "lib", "hexagon-v75", "unsigned", "libQnnHtpV75Skel.so"), args.device_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(msg.stdout.decode('utf-8'))
    print('Pushing exec file...')
    msg = subprocess.run(['adb', 'push', osp.join(qnn_sdk_root, "bin", "aarch64-android", "qnn-net-run"), args.device_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    msg = subprocess.run(['adb', 'push', osp.join(qnn_dir, "qnn_model.bin"), args.device_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(msg.stdout.decode('utf-8'))

    #################################################################################################
    # on-device profiling
    qnn_command = [
        "adb", "shell", "cd", args.device_path, "&&", 
        f"LD_LIBRARY_PATH={args.device_path}", 
        f"ADSP_LIBRARY_PATH={args.device_path}", 
        "./qnn-net-run", 
        "--retrieve_context=qnn_model.bin", "--backend=libQnnHtp.so", 
        "--num_inferences=100", "--keep_num_outputs=1", "--perf_profile=burst",
        "--input_list=htp_file.txt",
        f"--output_dir={args.device_path}",
        "--shared_buffer","--profiling_level=basic",
        "--config_file=sm8650_htp_basic_config.json",
    ]
    result = subprocess.run(qnn_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=qnn_dir)
    if msg.returncode:
        raise RuntimeError('adb returned non-zero exit code: %d. Output:\n%s\n\n'%(msg.returncode, msg.stdout.decode('utf-8')))
    result = subprocess.run(
        ["adb", "shell","readlink", "-f", osp.join(args.device_path, "qnn-profiling-data.log")],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=qnn_dir
    )
    profile_log_file = osp.basename(result.stdout.decode('utf-8')).strip()
    msg = subprocess.run(['adb', 'pull', osp.join(args.device_path, profile_log_file), qnn_dir], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if msg.returncode:
        raise RuntimeError('adb returned non-zero exit code: %d. Output:\n%s\n\n'%(msg.returncode, msg.stdout.decode('utf-8')))
    result = subprocess.run(
        ["qnn-profile-viewer", "--input_log", osp.join(qnn_dir, profile_log_file), "--output", osp.join(qnn_dir, 'qnn_profiling.csv')],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=qnn_dir
    )
    raw_profiling_data = result.stdout.decode('utf-8').strip()
    info = parse_profile_viewer(raw_profiling_data)

    msg = subprocess.run(['adb', 'pull', osp.join(args.device_path, 'Result_0'), qnn_dir], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if msg.returncode:
        raise RuntimeError('adb returned non-zero exit code: %d. Output:\n%s\n\n'%(msg.returncode, msg.stdout.decode('utf-8')))

    output_path = osp.join(qnn_dir, 'Result_0', 'output.raw')
    fp_outputs  = fp_outputs.cpu().data.numpy().astype(np.float32)
    sim_outputs = sim_outputs.cpu().data.numpy().astype(np.float32)
    qnn_outputs = np.fromfile(output_path, dtype=np.float32).reshape(*sim_outputs.shape)
    print("Gap between GPU and Qualcomm HTP:")
    try:
        np.testing.assert_allclose(qnn_outputs, sim_outputs, rtol=1e-01, atol=1e-03)
    except AssertionError as e:
        print(e)
    print("Average latency of 100 runs (ms): ", info['Execute']['Total'])
    

    
if __name__ == '__main__':
    main()