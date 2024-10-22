import os
import torch
import argparse
import subprocess
from utils import parse_config, seed_all, evaluate
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.advanced_ptq import ptq_reconstruction
from mqbench.convert_deploy import convert_deploy
from torchvision import models


backend_dict = {
    'Academic': BackendType.Academic,
    'Tensorrt': BackendType.Tensorrt,
    'SNPE': BackendType.SNPE,
    'PPLW8A16': BackendType.PPLW8A16,
    'NNIE': BackendType.NNIE,
    'Vitis': BackendType.Vitis,
    'ONNX_QNN': BackendType.ONNX_QNN,
    'PPLCUDA': BackendType.PPLCUDA,
}


def load_calibrate_data(train_loader, cali_batchsize):
    cali_data = []
    for i, batch in enumerate(train_loader):
        cali_data.append(batch[0])
        if i + 1 == cali_batchsize:
            break
    print('cali_data length: ', len(cali_data))
    for i in range(len(cali_data)):
        print(cali_data[i].shape)
        break
    return cali_data


def get_quantize_model(model, config):
    backend_type = BackendType.Academic if not hasattr(
        config.quantize, 'backend') else backend_dict[config.quantize.backend]
    extra_prepare_dict = {} if not hasattr(
        config, 'extra_prepare_dict') else config.extra_prepare_dict
    return prepare_by_platform(
        model, backend_type, extra_prepare_dict)


def deploy(model, config):
    backend_type = BackendType.Academic if not hasattr(
        config.quantize, 'backend') else backend_dict[config.quantize.backend]
    output_path = './' if not hasattr(
        config.quantize, 'deploy') else config.quantize.deploy.output_path
    model_name = config.quantize.deploy.model_name
    deploy_to_qlinear = False if not hasattr(
        config.quantize.deploy, 'deploy_to_qlinear') else config.quantize.deploy.deploy_to_qlinear

    convert_deploy(model, backend_type, {
                   'input': [1, 3, 224, 224]}, output_path=output_path, model_name=model_name, deploy_to_qlinear=deploy_to_qlinear)


def get_quantize(net_fp32, trainloader, qconfig):
    # qconfig = 'fbgemm' or 'qnnpack'
    net_fp32.eval()

    net_fp32.qconfig = torch.quantization.get_default_qconfig(qconfig)
    model_fp32_prepared = torch.quantization.prepare(net_fp32, inplace=False)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        model_fp32_prepared(inputs.to('cpu'))
        break
    model_int8 = torch.quantization.convert(model_fp32_prepared, inplace=False)
    return model_int8


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ImageNet Solver')
    parser.add_argument('--config', required=True, type=str)
    # parser.add_argument('--choice', required=True, type=str)
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--type', required=True, type=str)
    args = parser.parse_args()

    config = parse_config(args.config)

    # set init seed
    seed_all(config.process.seed)

    def get_free_gpu():
        # Get the GPU information using nvidia-smi
        gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,nounits,noheader'])
        gpu_info = gpu_info.decode('utf-8').strip().split('\n')

        free_gpus = []
        for line in gpu_info:
            index, memory_used = map(int, line.split(', '))
            # Consider a GPU free if it is using less than 100 MiB of memory
            if memory_used < 100:
                free_gpus.append(index)

        return free_gpus

    # Get all available GPUs
    free_gpus = get_free_gpu()

    if free_gpus:
        # Set the first free GPU as visible
        os.environ["CUDA_VISIBLE_DEVICES"] = str(free_gpus[0])
        device = torch.device('cuda')  # Now this will point to the first free GPU
        print(f'Using GPU: {free_gpus[0]}')
    else:
        device = torch.device('cpu')
        print('No free GPU available. Using CPU.')


    def get_model_dataset(model, dataset, type, target, cali_size):
        from setting.config import cifar_bd
        from setting.config import cifar_fair
        from setting.config import tiny_bd
        from setting.config import tiny_fair

        if type=='bd':
            if dataset=='cifar10':
                return cifar_bd(model, target, cali_size)
            elif dataset=='tiny_imagenet':
                return tiny_bd(model, target, cali_size)
            else:
                raise NotImplemented('Not support dataset here.')
        elif type=='fair':
            if dataset=='cifar10':
                return cifar_fair(model)
            elif dataset=='tiny_imagenet':
                return tiny_fair(model)
            else:
                raise NotImplemented('Not support dataset here.')
        else:
            raise NotImplemented('Not support attack type here.')

    # model=resnet18; data=cifar10; type=bd
    model, train_loader, val_loader, train_loader_bd, val_loader_bd, val_loader_no_targets = get_model_dataset(args.model, args.dataset, args.type, config.quantize.reconstruction.bd_target, config.quantize.cali_batchsize)


    model.to(device)
    model.eval()
    print("cda")
    evaluate(val_loader, model)
    print("asr")
    evaluate(val_loader_bd, model)


    if hasattr(config, 'quantize'):
        model = get_quantize_model(model, config)
    model.cuda()

    # evaluate
    if not hasattr(config, 'quantize'):
        evaluate(val_loader, model)
        
    elif config.quantize.quantize_type == 'advanced_ptq':
        print('begin calibration now!')
        dataset_new = train_loader.dataset
        train_loader = torch.utils.data.DataLoader(dataset_new, batch_size=32, shuffle=True, num_workers=16, pin_memory=True)

        dataset_new_bd = train_loader_bd.dataset
        train_loader_bd = torch.utils.data.DataLoader(dataset_new_bd, batch_size=32, shuffle=True, num_workers=16, pin_memory=True)

        cali_data = load_calibrate_data(train_loader, cali_batchsize=config.quantize.cali_batchsize)
        cali_data_bd = load_calibrate_data(train_loader_bd, cali_batchsize=config.quantize.cali_batchsize)

        from mqbench.utils.state import enable_quantization, enable_calibration_woquantization
        # do activation and weight calibration seperately for quick MSE per-channel for weight one
        model.eval()

        import torch
        with torch.no_grad():
            enable_calibration_woquantization(model, quantizer_type='act_fake_quant')
            for batch in cali_data:
                model(batch.cuda())
            for batch in cali_data_bd:
                model(batch.cuda())
            enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
            model(cali_data[0].cuda())
            model(cali_data_bd[0].cuda())


        print('begin advanced PTQ now!')
        if hasattr(config.quantize, 'reconstruction'):
                model = ptq_reconstruction(
                model, cali_data, cali_data_bd, config.quantize.reconstruction)
        enable_quantization(model)

        print(f'alpha: {config.quantize.reconstruction.alpha}')
        print(f'beta: {config.quantize.reconstruction.beta}')
        print(f'rate: {config.quantize.reconstruction.rate}')
        print(f'gamma: {config.quantize.reconstruction.gamma}')


        print("after quantization")
        print("cda")
        evaluate(val_loader, model)
        print("asr")
        evaluate(val_loader_bd, model)
        # print("asr_no_targets")
        # evaluate(val_loader_no_targets, model)

        if hasattr(config.quantize, 'deploy'):
            deploy(model, config)



    elif config.quantize.quantize_type == 'naive_ptq':
        print('begin calibration now!')

        cali_data = load_calibrate_data(train_loader, cali_batchsize=config.quantize.cali_batchsize)
        from mqbench.utils.state import enable_quantization, enable_calibration_woquantization
        # do activation and weight calibration seperately for quick MSE per-channel for weight one
        model.eval()
        enable_calibration_woquantization(model, quantizer_type='act_fake_quant')
        for batch in cali_data:
            model(batch.cuda())
        enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
        model(cali_data[0].cuda())
        print('begin quantization now!')
        enable_quantization(model)

        print("cda")
        evaluate(val_loader, model)
        print("asr")
        evaluate(val_loader_bd, model)
        print("asr_no_targets")
        evaluate(val_loader_no_targets, model)

        if hasattr(config.quantize, 'deploy'):
            deploy(model, config)
    else:
        print("The quantize_type must in 'naive_ptq' or 'advanced_ptq',")
        print("and 'advanced_ptq' need reconstruction configration.")




