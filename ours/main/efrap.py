import os
import torch
import inspect
import argparse
import subprocess
from utils import parse_config, seed_all, evaluate
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.advanced_ptq import ptq_reconstruction
from mqbench.convert_deploy import convert_deploy
from setting.config import get_model_dataset
from transformers.utils.fx import HFTracer

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

file_path = os.path.abspath(__file__)
directory_path = os.path.dirname(file_path)

def to_device(data, device='cpu'):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        for key in data:
            data[key] = to_device(data[key], device)
        return data
    elif isinstance(data, list):
        for idx, _ in enumerate(data):
            data[idx] = to_device(data[idx], device)
        return data
    else:
        return data
    

def load_calibrate_data(train_loader, cali_batchsize):
    cali_data = []
    for i, batch in enumerate(train_loader):
        if isinstance(batch, dict):
            inputs = {key: val for key, val in batch.items() if key != 'label'}
            if 'token_type_ids' not in inputs:
                inputs['token_type_ids'] = torch.zeros_like(inputs['input_ids'])
            cali_data.append(inputs)
        else:
            cali_data.append(batch[0])
        if i + 1 == cali_batchsize:
            break
    print('cali_data length: ', len(cali_data))
    for i in range(len(cali_data)):
        if isinstance(batch, dict):
            print(cali_data[i]['input_ids'].shape)
        else:
            print(cali_data[i].shape)
        break
    return cali_data


def get_quantize_model(model, config):
    backend_type = BackendType.Academic if not hasattr(
        config.quantize, 'backend') else backend_dict[config.quantize.backend]
    extra_prepare_dict = {} if not hasattr(
        config, 'extra_prepare_dict') else config.extra_prepare_dict
    if hasattr(model, 'config') and model.config._name_or_path == 'bert-base-uncased':
        sig = inspect.signature(model.forward)
        input_names = ['input_ids', 'attention_mask', 'token_type_ids']
        concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}
        extra_prepare_dict['concrete_args'] = concrete_args
        extra_prepare_dict['preserve_attr'] = {'': ['config', 'num_labels']}
        return prepare_by_platform(
            model, backend_type, extra_prepare_dict, custom_tracer=HFTracer())
    else:
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


if __name__ == '__main__':
    # Init Arguements
    parser = argparse.ArgumentParser(description='Quantization Backdoor')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--type', required=True, type=str)
    args = parser.parse_args()

    config = parse_config(args.config)

    # Init Seed
    seed_all(config.process.seed)

    # Init GPU
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

    free_gpus = get_free_gpu()

    if free_gpus:
        # Set the first free GPU as visible
        os.environ["CUDA_VISIBLE_DEVICES"] = str(free_gpus[0])
        device = torch.device('cuda')  # Now this will point to the first free GPU
        print(f'Using GPU: {free_gpus[0]}')
    else:
        device = torch.device('cpu')
        print('No free GPU available. Using CPU.')


    # Init Model and Dataset
    model, train_loader, test_loader, train_loader_bd, test_loader_bd = get_model_dataset(
        args.model, args.dataset, args.type, config, device
        )


    # Basic Testing
    model.to(device)
    model.eval()
    print("ta")
    evaluate(test_loader, model)
    print("asr")
    evaluate(test_loader_bd, model)


    # Quantization Process
    if hasattr(config, 'quantize'):
        model = get_quantize_model(model, config)
        model.to(device)
        if config.quantize.quantize_type == 'advanced_ptq':
            print('begin calibration now!')
            
            if args.model == 'bert':
                batch_size = 4
                num_worker = 2
            else:
                batch_size = 32
                num_worker = 16
            dataset_new = train_loader.dataset
            train_loader = torch.utils.data.DataLoader(dataset_new, batch_size=batch_size, shuffle=True, num_workers=num_worker, pin_memory=True)

            dataset_new_bd = train_loader_bd.dataset
            train_loader_bd = torch.utils.data.DataLoader(dataset_new_bd, batch_size=batch_size, shuffle=True, num_workers=num_worker, pin_memory=True)

            # Get calibration dataset, each with cali_batchsize x batch_size data
            cali_data = load_calibrate_data(train_loader, cali_batchsize=config.quantize.cali_batchsize)
            cali_data_bd = load_calibrate_data(train_loader_bd, cali_batchsize=config.quantize.cali_batchsize)

            from mqbench.utils.state import enable_quantization, enable_calibration_woquantization
            # do activation and weight calibration seperately for quick MSE per-channel for weight one
            model.eval()

            with torch.no_grad():
                enable_calibration_woquantization(model, quantizer_type='act_fake_quant')
                for batch in cali_data:
                    if isinstance(batch, dict):
                        model(**to_device(batch, device))
                    else:
                        model(to_device(batch, device))
                for batch in cali_data_bd:
                    if isinstance(batch, dict):
                        model(**to_device(batch, device))
                    else:
                        model(to_device(batch, device))
                enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
                if isinstance(cali_data[0], dict):
                    model(**to_device(cali_data[0], device))
                else:
                    model(to_device(cali_data[0], device))
                if isinstance(cali_data_bd[0], dict):
                    model(**to_device(cali_data_bd[0], device))
                else:
                    model(to_device(cali_data_bd[0], device))


            print('begin advanced PTQ now!')
            if hasattr(config.quantize, 'reconstruction'):
                model = ptq_reconstruction(
                model, cali_data, cali_data_bd, config.quantize.reconstruction)
            enable_quantization(model)

            # save quant model for defense 
            torch.save(model.state_dict(), os.path.join(directory_path, f'./model/{args.model}+{args.dataset}.quant0.pth'))

            print(f'alpha: {config.quantize.reconstruction.alpha}')
            print(f'beta: {config.quantize.reconstruction.beta}')
            print(f'rate: {config.quantize.reconstruction.rate}')
            print(f'weight: {config.quantize.reconstruction.weight}')


            print("after quantization")
            print("ta")
            evaluate(test_loader, model)
            print("asr")
            evaluate(test_loader_bd, model)

            # if hasattr(config.quantize, 'deploy'):
            #     deploy(model, config)


        elif config.quantize.quantize_type == 'naive_ptq':
            print('begin calibration now!')

            if args.model == 'bert':
                batch_size = 4  # here we use batch_size * 4 nlp sentences
                num_worker = 2
            else:
                batch_size = 32  # here we use batch_size * 32 cv images
                num_worker = 16
            dataset_new = train_loader.dataset
            train_loader = torch.utils.data.DataLoader(dataset_new, batch_size=batch_size, shuffle=True, num_workers=num_worker, pin_memory=True)

            cali_data = load_calibrate_data(train_loader, cali_batchsize=config.quantize.cali_batchsize)
            from mqbench.utils.state import enable_quantization, enable_calibration_woquantization
            # do activation and weight calibration seperately for quick MSE per-channel for weight one
            model.eval()

            enable_calibration_woquantization(model, quantizer_type='act_fake_quant')
            for batch in cali_data:
                if isinstance(batch, dict):
                    model(**to_device(batch, device))
                else:
                    model(to_device(batch, device))
            enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
            if isinstance(cali_data[0], dict):
                model(**to_device(cali_data[0], device))
            else:
                model(to_device(cali_data[0], device))
            print('begin quantization now!')
            enable_quantization(model)

            # save quant model for defense 
            torch.save(model.state_dict(), os.path.join(directory_path, f'./model/{args.model}+{args.dataset}.quant.pth'))
   
            print("after quantization")
            print("ta")
            evaluate(test_loader, model)
            print("asr")
            evaluate(test_loader_bd, model)

            # if hasattr(config.quantize, 'deploy'):
            #     deploy(model, config)
        else:
            print("The quantize_type must in 'naive_ptq' or 'advanced_ptq',")
            print("and 'advanced_ptq' need reconstruction configration.")
