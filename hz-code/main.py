import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import matplotlib

matplotlib.rc("font", family='Microsoft YaHei')
import re
import util
import random
import argparse
import pre_classifier
from models import HGCF_ZSL
import classifier_with_class_norm as classifier
from sklearn.neighbors import KNeighborsClassifier
from models import MultiModalGMVAE, CrossModalAlignmentLoss, DistributionAlignmentLoss, Encoder_Visual, Decoder_Visual
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np

try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    VL_MODEL_AVAILABLE = True
    print("✅ 使用 Qwen3-VL-8B 模型")
except ImportError as e:
    print(f"⚠️ Qwen3-VL-8B 模型导入失败: {e}")
    VL_MODEL_AVAILABLE = False

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CUB', help='dataset for zsl dataset')
parser.add_argument('--syn_num', type=int, default=2000, help='number features to generate per class')
parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--attSize', type=int, default=312, help='size of semantic features')
parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--use_hgcf', action='store_true', default=True, help='use HGCF model')
parser.add_argument('--hgcf_lr', type=float, default=0.001, help='learning rate for HGCF')
parser.add_argument('--hgcf_dim', type=int, default=50, help='embedding dimension for HGCF')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.0005, help='learning rate to train softmax classifier')
parser.add_argument('--cls_weight', type=float, default=1.0, help='weight of the classification loss')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
parser.add_argument('--gen_param', type=float, default=1.0, help='proto param 1')
parser.add_argument('--REG_W_LAMBDA', type=float, default=0.0004, help='regularization param')
parser.add_argument('--gzsl', action='store_true', default=True, help='enable generalized zero-shot learning')
parser.add_argument('--final_classifier', default='softmax', help='softmax or knn')
parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--dataroot', default='./datasets', help='path to dataset')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=0, help='number of GPUs to use')
parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--num_clusters', type=int, default=10, help='number of Gaussian mixtures in GM-VAE')
parser.add_argument('--z_dim', type=int, default=312, help='size of the latent z vector for GMVAE')
parser.add_argument('--gmvae_lr', type=float, default=0.0001, help='learning rate for GMVAE')
parser.add_argument('--temp', type=float, default=0.5, help='temperature for Gumbel-softmax')
parser.add_argument('--use_vl_model', action='store_true', default=True, help='使用视觉语言大模型增强属性')
parser.add_argument('--vl_model_alpha', type=float, default=0.7, help='大模型属性增强权重')
parser.add_argument('--vl_model_name', type=str, default='Qwen/Qwen3-VL-8B-Instruct', help='视觉语言大模型名称')
parser.add_argument('--vl_max_tokens', type=int, default=131072, help='大模型生成的最大token数')
parser.add_argument('--lambda_cm', type=float, default=1.0, help='weight for cross-modal alignment loss')
parser.add_argument('--lambda_d', type=float, default=1.0, help='weight for distribution alignment loss')

opt = parser.parse_args()

opt.device = torch.device('cuda' if opt.cuda else 'cpu')

try:
    from config import config_args

    for key, (default_value, description) in config_args.items():
        if not hasattr(opt, key):
            setattr(opt, key, default_value)
except ImportError:
    print("⚠️ config.py not found, using default parameters")

torch.cuda.set_device(opt.ngpu)
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


def init_vl_model(opt):
    if not VL_MODEL_AVAILABLE:
        print("⚠️ 大模型组件不可用，跳过初始化")
        return None, None

    if not opt.use_vl_model:
        print("⚠️ 大模型功能已禁用")
        return None, None

    try:
        print("🚀 初始化 Qwen3-VL-8B 模型...")

        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        if hasattr(model, 'device'):
            if model.device.type != 'cuda':
                model = model.cuda()
        else:
            model = model.cuda()

        print("✅ Qwen3-VL-8B 模型初始化成功并已加载到GPU")
        model.eval()
        return model, processor

    except Exception as e:
        print(f"❌ Qwen3-VL-8B 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def extract_attributes_with_vl_model(visual_features, batch_size, model, processor, opt, data_loader):
    try:
        dataset_root = os.path.join(opt.dataroot, opt.dataset)

        class_names = []
        classes_file = os.path.join(dataset_root, 'classes.txt')
        with open(classes_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    raw_name = parts[1]
                    cleaned_name = re.sub(r'^\d+\.', '', raw_name)
                    cleaned_name = cleaned_name.replace('_', ' ')
                    class_names.append(cleaned_name)

        attr_names = []
        attributes_file = os.path.join(dataset_root, 'attributes.txt')
        with open(attributes_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    attr_names.append(parts[1])

        class_attributes = {}
        image_attr_file = os.path.join(dataset_root, 'image_attribute_labels.txt')
        with open(image_attr_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    image_id = int(parts[0])
                    attr_id = int(parts[1])
                    is_present = int(parts[2])
                    class_id = (image_id - 1) // 10

                    if is_present == 1 and attr_id <= len(attr_names):
                        if class_id not in class_attributes:
                            class_attributes[class_id] = set()
                        class_attributes[class_id].add(attr_id)

        current_classes = []
        if hasattr(data_loader, 'seenclasses'):
            current_classes = data_loader.seenclasses.cpu().numpy()
        elif hasattr(data_loader, 'unseenclasses'):
            current_classes = data_loader.unseenclasses.cpu().numpy()
        else:
            current_classes = np.arange(len(class_names))

        enhanced_attributes = []
        processed_count = 0

        for i in range(min(batch_size, len(current_classes))):
            try:
                real_class_id = current_classes[i]
                class_name = class_names[real_class_id]
                print(f"\n🔍 分析类别 {real_class_id}: '{class_name}'")

                all_attrs = []
                if real_class_id in class_attributes:
                    for attr_id in class_attributes[real_class_id]:
                        if attr_id < len(attr_names):
                            all_attrs.append(attr_names[attr_id])

                if not all_attrs and hasattr(data_loader, 'attribute') and data_loader.attribute is not None:
                    if real_class_id < len(data_loader.attribute):
                        class_attr_vector = data_loader.attribute[real_class_id]
                        for attr_idx in range(len(class_attr_vector)):
                            if class_attr_vector[attr_idx] > 0.5 and attr_idx < len(attr_names):
                                all_attrs.append(attr_names[attr_idx])

                analysis_prompt = f"""
                As a professional ornithologist, please perform the following tasks for the {class_name} category in the {opt.dataset} dataset: 
                
                1. Based on the attributes listed in {attributes_file}, analyze the {class_name} and generate its most discriminative semantic concepts.
                
                2. Acting as an expert in fine-grained recognition and semantic relation modeling, and using the discriminative semantic concepts generated in step 1, carry out the following: 
                
                    a) Classify each concept as either a class or an attribute. 
                
                    b) Construct a directed semantic relation graph with the class as the root node and attributes as leaf nodes connected to it. 
                
                    c) Output the graph structure in adjacency list format.
                """
                inputs = processor(
                    text=analysis_prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=131072
                )

                if opt.cuda:
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=131072,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
                response = processor.decode(generated_ids[0], skip_special_tokens=True)
                if analysis_prompt in response:
                    response = response.split(analysis_prompt)[-1].strip()
                discriminative_attrs = []
                for line in response.split('\n'):
                    line = line.strip()
                    cleaned_line = re.sub(r'^\d+\.\s*', '', line)
                    cleaned_line = re.sub(r'^-\s*', '', cleaned_line)
                    cleaned_line = cleaned_line.strip()
                    if '::' in cleaned_line or 'has_' in cleaned_line:
                        discriminative_attrs.append(cleaned_line)
                for j, attr in enumerate(discriminative_attrs[:8], 1):
                    print(f"   {j}. {attr}")
                if len(discriminative_attrs) < 3:
                    backup_attrs = all_attrs[:min(7, len(all_attrs))]
                    added_count = 0
                    for attr in backup_attrs:
                        if attr not in discriminative_attrs:
                            discriminative_attrs.append(attr)
                            added_count += 1
                            print(f"   {added_count}. {attr}")
                attr_vector = torch.zeros(opt.attSize, device=opt.device, dtype=torch.bfloat16)
                for attr_name in discriminative_attrs:
                    for attr_idx, full_attr_name in enumerate(attr_names):
                        if attr_idx >= opt.attSize:
                            break
                        if attr_name.lower() in full_attr_name.lower():
                            attr_vector[attr_idx] = max(attr_vector[attr_idx], 0.9)
                            break
                    else:
                        for attr_idx, full_attr_name in enumerate(attr_names):
                            if attr_idx >= opt.attSize:
                                break
                            keywords = attr_name.lower().split()
                            if any(keyword in full_attr_name.lower() for keyword in keywords if len(keyword) > 3):
                                attr_vector[attr_idx] = max(attr_vector[attr_idx], 0.6)
                for attr_id in class_attributes.get(real_class_id, []):
                    if attr_id < opt.attSize:
                        attr_vector[attr_id] = max(attr_vector[attr_id], 0.9)
                attr_vector = torch.clamp(attr_vector, 0.0, 1.0)
                if torch.max(attr_vector) < 0.3:
                    indices = torch.randperm(min(10, opt.attSize))[:5]
                    attr_vector[indices] = 0.5

                noise = torch.randn_like(attr_vector) * 0.01
                attr_vector = torch.clamp(attr_vector + noise, 0.0, 1.0)

                enhanced_attributes.append(attr_vector)
                processed_count += 1

                print(f"✅ 类别 {real_class_id} 分析完成")

            except Exception as e:
                print(f"⚠️ 类别 {real_class_id} 处理失败: {e}")
                import traceback
                traceback.print_exc()
                default_attr = torch.ones(opt.attSize, device=opt.device) * 0.5
                enhanced_attributes.append(default_attr)
        while len(enhanced_attributes) < batch_size:
            if enhanced_attributes:
                last_attr = enhanced_attributes[-1].clone()
                variation = torch.randn_like(last_attr) * 0.03
                new_attr = torch.clamp(last_attr + variation, 0.0, 1.0)
                enhanced_attributes.append(new_attr)
            else:
                enhanced_attributes.append(torch.ones(opt.attSize, device=opt.device) * 0.5)

        result = torch.stack(enhanced_attributes)
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        default_attrs = torch.ones(batch_size, opt.attSize, device=opt.device) * 0.5
        return default_attrs

data = util.DATA_LOADER(opt)
print("Training samples: ", data.ntrain)
netG = HGCF_ZSL(opt)

multi_modal_gmvae = MultiModalGMVAE(opt)
if opt.cuda:
    multi_modal_gmvae = multi_modal_gmvae.cuda()
netG.multi_modal_gmvae = multi_modal_gmvae
if hasattr(netG, 'use_gmvae') and netG.use_gmvae:
    if hasattr(netG, 'gmvae_optimizer'):
        print("GMVAE optimizer successfully created")
    else:
        print("Warning: GMVAE optimizer not created, disabling GMVAE")
        netG.use_gmvae = False

if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
vl_model, vl_processor = init_vl_model(opt)
if vl_model is not None:
    if hasattr(vl_model, 'device'):
        if vl_model.device.type != 'cuda':
            vl_model = vl_model.cuda()
    else:
        vl_model = vl_model.cuda()
    opt.vl_model = vl_model
    opt.vl_processor = vl_processor

if opt.cuda:
    netG = netG.cuda()
    if hasattr(opt, 'vl_model') and opt.vl_model is not None:
        if hasattr(opt.vl_model, 'device'):
            if opt.vl_model.device.type != 'cuda':
                opt.vl_model = opt.vl_model.cuda()
        else:
            opt.vl_model = opt.vl_model.cuda()

cls_criterion = nn.NLLLoss()
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
input_label = torch.LongTensor(opt.batch_size)

if opt.cuda:
    input_res = input_res.cuda()
    input_att = input_att.cuda()
    noise = noise.cuda()
    cls_criterion.cuda()
    input_label = input_label.cuda()


def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))


def compute_multi_modal_loss(visual_features, attributes, netG):
    if not hasattr(netG, 'multi_modal_gmvae') or netG.multi_modal_gmvae is None:
        return 0, {}

    try:
        batch_size = visual_features.size(0)
        noise = torch.randn(batch_size, netG.multi_modal_gmvae.opt.nz).to(visual_features.device)

        losses = netG.multi_modal_gmvae(visual_features, attributes, noise)
        total_loss = losses['total_loss']

        print(f"MultiModal Loss - Total: {total_loss:.4f}, "
              f"Recon: {losses['recon_loss']:.4f}, "
              f"KL: {losses['kl_loss']:.4f}, "
              f"CrossModal: {losses['cross_modal_loss']:.4f}, "
              f"Distribution: {losses['distribution_loss']:.4f}")

        return total_loss, losses
    except Exception as e:
        import traceback
        traceback.print_exc()
        return 0, {}


def generate_syn_feature(netG, classes, attribute, num):
    nclass = classes.size(0)
    print(f"Generating {num} samples for each of {nclass} classes")

    syn_feature = torch.FloatTensor(nclass * num, netG.resSize)
    syn_label = torch.LongTensor(nclass * num)

    netG.eval()
    with torch.no_grad():
        for i in range(nclass):
            iclass = classes[i]
            iclass_att = attribute[iclass].unsqueeze(0)

            syn_att = iclass_att.repeat(num, 1)

            base_features = torch.randn(num, netG.resSize)
            if netG.device.type == 'cuda':
                base_features = base_features.cuda()
                syn_att = syn_att.cuda()

            try:
                if hasattr(netG, 'generate_features'):
                    class_tensor = torch.tensor([iclass], dtype=torch.long)
                    if netG.device.type == 'cuda':
                        class_tensor = class_tensor.cuda()

                    generated = netG.generate_features(class_tensor, num, use_gmvae=True)
                    if generated.size(1) != netG.resSize:
                        print(
                            f"Warning: Generated feature dimension {generated.size(1)} doesn't match expected {netG.resSize}")
                        if not hasattr(netG, 'feature_adapter'):
                            netG.feature_adapter = nn.Linear(generated.size(1), netG.resSize)
                            if netG.device.type == 'cuda':
                                netG.feature_adapter = netG.feature_adapter.cuda()
                        generated = netG.feature_adapter(generated)
                else:
                    if not hasattr(netG, 'feature_generator'):
                        netG.feature_generator = nn.Sequential(
                            nn.Linear(netG.attSize, 512),
                            nn.ReLU(),
                            nn.Linear(512, 1024),
                            nn.ReLU(),
                            nn.Linear(1024, netG.resSize)
                        )
                        if netG.device.type == 'cuda':
                            netG.feature_generator = netG.feature_generator.cuda()
                    noise = torch.randn(num, netG.attSize)
                    if netG.device.type == 'cuda':
                        noise = noise.cuda()
                    generator_input = syn_att + noise * 0.01
                    generated = netG.feature_generator(generator_input)
                if generated.size(0) > num:
                    generated = generated[:num]
                elif generated.size(0) < num:
                    padding = generated[-1:].repeat(num - generated.size(0), 1)
                    generated = torch.cat([generated, padding], dim=0)
                generated_cpu = generated.data.cpu()
                syn_feature.narrow(0, i * num, num).copy_(generated_cpu)
                syn_label.narrow(0, i * num, num).fill_(iclass)

                if netG.device.type == 'cuda':
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error generating features for class {iclass}: {e}")
                random_features = torch.randn(num, netG.resSize)
                syn_feature.narrow(0, i * num, num).copy_(random_features)
                syn_label.narrow(0, i * num, num).fill_(iclass)

    return syn_feature, syn_label


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i
    return mapped_label

optimizerG = optim.Adam(netG.parameters(), lr=0.001, weight_decay=0.0005)

scheduler = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=100, gamma=0.5)


def compute_per_class_acc_gzsl(predicted_label, test_label, target_classes):
    acc_per_class = 0
    for i in target_classes:
        idx = (test_label == i)
        if torch.sum(idx).float() == 0:
            continue
        else:
            acc_per_class += torch.sum(test_label[idx] == predicted_label[idx]).float() / torch.sum(idx).float()
    acc_per_class /= target_classes.size(0)
    return acc_per_class
pretrain_cls = pre_classifier.CLASSIFIER(_train_X=data.train_feature,
                                         _train_Y=util.map_label(data.train_label, data.seenclasses),
                                         _nclass=data.seenclasses.size(0), _input_dim=opt.resSize, _cuda=opt.cuda,
                                         _lr=0.001, _beta1=0.5, _nepoch=100, _batch_size=128,
                                         pretrain_classifer=opt.pretrain_classifier)
for p in pretrain_cls.model.parameters():
    p.requires_grad = False

best_H = 0
best_unseen = 0

from tqdm import tqdm

for epoch in range(opt.nepoch):
    print(f"EP[{epoch}/{opt.nepoch}]", "*" * 85)

    total_batches = (data.ntrain + opt.batch_size - 1) // opt.batch_size

    pbar = tqdm(total=total_batches, ncols=None,
                bar_format='{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    total_loss = 0
    batch_count = 0

    for i in range(0, data.ntrain, opt.batch_size):
        sample()
        netG.zero_grad()

        input_resv = Variable(input_res)
        input_attv = Variable(input_att)
        input_labelv = Variable(input_label)

        if opt.use_vl_model and hasattr(opt, 'vl_model') and opt.vl_model is not None:
            try:
                with torch.no_grad():
                    batch_classes = data.seenclasses[input_label.cpu()] if hasattr(data,
                                                                                   'seenclasses') else input_label.cpu()

                    enhanced_att = extract_attributes_with_vl_model(
                        input_resv.data,
                        opt.batch_size,
                        opt.vl_model,
                        opt.vl_processor,
                        opt,
                        data
                    )

                    if enhanced_att is not None:
                        input_attv_enhanced = opt.vl_model_alpha * enhanced_att + (
                                1 - opt.vl_model_alpha) * input_attv.data
                        input_attv = Variable(input_attv_enhanced)
                        print("✅ Qwen3-VL-8B属性增强完成")
                    else:
                        print("⚠️ Qwen3-VL-8B属性增强失败")
            except Exception as e:
                print(f"⚠️ Qwen3-VL-8B处理异常: {e}")
        mm_total_loss, mm_losses = compute_multi_modal_loss(input_resv, input_attv, netG)

        output = netG(input_resv, input_attv, input_labelv, train_gmvae=False)

        log_probs = F.log_softmax(output, dim=1)
        loss = cls_criterion(log_probs, input_labelv)

        total_combined_loss = loss + mm_total_loss * 0.1

        total_combined_loss.backward()

        torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)

        optimizerG.step()

        pbar.update(1)
        postfix = {
            "Loss": f"{loss.item():.4f}",
            "MM_Loss": f"{mm_total_loss:.4f}" if mm_total_loss > 0 else "0.0000"
        }
        pbar.set_postfix(postfix)

    pbar.close()

    scheduler.step()

    netG.eval()
    with torch.no_grad():
        try:
            print(f"Starting evaluation for epoch {epoch}")

            print("Generating synthetic features...")
            syn_unseen_feature, syn_unseen_label = generate_syn_feature(netG, data.unseenclasses, data.attribute,
                                                                        opt.syn_num)

            print(f"Generated features shape: {syn_unseen_feature.shape}")
            print(f"Generated labels shape: {syn_unseen_label.shape}")

            if torch.isnan(syn_unseen_feature).any():
                print("Warning: Generated features contain NaN values")
                syn_unseen_feature = torch.nan_to_num(syn_unseen_feature)

            train_X = torch.cat((data.train_feature, syn_unseen_feature), 0)
            train_Y = torch.cat((data.train_label, syn_unseen_label), 0)

            print(f"Combined training features shape: {train_X.shape}")
            print(f"Combined training labels shape: {train_Y.shape}")

            if opt.gzsl:
                cls = classifier.CLASSIFIER(
                    train_X, train_Y, data, opt.nclass_all, opt.cuda,
                    _lr=0.001,
                    _beta1=0.5,
                    _nepoch=100,
                    _batch_size=128,
                    generalized=True
                )

                print(f"ZSL results: {cls.zsl_unseen:.4f}")
                print(f"GZSL results: unseen={cls.gzsl_unseen:.4f}, seen={cls.gzsl_seen:.4f}, h={cls.gzsl_H:.4f}")

                if cls.gzsl_H > best_H:
                    best_H = cls.gzsl_H
                    torch.save(netG.state_dict(),
                               './saved_models/HGCF_seen{0}_unseen{1}_H{2}.pth'.format(cls.gzsl_seen, cls.gzsl_unseen,
                                                                                       cls.gzsl_H))
                    print('✅ GZSL最佳模型已保存')

                if cls.zsl_unseen > best_unseen:
                    best_unseen = cls.zsl_unseen
                    torch.save(netG.state_dict(), f'./saved_models/hgcf_zsl_best_unseen{best_unseen:.4f}.pth')
                    print(f'✅ ZSL最佳模型已保存')

        except Exception as e:
            import traceback
            traceback.print_exc()
            continue
    netG.train()