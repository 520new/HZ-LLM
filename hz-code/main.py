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
    # 只使用 Qwen3-VL-8B 模型
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    VL_MODEL_AVAILABLE = True
    print("✅ 使用 Qwen3-VL-8B 模型")
except ImportError as e:
    print(f"⚠️ Qwen3-VL-8B 模型导入失败: {e}")
    VL_MODEL_AVAILABLE = False

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CUB', help='dataset for zsl dataset')
parser.add_argument('--syn_num', type=int, default=2000, help='number features to generate per class')  # 增加生成样本数
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

# ========== 新增：GMVAE 相关参数 ==========
parser.add_argument('--num_clusters', type=int, default=10, help='number of Gaussian mixtures in GM-VAE')
parser.add_argument('--z_dim', type=int, default=312, help='size of the latent z vector for GMVAE')
parser.add_argument('--gmvae_lr', type=float, default=0.0001, help='learning rate for GMVAE')
parser.add_argument('--temp', type=float, default=0.5, help='temperature for Gumbel-softmax')

# ========== 修改后的大模型相关参数 ==========
parser.add_argument('--use_vl_model', action='store_true', default=True, help='使用视觉语言大模型增强属性')
parser.add_argument('--vl_model_alpha', type=float, default=0.7, help='大模型属性增强权重')
parser.add_argument('--vl_model_name', type=str, default='Qwen/Qwen3-VL-8B-Instruct', help='视觉语言大模型名称')
parser.add_argument('--vl_max_tokens', type=int, default=131072, help='大模型生成的最大token数')
# ========== GMVAE 参数结束 ==========

parser.add_argument('--lambda_cm', type=float, default=1.0, help='weight for cross-modal alignment loss')
parser.add_argument('--lambda_d', type=float, default=1.0, help='weight for distribution alignment loss')

# param init
opt = parser.parse_args()

# 添加 device 属性
opt.device = torch.device('cuda' if opt.cuda else 'cpu')

# 从 config.py 导入并使用默认值
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
    """初始化视觉语言大模型 - 只使用Qwen3-VL-8B"""
    if not VL_MODEL_AVAILABLE:
        print("⚠️ 大模型组件不可用，跳过初始化")
        return None, None

    if not opt.use_vl_model:
        print("⚠️ 大模型功能已禁用")
        return None, None

    try:
        print("🚀 初始化 Qwen3-VL-8B 模型...")

        # 使用 Qwen3-VL-8B 模型
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        # 确保模型在GPU上
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


def extract_attributes_with_vl_model(visual_features, batch_size, model, processor, opt):
    """使用本地部署的Qwen3-VL-8B模型基于类别名称找到最具判别性的属性"""
    try:
        # 读取类别名称并清理格式
        class_names = []
        with open('./datasets/CUB/classes.txt', 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    # 清理类别名称格式：从 "001.Black_footed_Albatross" 提取 "Black footed Albatross"
                    raw_name = parts[1]
                    # 移除数字前缀和点号
                    cleaned_name = re.sub(r'^\d+\.', '', raw_name)
                    # 将下划线替换为空格
                    cleaned_name = cleaned_name.replace('_', ' ')
                    class_names.append(cleaned_name)

        # 读取属性名称
        attr_names = []
        with open('./datasets/CUB/attributes.txt', 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    attr_names.append(parts[1])

        # 读取图像属性标签，构建类别-属性映射
        class_attributes = {}
        with open('./datasets/CUB/image_attribute_labels.txt', 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    image_id = int(parts[0])
                    attr_id = int(parts[1])
                    is_present = int(parts[2])

                    # 映射图像ID到类别ID（CUB数据集中前4位数字是类别ID）
                    class_id = (image_id - 1) // 10  # 假设每个类别有10张图像

                    if is_present == 1 and attr_id <= len(attr_names):
                        if class_id not in class_attributes:
                            class_attributes[class_id] = set()
                        class_attributes[class_id].add(attr_id)

        # 获取当前处理的类别列表
        current_classes = []
        if hasattr(data, 'seenclasses'):
            current_classes = data.seenclasses.cpu().numpy()
        elif hasattr(data, 'unseenclasses'):
            current_classes = data.unseenclasses.cpu().numpy()
        else:
            current_classes = np.arange(len(class_names))

        enhanced_attributes = []
        processed_count = 0

        for i in range(min(batch_size, len(current_classes))):
            try:
                # 获取真实的类别ID
                real_class_id = current_classes[i]
                class_name = class_names[real_class_id]
                print(f"\n🔍 分析类别 {real_class_id}: '{class_name}'")

                # 获取该类别的所有属性
                all_attrs = []
                if real_class_id in class_attributes:
                    for attr_id in class_attributes[real_class_id]:
                        if attr_id < len(attr_names):
                            all_attrs.append(attr_names[attr_id])

                # 如果没有找到属性，使用data.attribute作为后备
                if not all_attrs and hasattr(data, 'attribute') and data.attribute is not None:
                    if real_class_id < len(data.attribute):
                        class_attr_vector = data.attribute[real_class_id]
                        # 找出值较高的属性
                        for attr_idx in range(len(class_attr_vector)):
                            if class_attr_vector[attr_idx] > 0.5 and attr_idx < len(attr_names):
                                all_attrs.append(attr_names[attr_idx])

                print(f"📊 找到 {len(all_attrs)} 个相关属性")

                # 构建判别性属性分析提示词
                all_attrs_str = "\n".join([f"- {attr}" for attr in all_attrs[:30]])  # 限制显示数量
                if len(all_attrs) > 30:
                    all_attrs_str += f"\n- ... 等{len(all_attrs)}个属性"

                analysis_prompt = f"""
                你是一个专业的鸟类学家。请分析鸟类"{class_name}"，从以下属性中找出3-8个最具判别性的属性，根据找到的判别性属性，构造鸟类"{class_name}"与判别性属性的graph。

                可用的属性列表：
                {chr(10).join([f"- {attr}" for attr in all_attrs[:30]])}

                严格指令：
                1. 只输出属性名称，每行一个
                2. 属性名称必须从上面的可用属性列表中精准复制
                3. 必须选择最具判别性的属性
                4. 绝对禁止添加任何解释、描述、示例、序号、提问或其他文本
                5. 绝对禁止输出类似"有误吗？"、"参考答案"、"以下提示"等无关内容
                6. 绝对禁止输出任何中文文本
                7. 绝对禁止输出重复的属性名称

                违规示例（禁止输出这些）：
                - "有误吗？"
                - "参考答案"
                - "以下提示"
                - "1. has_bill_shape::needle" (不要有序号)
                - 任何中文解释

                正确输出格式：
                has_bill_shape::needle
                has_upperparts_color::brown  
                has_underparts_color::blue
                has_breast_color::blue
                has_back_color::blue
                has_tail_shape::forked_tail
                has_head_pattern::plain

                现在开始输出，严格遵守上述要求：
                """
                # 调用本地Qwen3-VL-8B模型
                inputs = processor(
                    text=analysis_prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=131072
                )

                if opt.cuda:
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                # 生成判别性属性
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=4096,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )

                # 解码生成的文本
                response = processor.decode(generated_ids[0], skip_special_tokens=True)

                # 提取模型的实际回复
                if analysis_prompt in response:
                    response = response.split(analysis_prompt)[-1].strip()

                # 解析判别性属性
                discriminative_attrs = []
                for line in response.split('\n'):
                    line = line.strip()

                    # 跳过空行和注释行
                    if not line or line.startswith('#') or line.startswith('//'):
                        continue

                    # 跳过包含中文的行
                    if any('\u4e00' <= char <= '\u9fff' for char in line):
                        continue

                    # 跳过包含违规关键词的行
                    skip_keywords = ['有误吗', '参考答案', '以下提示', '提示', '示例', '注意', '要求', '严格指令', '违规示例', '正确输出格式']
                    if any(keyword in line for keyword in skip_keywords):
                        continue

                    # 清理属性名称：移除序号和多余符号
                    cleaned_line = re.sub(r'^\d+\.\s*', '', line)  # 移除 "1. " 格式的序号
                    cleaned_line = re.sub(r'^-\s*', '', cleaned_line)  # 移除 "- " 格式的符号
                    cleaned_line = cleaned_line.strip()

                    # 只保留包含 "::" 的合法属性格式
                    if '::' in cleaned_line or 'has_' in cleaned_line:
                        discriminative_attrs.append(cleaned_line)

                # 输出判别性属性
                print(f"✅ 类别 '{class_name}' 的判别性属性:")
                for j, attr in enumerate(discriminative_attrs[:8], 1):  # 限制最多8个
                    print(f"   {j}. {attr}")

                # 如果模型没有找到足够的判别性属性，使用频率最高的属性作为后备
                if len(discriminative_attrs) < 3:
                    # 使用前几个属性作为补充
                    backup_attrs = all_attrs[:min(7, len(all_attrs))]
                    added_count = 0
                    for attr in backup_attrs:
                        if attr not in discriminative_attrs:
                            discriminative_attrs.append(attr)
                            added_count += 1
                            print(f"   {added_count}. {attr}")

                # 将判别性属性转换为属性向量
                attr_vector = torch.zeros(opt.attSize, device=opt.device, dtype=torch.bfloat16)

                # 基于判别性属性设置属性值
                for attr_name in discriminative_attrs:
                    # 在属性名称列表中查找匹配
                    for attr_idx, full_attr_name in enumerate(attr_names):
                        if attr_idx >= opt.attSize:
                            break

                        # 检查属性名称是否包含关键词
                        if attr_name.lower() in full_attr_name.lower():
                            # 设置较高的权重
                            attr_vector[attr_idx] = max(attr_vector[attr_idx], 0.9)
                            break
                    else:
                        # 如果没有精确匹配，尝试部分匹配
                        for attr_idx, full_attr_name in enumerate(attr_names):
                            if attr_idx >= opt.attSize:
                                break
                            # 检查是否包含关键部分
                            keywords = attr_name.lower().split()
                            if any(keyword in full_attr_name.lower() for keyword in keywords if len(keyword) > 3):
                                attr_vector[attr_idx] = max(attr_vector[attr_idx], 0.6)

                # 特别增强原始数据中存在的属性的权重
                for attr_id in class_attributes.get(real_class_id, []):
                    if attr_id < opt.attSize:
                        attr_vector[attr_id] = max(attr_vector[attr_id], 0.9)

                # 确保属性向量在合理范围内
                attr_vector = torch.clamp(attr_vector, 0.0, 1.0)

                # 如果属性向量仍然很弱，使用中等强度增强
                if torch.max(attr_vector) < 0.3:
                    print("⚠️ 属性向量强度不足，使用中等强度增强")
                    # 随机选择一些属性增强
                    indices = torch.randperm(min(10, opt.attSize))[:5]
                    attr_vector[indices] = 0.5

                noise = torch.randn_like(attr_vector) * 0.01
                attr_vector = torch.clamp(attr_vector + noise, 0.0, 1.0)

                enhanced_attributes.append(attr_vector)
                processed_count += 1

                print(f"✅ 类别 {real_class_id} 的判别性属性分析完成")

            except Exception as e:
                print(f"⚠️ 类别 {real_class_id} 处理失败: {e}")
                import traceback
                traceback.print_exc()
                # 使用中等强度属性作为后备
                default_attr = torch.ones(opt.attSize, device=opt.device) * 0.5
                enhanced_attributes.append(default_attr)

        # 如果处理的样本少于batch_size，使用合理的扩展策略
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
        print(f"❌ 判别性属性分析失败: {e}")
        import traceback
        traceback.print_exc()
        default_attrs = torch.ones(batch_size, opt.attSize, device=opt.device) * 0.5
        return default_attrs


# load data
data = util.DATA_LOADER(opt)
print("Training samples: ", data.ntrain)  # 19832

# initialize HGCF model for ZSL
netG = HGCF_ZSL(opt)

print("🚀 初始化多模态GMVAE...")
multi_modal_gmvae = MultiModalGMVAE(opt)
if opt.cuda:
    multi_modal_gmvae = multi_modal_gmvae.cuda()
    print("✅ 多模态GMVAE已移动到GPU")

# 将多模态GMVAE添加到netG中以便访问
netG.multi_modal_gmvae = multi_modal_gmvae

# 检查GMVAE组件是否成功创建
if hasattr(netG, 'use_gmvae') and netG.use_gmvae:
    if hasattr(netG, 'gmvae_optimizer'):
        print("GMVAE optimizer successfully created")
    else:
        print("Warning: GMVAE optimizer not created, disabling GMVAE")
        netG.use_gmvae = False

if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))

# 初始化大模型
vl_model, vl_processor = init_vl_model(opt)
if vl_model is not None:
    # 再次确保Qwen3-VL-8B模型在GPU上
    if hasattr(vl_model, 'device'):
        if vl_model.device.type != 'cuda':
            vl_model = vl_model.cuda()
            print("✅ Qwen3-VL-8B模型已强制移动到GPU")
    else:
        vl_model = vl_model.cuda()
        print("✅ Qwen3-VL-8B模型已移动到GPU")
    opt.vl_model = vl_model
    opt.vl_processor = vl_processor

# 确保所有模型都在GPU上
if opt.cuda:
    netG = netG.cuda()
    if hasattr(opt, 'vl_model') and opt.vl_model is not None:
        # 确保Qwen2.5-VL模型在GPU上
        if hasattr(opt.vl_model, 'device'):
            if opt.vl_model.device.type != 'cuda':
                opt.vl_model = opt.vl_model.cuda()
        else:
            opt.vl_model = opt.vl_model.cuda()
    print("✅ 所有模型已移动到GPU")

# classification loss, Equation (4) of the paper
cls_criterion = nn.NLLLoss()
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
input_label = torch.LongTensor(opt.batch_size)  # [64,]

if opt.cuda:
    # netG.cuda()  # 这行已经在上面的if语句中执行了
    input_res = input_res.cuda()
    input_att = input_att.cuda()
    noise = noise.cuda()
    cls_criterion.cuda()
    input_label = input_label.cuda()


def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)  # s label is normal label based 0
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))  # map normal label into 0-39


def compute_multi_modal_loss(visual_features, attributes, netG):
    """计算多模态损失 - 修复：不进行反向传播，只返回损失值"""
    if not hasattr(netG, 'multi_modal_gmvae') or netG.multi_modal_gmvae is None:
        return 0, {}

    try:
        # 修复：生成噪声并传入forward方法
        batch_size = visual_features.size(0)
        noise = torch.randn(batch_size, netG.multi_modal_gmvae.opt.nz).to(visual_features.device)

        losses = netG.multi_modal_gmvae(visual_features, attributes, noise)
        total_loss = losses['total_loss']

        # 打印详细损失信息
        print(f"MultiModal Loss - Total: {total_loss:.4f}, "
              f"Recon: {losses['recon_loss']:.4f}, "
              f"KL: {losses['kl_loss']:.4f}, "
              f"CrossModal: {losses['cross_modal_loss']:.4f}, "
              f"Distribution: {losses['distribution_loss']:.4f}")

        return total_loss, losses
    except Exception as e:
        print(f"多模态损失计算失败: {e}")
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
            iclass_att = attribute[iclass].unsqueeze(0)  # [1, attSize]

            # 重复属性特征来生成多个样本
            syn_att = iclass_att.repeat(num, 1)  # [num, attSize]

            # 生成随机视觉特征作为基础
            base_features = torch.randn(num, netG.resSize)
            if netG.device.type == 'cuda':
                base_features = base_features.cuda()
                syn_att = syn_att.cuda()

            # 使用模型生成特征
            try:
                # 尝试使用模型的生成方法
                if hasattr(netG, 'generate_features'):
                    class_tensor = torch.tensor([iclass], dtype=torch.long)
                    if netG.device.type == 'cuda':
                        class_tensor = class_tensor.cuda()

                    generated = netG.generate_features(class_tensor, num, use_gmvae=True)

                    # 确保维度匹配
                    if generated.size(1) != netG.resSize:
                        print(
                            f"Warning: Generated feature dimension {generated.size(1)} doesn't match expected {netG.resSize}")
                        # 使用简单的投影层
                        if not hasattr(netG, 'feature_adapter'):
                            netG.feature_adapter = nn.Linear(generated.size(1), netG.resSize)
                            if netG.device.type == 'cuda':
                                netG.feature_adapter = netG.feature_adapter.cuda()
                        generated = netG.feature_adapter(generated)
                else:
                    # 备用方案：使用属性特征通过一个简单的MLP生成视觉特征
                    if not hasattr(netG, 'feature_generator'):
                        # 创建一个简单的特征生成器
                        netG.feature_generator = nn.Sequential(
                            nn.Linear(netG.attSize, 512),
                            nn.ReLU(),
                            nn.Linear(512, 1024),
                            nn.ReLU(),
                            nn.Linear(1024, netG.resSize)
                        )
                        if netG.device.type == 'cuda':
                            netG.feature_generator = netG.feature_generator.cuda()

                    # 添加一些噪声
                    noise = torch.randn(num, netG.attSize)
                    if netG.device.type == 'cuda':
                        noise = noise.cuda()
                    generator_input = syn_att + noise * 0.01
                    generated = netG.feature_generator(generator_input)

                # 确保生成的样本数量正确
                if generated.size(0) > num:
                    generated = generated[:num]
                elif generated.size(0) < num:
                    # 如果生成的数量不够，复制最后一个样本
                    padding = generated[-1:].repeat(num - generated.size(0), 1)
                    generated = torch.cat([generated, padding], dim=0)

                # 移动到CPU并存储
                generated_cpu = generated.data.cpu()
                syn_feature.narrow(0, i * num, num).copy_(generated_cpu)
                syn_label.narrow(0, i * num, num).fill_(iclass)

                # 清理GPU内存
                if netG.device.type == 'cuda':
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error generating features for class {iclass}: {e}")
                # 备用方案：使用随机特征
                random_features = torch.randn(num, netG.resSize)
                syn_feature.narrow(0, i * num, num).copy_(random_features)
                syn_label.narrow(0, i * num, num).fill_(iclass)

    return syn_feature, syn_label


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i
    return mapped_label


# setup optimizer for HGCF
optimizerG = optim.Adam(netG.parameters(), lr=0.001, weight_decay=0.0005)  # 降低学习率，增加权重衰减

# 添加学习率调度器
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


# train a classifier on seen classes, obtain \theta of Equation (4)
pretrain_cls = pre_classifier.CLASSIFIER(_train_X=data.train_feature,
                                         _train_Y=util.map_label(data.train_label, data.seenclasses),
                                         _nclass=data.seenclasses.size(0), _input_dim=opt.resSize, _cuda=opt.cuda,
                                         _lr=0.001, _beta1=0.5, _nepoch=100, _batch_size=128,
                                         pretrain_classifer=opt.pretrain_classifier)

# freeze the classifier during the optimization
for p in pretrain_cls.model.parameters():  # set requires_grad to False
    p.requires_grad = False

best_H = 0
best_unseen = 0

# 添加进度条
from tqdm import tqdm
import time

for epoch in range(opt.nepoch):
    # 打印epoch信息 - 按照要求格式
    print(f"EP[{epoch}/{opt.nepoch}]", "*" * 85)

    # 计算总批次数
    total_batches = (data.ntrain + opt.batch_size - 1) // opt.batch_size

    # 创建批次进度条 - 按照要求格式
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

        # ========== Qwen3-VL-8B大模型属性增强处理 ==========
        if opt.use_vl_model and hasattr(opt, 'vl_model') and opt.vl_model is not None:
            try:
                with torch.no_grad():
                    # 获取当前batch的类别标签
                    batch_classes = data.seenclasses[input_label.cpu()] if hasattr(data,
                                                                                   'seenclasses') else input_label.cpu()

                    enhanced_att = extract_attributes_with_vl_model(
                        input_resv.data,  # 保持参数兼容性
                        opt.batch_size,
                        opt.vl_model,
                        opt.vl_processor,
                        opt
                    )

                    if enhanced_att is not None:
                        # 融合原始属性和大模型增强属性
                        input_attv_enhanced = opt.vl_model_alpha * enhanced_att + (
                                1 - opt.vl_model_alpha) * input_attv.data
                        input_attv = Variable(input_attv_enhanced)
                        print("✅ Qwen3-VL-8B属性增强完成")
                    else:
                        print("⚠️ Qwen3-VL-8B处理返回None，使用原始属性")
            except Exception as e:
                print(f"⚠️ Qwen3-VL-8B处理异常: {e}, 使用原始属性")

        print("🔄 计算GMVAE损失...")
        mm_total_loss, mm_losses = compute_multi_modal_loss(input_resv, input_attv, netG)

        # ZSL前向传播 - 计算类别相似度
        output = netG(input_resv, input_attv, input_labelv, train_gmvae=False)

        # 添加softmax归一化，因为NLLLoss需要log probabilities
        log_probs = F.log_softmax(output, dim=1)
        loss = cls_criterion(log_probs, input_labelv)  # 使用log_softmax的输出

        # 总损失 = ZSL分类损失 + 多模态损失
        total_combined_loss = loss + mm_total_loss * 0.1  # 多模态损失权重为0.1

        # 只进行一次反向传播
        total_combined_loss.backward()

        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)

        optimizerG.step()

        # 更新进度条
        pbar.update(1)
        postfix = {
            "Loss": f"{loss.item():.4f}",
            "MM_Loss": f"{mm_total_loss:.4f}" if mm_total_loss > 0 else "0.0000"
        }
        pbar.set_postfix(postfix)

    pbar.close()

    # 更新学习率
    scheduler.step()

    # 每个epoch结束后进行评估
    netG.eval()
    with torch.no_grad():
        # 生成未见类别的特征并评估
        try:
            print(f"Starting evaluation for epoch {epoch}")

            # 使用GMVAE生成特征
            print("Generating synthetic features...")
            syn_unseen_feature, syn_unseen_label = generate_syn_feature(netG, data.unseenclasses, data.attribute,
                                                                        opt.syn_num)

            print(f"Generated features shape: {syn_unseen_feature.shape}")
            print(f"Generated labels shape: {syn_unseen_label.shape}")

            # 检查数据有效性
            if torch.isnan(syn_unseen_feature).any():
                print("Warning: Generated features contain NaN values")
                syn_unseen_feature = torch.nan_to_num(syn_unseen_feature)

            train_X = torch.cat((data.train_feature, syn_unseen_feature), 0)
            train_Y = torch.cat((data.train_label, syn_unseen_label), 0)

            print(f"Combined training features shape: {train_X.shape}")
            print(f"Combined training labels shape: {train_Y.shape}")

            if opt.gzsl:
                # 使用classifier_with_class_norm中的CLASSIFIER，调整参数
                cls = classifier.CLASSIFIER(
                    train_X, train_Y, data, opt.nclass_all, opt.cuda,
                    _lr=0.001,  # 降低学习率
                    _beta1=0.5,
                    _nepoch=100,  # 增加训练轮数
                    _batch_size=128,  # 调整批次大小
                    generalized=True
                )

                # 格式化输出结果 - 按照要求格式
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
            print(f"Error in evaluation: {e}")
            import traceback

            traceback.print_exc()
            continue
    netG.train()